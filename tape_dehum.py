#!/usr/bin/env python3
# tape-dehum: subtract mains hum, video buzz and whine tones from tape transfers.
# Copyright (C) 2026 Aidan Siegel
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU Affero General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
"""Remove hum, buzz and whine from a tape transfer by subtracting each tone instead of filtering it.

Two passes:

1. Whole-track tones. Every narrow peak in the track's long-term spectrum (8 s FFTs, 0.125 Hz bins)
   that stands at least 5 dB over the background around it is a steady tone: mains hum, video
   field-rate buzz, a capture device's whine. Each one is subtracted with an amplitude-and-phase
   follower: demodulate at the tone, low-pass to about 2 Hz, remodulate, subtract. Only about 1 Hz
   around each tone is touched, so speech and music next to it are left alone. All the followers are
   one linear filter, so they run together as a single FFT convolution.

2. Drifting hum comb. Tape speed wanders, so the hum's fundamental drifts (59.92-60.00 Hz is typical
   on NTSC VHS). Over a long track its harmonics smear out of the average spectrum and pass 1 misses
   them. Pass 2 finds the fundamental every 15 s from the harmonics themselves, subtracts that window's
   whole comb the same way, and crossfades neighbouring windows.

Then rumble under --highpass and the TV line-scan whine above --lowpass are filtered away.
"""
import argparse, json, os, shutil, signal, subprocess, sys, tempfile

__version__ = "1.0.0"

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130


def log(msg=""):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- analysis

def spectrum_db(x, sr, secs=8):
    """Average power spectrum of 'secs'-long Hann frames, 50% overlap, in dB; and the frequency axis."""
    import numpy as np
    n = int(sr * secs); f = np.fft.rfftfreq(n, 1 / sr); w = np.hanning(n); P = np.zeros(len(f))
    if len(x) < n: x = np.concatenate([x, np.zeros(n - len(x), x.dtype)])
    for i in range(0, len(x) - n + 1, n // 2):
        P += np.abs(np.fft.rfft(x[i:i + n] * w)) ** 2
    return f, 10 * np.log10(P + 1e-20)


def background(db, secs=8):
    """Running median over +-20 Hz: the noise floor each peak is measured against."""
    from scipy.ndimage import median_filter
    return median_filter(db, size=int(40 * secs) | 1)


def refine(f, db, i, secs=8):
    """Parabolic interpolation of a spectral peak's frequency."""
    a, b, c = db[i - 1], db[i], db[i + 1]
    den = a - 2 * b + c
    return f[i] + (0.5 * (a - c) / den / secs if den else 0.0)


def find_tones(x, sr, lo, hi, thr=5.0, secs=8):
    """Pass 1: every narrow peak between lo and hi Hz standing thr dB or more over its background."""
    f, db = spectrum_db(x, sr, secs); base = background(db, secs); out = []
    for i in range(3, len(f) - 3):
        if lo < f[i] < hi and db[i] - base[i] >= thr and db[i] == db[i - 3:i + 4].max():
            out.append((refine(f, db, i, secs), db[i] - base[i]))
    return out


def hum_comb(seg, sr, mains, hi, thr=4.0, always=12, secs=8):
    """Pass 2, one window: the drifting fundamental near 'mains' Hz, found by summing harmonics 2..16
    (the fundamental itself may already be gone after pass 1), then each harmonic that shows as a peak
    in this window, plus harmonics 1..'always' whether they show or not."""
    import numpy as np
    f, db = spectrum_db(seg, sr, secs); base = background(db, secs); ex = np.maximum(db - base, 0)
    cands = np.arange(mains * 0.9975, mains * 1.0017, 0.002)
    score = [sum(ex[int(round(k * f0 * secs)) - 1:int(round(k * f0 * secs)) + 2].max() for k in range(2, 17))
             for f0 in cands]
    f0 = float(cands[int(np.argmax(score))]); out = []
    for k in range(1, int(hi / f0) + 1):
        c = k * f0; idx = np.where(np.abs(f - c) <= 0.15 + 0.01 * k)[0]
        if len(idx) == 0: continue
        j = idx[np.argmax(db[idx])]
        if 2 < j < len(f) - 2 and db[j] - base[j] >= thr and db[j] == db[j - 2:j + 3].max():
            out.append(refine(f, db, j, secs))
        elif k <= always:
            out.append(c)
    out.sort(); tones = []
    for q in out:
        if not tones or q - tones[-1] > 0.25: tones.append(q)
    return f0, tones


# ---------------------------------------------------------------- subtraction

def tone_kernel(sr, freqs, L):
    """All the followers as one FIR: hum = x * g, g(m) = 2 b(m) sum_k cos(w_k m), with b the triangle
    (box * box / L^2) that the two-pass moving-average low-pass makes. 2L-1 taps."""
    import numpy as np
    b = np.convolve(np.ones(L), np.ones(L)) / L ** 2
    m = np.arange(-(L - 1), L); c = np.zeros(len(m)); fr = np.asarray(freqs, float)
    for i in range(0, len(fr), 64):
        c += np.cos(2 * np.pi * np.outer(fr[i:i + 64], m) / sr).sum(0)
    return 2 * b * c


def fftconv_same(a, g):
    """FFT convolution, centred ('same' length as a)."""
    import numpy as np
    n = len(a) + len(g) - 1; N = 1 << (n - 1).bit_length()
    full = np.fft.irfft(np.fft.rfft(a, N) * np.fft.rfft(g, N), N)[:n]
    o = (len(g) - 1) // 2
    return full[o:o + len(a)]


def subtract_fixed(x, sr, freqs, L, step_s=60, label=""):
    """Pass 1 subtraction, in step_s pieces with the kernel's half-length of context either side."""
    import numpy as np
    if not len(freqs): return x
    g = tone_kernel(sr, freqs, L); h = L - 1; y = np.empty_like(x); step = step_s * sr
    for s in range(0, len(x), step):
        e = min(len(x), s + step); a, b = max(0, s - h), min(len(x), e + h)
        y[s:e] = x[s:e] - fftconv_same(x[a:b].astype(np.float64), g)[s - a:s - a + (e - s)]
        log("  %stones: %d/%d s" % (label, e // sr, len(x) // sr))
    return y


def subtract_drifting(x, sr, mains, hi, L, hop_s=15, label=""):
    """Pass 2: per-window hum combs, 2*hop_s windows on a hop_s grid, triangle-weighted so they sum to 1."""
    import numpy as np
    H = hop_s * sr; h = L - 1; hum = np.zeros(len(x), np.float32); trace = []
    for c in range(0, len(x) + H, H):
        a, b = max(0, c - H), min(len(x), c + H)
        if b <= a: continue
        m0 = max(0, min(a, len(x) - 2 * H))                   # windows at the ends still measure 2*hop_s
        f0, tones = hum_comb(x[m0:m0 + 2 * H], sr, mains, hi)
        aa, bb = max(0, a - h), min(len(x), b + h)
        est = fftconv_same(x[aa:bb].astype(np.float64), tone_kernel(sr, tones, L))[a - aa:a - aa + (b - a)]
        hum[a:b] += np.clip(1 - np.abs(np.arange(a, b) - c) / H, 0, 1) * est
        trace.append((c / sr, f0, len(tones)))
        log("  %shum comb: %d/%d s, fundamental %.3f Hz, %d harmonics" % (label, min(c, len(x)) // sr, len(x) // sr, f0, len(tones)))
    return (x - hum).astype(np.float32), trace


# ---------------------------------------------------------------- I/O

def need_ffmpeg():
    if not shutil.which("ffmpeg"):
        log("tape-dehum: ffmpeg not found. Install it (macOS: brew install ffmpeg; Debian/Ubuntu: apt install ffmpeg).")
        sys.exit(EXIT_FAIL)


def decode(src, start, duration, tmp):
    """Decode to 32-bit float WAV at the source's own rate and channel count."""
    import soundfile as sf
    dst = os.path.join(tmp, "in.wav"); c = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if start is not None: c += ["-ss", str(start)]
    c += ["-i", src]
    if duration is not None: c += ["-t", str(duration)]
    r = subprocess.run(c + ["-map", "0:a:0", "-c:a", "pcm_f32le", "-f", "wav", dst], capture_output=True, text=True)
    if r.returncode:
        log("tape-dehum: ffmpeg could not read %s: %s" % (src, r.stderr.strip() or "no audio stream?"))
        sys.exit(EXIT_FAIL)
    x, sr = sf.read(dst, dtype="float32", always_2d=True)
    return x, sr


def encode(y, sr, dst, highpass, lowpass, tmp):
    """Filter and write, via a temporary file renamed into place, so an interrupted run leaves nothing half-written."""
    import soundfile as sf
    mid = os.path.join(tmp, "out.wav"); sf.write(mid, y, sr, subtype="FLOAT")
    af = []
    if highpass: af += ["highpass=f=%g:p=2" % highpass] * 2           # 4th order
    if lowpass: af += ["lowpass=f=%g:p=2" % lowpass] * 3              # 6th order: the scan-whine comb is close above
    ext = os.path.splitext(dst)[1].lower()
    codec = {".wav": ["-c:a", "pcm_s16le"], ".flac": ["-c:a", "flac", "-sample_fmt", "s16"]}.get(ext, [])
    part = os.path.join(os.path.dirname(os.path.abspath(dst)), ".%s.partial%s" % (os.path.basename(dst), ext))
    c = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", mid]
    if af: c += ["-af", ",".join(af)]
    try:
        r = subprocess.run(c + codec + [part], capture_output=True, text=True)
        if r.returncode:
            log("tape-dehum: could not write %s: %s" % (dst, r.stderr.strip())); sys.exit(EXIT_FAIL)
        os.replace(part, dst)
    finally:
        if os.path.exists(part): os.remove(part)


# ---------------------------------------------------------------- main

EXAMPLES = """examples:
  tape-dehum newscast.mp4 newscast-clean.flac
  tape-dehum --list newscast.mp4                    show the tones it would remove, change nothing
  tape-dehum --music commercials.mkv ads-clean.wav  spare sustained music notes under 1 kHz
  tape-dehum --mains 50 pal-tape.wav clean.wav      50 Hz countries (PAL)
  tape-dehum --start 600 --duration 30 tape.mp4 sample.wav
"""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tape-dehum", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Subtract mains hum, video buzz and whine tones from a tape transfer.", epilog=EXAMPLES)
    ap.add_argument("input", help="any audio or video file ffmpeg can read (first audio stream is used)")
    ap.add_argument("output", nargs="?", help=".wav or .flac (16-bit), or any extension ffmpeg can write")
    ap.add_argument("--list", action="store_true", help="print the tones found and exit; writes nothing")
    ap.add_argument("--json", action="store_true", help="with --list, print JSON")
    ap.add_argument("--mains", type=float, default=60, choices=[50, 60], help="mains frequency (default 60)")
    ap.add_argument("--music", action="store_true",
        help="keep pass-1 tones under 1 kHz only if they sit on a hum harmonic, so held music notes survive")
    ap.add_argument("--no-drift", action="store_true", help="skip pass 2 (the drifting hum comb)")
    ap.add_argument("--threshold", type=float, default=5.0, metavar="DB", help="pass-1 peak threshold (default 5)")
    ap.add_argument("--highpass", type=float, default=45, metavar="HZ", help="rumble cut, 0 for none (default 45)")
    ap.add_argument("--lowpass", type=float, default=14800, metavar="HZ",
        help="cut under the TV line-scan whine (15.7 kHz NTSC, 15.6 kHz PAL), 0 for none (default 14800)")
    ap.add_argument("--start", type=float, metavar="S", help="process from S seconds in (tones are still found over the whole track, so a piece matches a full run)")
    ap.add_argument("--duration", type=float, metavar="S", help="process S seconds")
    ap.add_argument("-f", "--force", action="store_true", help="overwrite OUTPUT if it exists")
    ap.add_argument("--version", action="version", version="tape-dehum " + __version__)
    a = ap.parse_args(argv)

    if not os.path.isfile(a.input):
        ap.error("no such file: %s" % a.input)
    if not a.list:
        if not a.output: ap.error("give an OUTPUT file, or use --list")
        if os.path.abspath(a.output) == os.path.abspath(a.input): ap.error("OUTPUT would overwrite INPUT")
        if os.path.exists(a.output) and not a.force: ap.error("%s exists; use --force to overwrite" % a.output)
    need_ffmpeg()
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="tape-dehum-") as tmp:
        x, sr = decode(a.input, None, None, tmp)          # the whole track: tones are always found over all of it
        nyq = sr / 2; hi = min(a.lowpass or nyq * 0.95, nyq * 0.95); lo = max(a.highpass or 20, 20)
        rms = lambda v: float(np.sqrt(np.mean(np.square(v, dtype=np.float64))) + 1e-12)
        same = x.shape[1] == 1 or all(rms(x[:, i] - x[:, 0]) < 0.003 * rms(x[:, 0]) for i in range(1, x.shape[1]))  # mono tape captured as stereo (within -50 dB)
        log("tape-dehum: %s, %.0f s, %d Hz, %d channel%s%s" % (os.path.basename(a.input), len(x) / sr, sr, x.shape[1],
            "" if x.shape[1] == 1 else "s", " (identical, processed once)" if same and x.shape[1] > 1 else ""))
        tones = find_tones(x.mean(1), sr, lo, hi, a.threshold)
        pad = 3.0 if a.start is not None or a.duration is not None else 0.0   # run-in so the followers settle
        if pad:                                           # then only the asked-for piece (plus run-in) is processed
            s0 = max(0, int(((a.start or 0.0) - pad) * sr))
            e0 = len(x) if a.duration is None else min(len(x), int(((a.start or 0.0) + a.duration + pad) * sr))
            if s0 >= len(x): ap.error("--start is past the end (%.0f s)" % (len(x) / sr))
            x = x[s0:e0]
        mono = x.mean(1)
        if a.music:
            onhum = lambda q: any(abs(q - k * b) < 0.4 for b in (a.mains, a.mains * 1000 / 1001) for k in range(1, 18))
            tones = [t for t in tones if t[0] >= 1000 or onhum(t[0])]
        log("pass 1: %d steady tones" % len(tones))

        if a.list:
            top = sorted(tones, key=lambda t: -t[1])
            if a.json:
                print(json.dumps({"input": a.input, "sample_rate": sr,
                                  "tones": [{"hz": round(f, 3), "db_over_background": round(d, 1)} for f, d in top]}, indent=1))
            else:
                for f, d in top: print("%10.3f Hz  +%.1f dB" % (f, d))
            return EXIT_OK

        L = sr // 2; freqs = [t[0] for t in tones]
        chans = [mono] if same else [x[:, i] for i in range(x.shape[1])]; done = []
        for i, ch in enumerate(chans):
            label = "" if len(chans) == 1 else "ch%d " % (i + 1)
            y = subtract_fixed(ch.astype(np.float32), sr, freqs, L, label=label)
            if not a.no_drift:
                y, _ = subtract_drifting(y, sr, a.mains, hi, L, label=label)
            done.append(y)
        y = np.stack(done * x.shape[1] if same else done, 1)
        if pad:
            k = int((a.start or 0.0) * sr) - s0; n = int(a.duration * sr) if a.duration is not None else len(y)
            y = y[k:k + n]
        encode(y, sr, a.output, a.highpass, a.lowpass, tmp)
        log("wrote %s" % a.output)
    return EXIT_OK


def cli():
    """Entry point: Ctrl-C exits 130 with no traceback and no half-written output."""
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\ntape-dehum: interrupted; nothing written")
        sys.exit(130)


if __name__ == "__main__":
    cli()
