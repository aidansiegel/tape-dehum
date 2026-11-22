#!/usr/bin/env python3
# tape-denoise: optional light hiss reduction with DeepFilterNet, for speech, after tape-dehum.
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
"""Light tape-hiss reduction for speech, using DeepFilterNet3 (installed separately: pip install deepfilternet).

The attenuation limit caps how far noise is pushed down (12 dB by default), which keeps voices natural.
DeepFilterNet is trained on speech: it removes jingles and music too, so don't use it on music or ads.
Runs in 60 s pieces crossfaded over 2 s, so long tracks fit in memory. Input must be 48 kHz
(tape-dehum keeps the source rate; most video files are already 48 kHz).
"""
import argparse, os, signal, sys

__version__ = "1.0.0"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tape-denoise", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Light hiss reduction for speech, with DeepFilterNet3.",
        epilog="example:\n  tape-dehum news.mp4 news-dehum.wav && tape-denoise news-dehum.wav news-clean.flac")
    ap.add_argument("input", help="48 kHz WAV or FLAC (for example tape-dehum's output)")
    ap.add_argument("output", help=".wav or .flac, 16-bit")
    ap.add_argument("--atten", type=float, default=12, metavar="DB",
        help="most the noise is turned down, in dB (default 12; 20 is stronger, 0 means no limit)")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for torch (default 2)")
    ap.add_argument("-f", "--force", action="store_true", help="overwrite OUTPUT if it exists")
    ap.add_argument("--version", action="version", version="tape-denoise " + __version__)
    a = ap.parse_args(argv)
    if not os.path.isfile(a.input): ap.error("no such file: %s" % a.input)
    if os.path.exists(a.output) and not a.force: ap.error("%s exists; use --force to overwrite" % a.output)

    try:
        import numpy as np, soundfile as sf, torch
        from df.enhance import enhance, init_df
    except ImportError as e:
        log("tape-denoise: %s. Install DeepFilterNet first: pip install deepfilternet" % e); return 1
    torch.set_num_threads(a.threads)
    model, state, _ = init_df(log_level="ERROR")
    x, sr = sf.read(a.input, dtype="float32", always_2d=True)
    if sr != state.sr():
        log("tape-denoise: %s is %d Hz; DeepFilterNet needs %d Hz. Resample first, e.g. ffmpeg -i IN -ar 48000 OUT.wav"
            % (a.input, sr, state.sr())); return 1
    same = x.shape[1] == 1 or np.max(np.abs(x - x[:, :1])) < 1e-4
    chans = [x.mean(1)] if same else [x[:, i] for i in range(x.shape[1])]
    lim = a.atten or None; step, ov = 60 * sr, 2 * sr; fade = np.linspace(0, 1, ov, dtype=np.float32); done = []
    for ci, mono in enumerate(chans):
        out = np.zeros_like(mono); pos = 0; tail = None
        while pos < len(mono):
            s, e = max(0, pos - ov), min(len(mono), pos + step)
            y = enhance(model, state, torch.from_numpy(mono[s:e].copy())[None], atten_lim_db=lim)[0].numpy()
            if tail is not None: y[:ov] = tail * (1 - fade) + y[:ov] * fade
            out[s:e] = y; tail = y[-ov:] if e < len(mono) else None; pos = e
            log("  %s%d/%d s" % ("" if len(chans) == 1 else "ch%d " % (ci + 1), e // sr, len(mono) // sr))
        done.append(out)
    y = np.stack(done * x.shape[1] if same else done, 1)
    ext = os.path.splitext(a.output)[1].lower() or ".wav"
    part = os.path.join(os.path.dirname(os.path.abspath(a.output)), ".%s.partial%s" % (os.path.basename(a.output), ext))
    try:
        sf.write(part, y, sr, subtype="PCM_16"); os.replace(part, a.output)
    finally:
        if os.path.exists(part): os.remove(part)
    log("wrote %s" % a.output)
    return 0


def cli():
    """Entry point: Ctrl-C exits 130 with no traceback and no half-written output."""
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\ntape-denoise: interrupted; nothing written")
        sys.exit(130)


if __name__ == "__main__":
    cli()
