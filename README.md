# tape-dehum

Removes hum, buzz and whine from tape transfers by **subtracting** each tone instead of filtering it.
It was built for off-air VHS recordings, but it works on any recording where the noise is made of steady
tones: power-line hum, video buzz, a capture card's whine.

Ordinary noise reduction (spectral gating, `afftdn`, most "denoise" plugins) estimates a noise
profile and turns down every frequency band where noise lives. On tape hum that leaves the voices sounding
watery and still doesn't remove the hum. `tape-dehum` treats hum as what it is: a set of exact
pitches. It measures each one, rebuilds it with its own changing loudness and phase, and subtracts it.
About 1 Hz around each tone is affected, so speech and music right next to it pass through untouched.

## What it removes

On a typical NTSC off-air VHS tape:

- 60 Hz mains hum and its harmonics (120, 180, 240 Hz and up), the part you hear as "hum"
- video field-rate buzz: a dense comb of tones across the whole band
- capture-device whine (8 kHz on many USB capture sticks) and its sidebands
- the TV's 15.7 kHz horizontal-scan whine (filtered off with a steep low-pass, since it is a wide comb)
- rumble under 45 Hz

Use `--mains 50` for PAL tapes and 50 Hz countries.

## Install

Needs Python 3.9 or newer and [ffmpeg](https://ffmpeg.org/) on your `PATH`.

```bash
pip install .
```

Or run it without installing: `pip install -r requirements.txt`, then `python3 tape_dehum.py ...`.

## Use

```bash
tape-dehum newscast.mp4 newscast-clean.flac
```

The input can be any audio or video file ffmpeg can read; the first audio stream is used. The output
keeps the source's sample rate and channels, written as 16-bit `.wav` or `.flac`. A mono tape captured as
two identical channels is processed once and written back as two.

```bash
tape-dehum --list newscast.mp4                    # show the tones it found; writes nothing
tape-dehum --list --json newscast.mp4             # the same, as JSON
tape-dehum --music commercials.mkv ads-clean.wav  # spare sustained music notes under 1 kHz
tape-dehum --start 600 --duration 30 tape.mp4 sample.wav   # try a 30 s piece first
```

| Option | What it does |
|---|---|
| `--mains 50\|60` | Mains frequency. Default 60. |
| `--music` | For music, jingles and commercials. Held notes under 1 kHz can look like steady tones over a long track; this keeps only the low tones that sit on a hum harmonic. |
| `--no-drift` | Skip the second pass (the drifting hum comb). |
| `--threshold DB` | How far a peak must stand above the background to count as a tone. Default 5. |
| `--highpass HZ` | Rumble cut. Default 45; 0 turns it off. |
| `--lowpass HZ` | Cut under the TV scan whine. Default 14800; 0 turns it off. |
| `--start S`, `--duration S` | Process part of the file. |
| `-f`, `--force` | Overwrite the output file. |

Progress goes to stderr. With `--list` the tone list goes to stdout. Exit codes: 0 success,
1 failure (ffmpeg missing or unreadable input), 2 usage error, 130 interrupted. An interrupted run
leaves no half-written output.

The whole track is held in memory as 32-bit floats: about 11 MB per minute per channel at 48 kHz,
plus working space. A 30-minute mono tape needs about 1.5 GB.

## Optional: light hiss reduction for speech

After the hum is gone, what's left is tape hiss. `tape_denoise.py` runs
[DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) with a cap on how far it turns noise down
(12 dB by default), which keeps voices natural:

```bash
pip install ".[denoise]"
tape-dehum news.mp4 news-dehum.wav
tape-denoise news-dehum.wav news-clean.flac          # --atten 20 for stronger
```

DeepFilterNet is trained on speech: it removes music and jingles along with the hiss. Use it on talk,
not on music or commercial breaks. It needs 48 kHz input. DeepFilterNet is a separate project under its own
license (MIT or Apache-2.0); it is not included here.

## How it works

**Pass 1: whole-track tones.** The track's long-term spectrum is measured with 8-second FFTs
(0.125 Hz resolution). Every narrow peak that stands at least 5 dB above the median level within
±20 Hz of it is treated as a steady tone. A 30-minute VHS newscast typically has 300 to 1,100 of them.

Each tone is removed with an amplitude-and-phase follower:

1. demodulate: multiply the signal by e<sup>−iωt</sup> at the tone's frequency, so the tone becomes
   a slowly varying number near 0 Hz;
2. low-pass: a two-pass moving average half a second long (a triangle filter about 2 Hz wide)
   keeps only that slowly varying number, which tracks the tone's loudness and phase as they wander;
3. remodulate and subtract.

That chain is linear, so the followers for every tone add up to one FIR filter,
g(m) = 2·b(m)·Σ<sub>k</sub> cos(ω<sub>k</sub>m), where b is the triangle. The tool runs that as a single
FFT convolution, so 1,000 tones cost about the same as one.

**Pass 2: the drifting hum comb.** Tape speed wanders, and the hum's pitch wanders with it.
On NTSC VHS it moves between about 59.92 and 60.00 Hz. That is a small change at 60 Hz, but by
the 10th harmonic it is almost 1 Hz. Averaged over a long track, the harmonics smear out and fall below
pass 1's threshold, while you can still clearly hear them. So every 15 seconds the tool:

1. finds that window's fundamental from its harmonics: the candidate pitch between 59.85 and 60.10 Hz
   whose harmonics 2 to 16 stand highest above the background (the fundamental itself may already have been
   removed in pass 1);
2. lists every harmonic of that pitch that shows as a peak in the window, plus the first 12 harmonics
   regardless;
3. subtracts that comb with the same followers, and crossfades each 30-second window into its neighbors
   with triangle weights that sum to 1.

**Then filters.** A 4th-order high-pass at 45 Hz for rumble, and a 6th-order low-pass at 14.8 kHz,
under the TV's 15.7 kHz line-scan comb.

## Checking results

Measure hum one harmonic at a time against the background right around it, not with an overall level:
broadband level is dominated by speech and hides hum entirely. `--list` on the cleaned file should
show no comb at multiples of 60 Hz.

## License

Copyright (C) 2026 Aidan Siegel

This program is free software under the [GNU Affero General Public License v3.0](LICENSE) or later.
If you run a modified version of it as a network service, the AGPL requires you to offer that version's
source code to its users.
