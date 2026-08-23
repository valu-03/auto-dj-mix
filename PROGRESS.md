# Auto DJ Mix — Progress

## Layout (tidied 2026-08-22)
`autodj/` package · `musica/` + `musica2/` input sets · `output/` rendered mixes
· `demos/` teaching artefacts (click tracks, transition A/Bs, spectrogram)
· `cache/` analysis JSON + separation models + stems. See `README.md`.

## Two finished mixes
| | tracks | length | BPM | stretch | violations | transitions |
|---|---|---|---|---|---|---|
| `output/mix.wav` | musica/ | 34.6 min | 134.9 | 2.27% | 2 | slam, bass, eq, echo, bass, bass, sweep, bass, echo |
| `output/mix2.wav` | musica2/ | 38.0 min | 124.0 | 0.71% | 0 | eq, bass, eq, bass, eq, bass, bass, echo, echo |

Set 2 was chosen from the 686-track analysed pool as a legal chain (Camelot ≤1,
BPM jump ≤3) with a same-key run penalty so it moves round the wheel — without
that penalty the search finds the degenerate optimum of ten tracks all in 8A.

## Phase 5 — stems (IN PROGRESS)
Installed and CUDA verified working; **separation itself not yet run**.
`autodj/stems/cuda.py` + `separate.py` written. Demucs model (321 MB) is
downloaded in `cache/models`.


Plan: `C:\Users\Valentin\.claude\plans\i-want-to-build-velvety-peach.md` (28 lessons, 6 phases)

## Environment (verified)

- Python 3.13.0, numpy 2.4.6, scipy 1.17.1, numba 0.65.1, PyQt6 6.11.0, torch 2.13.0+cpu
- librosa 1.0.0, soundfile 0.14.0 (libsndfile 1.2.2), pyloudnorm 0.2.0, mutagen 1.48.1
- **MP3 reads and writes with no ffmpeg** (libsndfile 1.2.2)
- GPU: RTX 3060 12 GB, compute 8.6. Torch is the CPU build. GPU only matters for stem
  separation (Lesson 20) — install `onnxruntime-gpu`, NOT a CUDA torch (cu128 tops out at
  torch 2.11 and would downgrade us). librosa has no GPU path; use the 12 CPU cores instead.
- librosa in a `ProcessPoolExecutor`: **4 workers max** — 10 gave `BrokenProcessPool`.

## Test set — `musica\` (10 tracks, chosen by measured BPM + key)

Picked from `Downloads\Disco.80s.90s.The.Greatest.Hits` (3,007 MP3s, 23.5 GB).

| # | BPM | Camelot | Track |
|---|---|---|---|
| 1 | 136.01 | 6B | Whigfield — Another Day |
| 2 | 136.14 | 6A | 2 Brothers On The 4th Floor — Dance With Me |
| 3 | 136.05 | 7A | Ice MC — Thing About The Way |
| 4 | 136.04 | 8A | Real McCoy — One More Time |
| 5 | 135.98 | 7A | Maxx — No More |
| 6 | 135.91 | 6A | Dr Alban — Let The Beat Go On |
| 7 | 136.07 | 6B | La Bouche — In Your Life (energy peak) |
| 8 | 135.86 | 3B | Culture Beat — Inside Out |
| 9 | 135.96 | 3B | Jam & Spoon — I Pull My Gun Once |
| 10 | 135.96 | 3A | Fun Factory — Doh Wah Diddy |

BPM spread 0.33 → max stretch **0.16%** (inaudible). 8 of 9 transitions Camelot-legal;
7→8 (6B→3B) is a deliberate clash, reserved for testing filter/echo transitions.

## Lessons

### L1 — Audio foundations ✅ DONE
`requirements.txt`, `autodj/__init__.py`, `autodj/audio.py`
(`describe`, `load`, `to_samples`, `to_seconds`). Convention fixed: **audio arrays are
`(channels, frames)`, always float32, always 2-D.**

Verified: mono+resample to 22050 gives exactly `duration × 22050` frames; C-contiguous;
seconds↔samples roundtrip exact.

**Two real-world findings from the test set — both need handling in L2:**
1. **Header duration lies.** 3 of 10 tracks decode shorter than `sf.info().frames` says
   (−0.047 s, −0.125 s, −0.364 s). The Xing/LAME header length is an *estimate*.
   → Rule: never compute a position from header duration. Use the decoded array length.
2. **Files are damaged.** `10 - Fun Factory` has a bad MPEG frame at 211.0 s of 212.1 s;
   `sf.read` raises `LibsndfileError` and we lose the whole track over 1.1 s of garbage.
   → `load` must fall back to block reading and keep what decodes.

### L2 — Slicing, writing, surviving dirty files ✅ DONE
Added to `audio.py`: `_read_forgiving` (block-read fallback), `duration`, `clip`, `save`.

Verified: **all 10 tracks now load**, Fun Factory recovered to 211.02 s instead of crashing.
Clip exactly 8.000 s. 16-bit roundtrip error 0.000031 = one LSB. `clip()` survives negative
start / past-end / reversed range. `save()` peak protection 1.899 → 1.0.

Measured drift (real − header): Ice MC −0.047 s, Dr Alban −0.125 s, Jam & Spoon −0.364 s,
Fun Factory −1.029 s. **Always use `audio.duration(array, sr)`, never `describe()["duration"]`,
for anything positional.**

### L3 — STFT and the spectrogram ⬅ CURRENT
New file `autodj/spectral.py`: `as_mono_1d`, `stft`, `magnitude`, `to_db`, `frame_times`,
`bin_freqs`, `band_energy`. Constants `N_FFT=2048`, `HOP=512` — **must stay identical across
every module** or frame indices stop meaning the same time.

`band_energy` is the workhorse: it feeds L7 downbeat detection (kick = LOW_BAND peak),
L10 breakdown detection (LOW_BAND drops out), and L12's EQ crossovers.

### L4-L7 — library, onsets, tempo, beatgrid ✅ DONE
`library.py`, `analysis/onsets.py`, `analysis/beats.py`.

**Beatgrid took three attempts. The failures are the lesson:**
1. Folding into phase bins — 16x biased toward integer-frame periods
   (19.000 → 25/96 bins, max 56.68; 19.141 → 96/96 bins, max 3.54). Also scored
   136 and 272 identically: folding is blind to octaves.
2. Contrast (on-grid minus midpoints) — eurodance basslines play **offbeat
   eighths**, so midpoints carry real low-band energy. Chose ~90 BPM (2/3 of
   136) on 6 of 10 tracks.
3. **What works:** autocorrelation + octave prior picks *which* tempo (right on
   all 10, by a wide margin); grid fitting refines BPM/phase inside ±2.5% where
   no octave ambiguity remains. Sample by `np.interp`, never bucket.

Validated vs an independent hop=128 sweep: **max error 0.011 BPM, median 0.005.**
Two-stage refine is required — 0.25 BPM error = one full beat of drift over 4 min.

**REAL BPMs (the earlier "all 136" was librosa tempogram quantization, lag 19):**
133.27 / 135.01 / 132.97 / 133.00 / 137.95 / 136.00 / 134.72 / 135.86 / 138.01 / 133.00
→ spread 132.98-138.00 = **3.8%**, not the 0.16% first claimed. Still under the 6% cap.

### L8-L10 — key, energy, structure ✅ DONE
`analysis/key.py`, `analysis/energy.py`, `analysis/structure.py`, `analysis/track.py`.

- **HPSS removed.** Cost 12.4 s of 14.5 s per track and *reduced* the key
  confidence gap on 8 of 10 tracks. Analysis: **14.5 s → 3.1 s/track.**
  (3,007 tracks ≈ 2.6 h single-threaded, ~40 min on 4 workers.)
- **Key validated 20/24 on synthetic progressions of known key** (minor 12/12,
  major 8/12). An earlier "all minors wrong" result was a broken *test*:
  i-VI-III-VII has the same pitch-class set as its relative major. A realistic
  i-iv-**V**-i is unambiguous and the detector handles it.
- `BASS_WEIGHT = 0.0`: bass-register tonic voting scored 20/20/20/19 at
  0.0/0.25/0.35/0.5 — no gain, costs a second CQT. Mechanism kept for later.
- **Cross-track energy fix:** `energy_curve` is normalised per track (right for
  finding *that* track's drops, useless for comparing tracks — everything scored
  0.86-0.92). Now storing absolute `loudness` (LUFS), `density`, `low_level`,
  `brightness`; the planner z-scores them across the actual set.
  Spreads: LUFS 8.4 dB, density 0.67-0.91, low_level 2.4x.
- Cue points 10/10 self-consistent (intro ≤ first_full ≤ last_full < outro).
- Weak key flagged: Jam & Spoon, gap 0.039.

### L11-L16 — the mixing engine ✅ DONE
`dsp/stretch.py`, `dsp/filters.py`, `dsp/automation.py`, `dsp/effects.py`,
`dsp/master.py`, `transitions.py`.

- Band split reconstructs to **5.55e-17**. Mid is defined as `x - low - high`,
  not a bandpass — three independent filters would each add phase and the sum
  would comb-filter at the crossovers.
- **`sosfiltfilt` doubles the magnitude response** (forward + backward). A shelf
  designed for +2.5 dB measured +4.48 dB. All mastering EQ gains are now
  designed at **half** the requested value.
- **Bass handover, not crossfade.** An ease-in/ease-out pair leaves both
  basslines at 0.75 mid-transition. Outgoing low must hit zero *before*
  incoming low rises. Now 0.0000 overlap on all 6 transitions.
- Equal-power crossfade holds power at 1.000; a linear fade dips **-3.01 dB**.
- Limiter: running-min window must be ~1/4 of the release, not the full
  release — the full window ducked 300 ms per transient (69% GR, -16 LUFS out).
- **A loudness target you cannot reach without heavy limiting makes the result
  quieter, not louder.** -9 LUFS asked → -16.0 LUFS delivered. Target is -14
  with a `robust_peak` guard (99.9th percentile of block maxima, never the
  absolute max — one freak transient must not set the gain of a 35-min mix).
- Compressor: `maximum_filter1d` for attack, one-pole `lfilter` for release —
  asymmetric envelope without a per-sample Python loop. Measure gain reduction
  **before** makeup or it reports 0.00.

### L17-L19 — planner, setlist, render ✅ DONE
`planner.py`, `render.py`, `export.py`, `profile.py`, `playbook.py`,
`selector.py`, `main.py`.

Runs end to end: `python main.py --input musica --output mix.wav`
→ 34.6 min mix, max stretch 2.27%, peak 0.97, plus `.txt` tracklist, `.cue`,
and three JSON schemas (`mix.json`, `mix_dsp.json`, `mix_playbook.json`).

**DJ rules implemented as hard constraints** (penalty 50, not exclusion — a bad
pool must still produce *a* set): Camelot ≤1 (+2 "energy flash" only at 55-85%
through the set), BPM jump ≤3, phrases snapped to 4/8/16/32. Genre playbooks
pick the phrase window and legal moves per style.

### Reverted on request
The genre-playbook layer and everything built after it were removed — the mix
they produced was worse (transitions homogenised: first all `bass_swap`, then
4 of 9 `vocal_slam_drop`). Current output is the varied-transition set:
`vocal_slam_drop, bass_swap, eq_blend, echo_out, bass_swap, bass_swap,
filter_sweep, bass_swap, echo_out`.

**Removed:** `playbook.py`, `selector.py`, `--genre`, `--select`, the strategy
override in `render()`, `mix_playbook.json`, and mastering profiles being
*applied* to the audio (`profile.py` still computes them for the JSON).

**Kept from the first four specs:** eurodance transitions
(`vocal_slam_drop`, `euro_rap_breakout`), hard rules (Camelot / BPM≤3 /
phrase 4-8-16-32), arc presets, `profile.py`, `mix.json` + `mix_dsp.json`,
tonal balance + intro character, and the `band_power` fix.

**Also kept deliberately** (inert unless used): MP3 output via
`audio.save()` — `--output mix.mp3` gives ~44 MB instead of 349 MB, which is
what makes a car-playable file. Plus `filters.shelf_sos` / `peaking_sos` /
`apply_static_eq` and `master.compress`, which nothing calls now. Say the word
and they go too.

**Cue points VALIDATED BY LISTENING (2026-08-22).** This was the weakest link:
`structure.py` was only ever checked for self-consistency
(`intro ≤ first_full ≤ last_full < outro`), which proves the numbers don't
contradict each other and nothing about whether they're *right*. Confirmed by
ear that transitions land where the music wants them. The 8-bar phrase snapping
and the 0.62-of-p75 "full" threshold are good enough — do not re-tune them
without a reason.

**Known open issues:**
1. The `musica/` pool is **not harmonically closed** — keys 8A,7A,6A,6A,6B,6B,7B,
   3B,3B,3A. The 3-cluster is 3+ steps from the rest, so *every* ordering breaks
   a rule. BPM span 5.04 also forces one jump >3. `selector.py` fixes this by
   searching the full library for a legal chain instead of ordering a fixed set.
2. **All ID3 years read 2013** (compilation dates) — era detection via tags is
   useless on this library; vintage shelf never triggers.
3. `vocal_forward_intro` fires on only 1/10 tracks; threshold (mid/low > 1.6)
   too strict, so eurodance slam drops rarely get chosen.
4. -16.7 LUFS vs the -11 club target. Source peaks at ~0 dBFS while sitting at
   -11..-19.6 LUFS; the dB simply are not there without multiband compression.
5. Not built: wordplay cut, half/double-time. (Stems L20-22 done — see Phase 5.)
· L9 energy · L10 structure · L11 stretch · L12 EQ · L13 automation · L14 bass swap
· L15 sweeps/echo · L16 LUFS+limiter · L17 cost matrix · L18 setlist · L19 render
· L20 stems · L21 stem swaps · L22 mashups · L23–28 PyQt6 GUI

## Notes to self

- L7 needs **outlier rejection (RANSAC-style)**, not naive least squares. A quick LS fit on
  the test set gave 90–130 ms residuals from beat-tracker jitter (Culture Beat: 11 ms).
  Tempo was still right — 14 independent tracks landed within 0.33 BPM.
- librosa's reported tempo quantizes to coarse tempogram bins (all 10 tracks read exactly
  136.00). Precise BPM must come from fitting the beat *times*, not `beat_track`'s tempo.

---

## Phase 5 — stems (L20–L22) — DONE

**L20 separation.** `htdemucs_ft` (4-stem Demucs), FLAC cached under
`cache/stems/<fingerprint>/`, evicted oldest-first at 20 GB. **~60 s/track** on
the RTX 3060.

**L21 stem bass swap.** `mashup.stem_bass_swap` — same handover shape as the
frequency version (outgoing bass reaches zero *before* incoming rises), applied
to the bass stem alone. Verified: max simultaneous bass gain **0.000000**,
7 ms silent gap.

**L22 mashup.** `mashup.acapella_over` — A's vocal over B's instrumental, both
stretched to one mix BPM from their own fitted grids, both cut on downbeats via
`bar_slice`, bed's own vocal removed, vocal loudness-matched to the bed then
ducked 3 dB. Demo: Whigfield 6B vocal over Dr Alban 6A bed, +2.05% stretch.

**Separation quality.** Stems reconstruct the source at 0.984 correlation
(−21 dB residual, no gain or sample offset — that is just Demucs, which is
trained per-stem and not constrained to sum). Band content confirms the split:

| stem | RMS dB | low | mid | high |
|---|---|---|---|---|
| vocals | −25.2 | 0.3% | 80.4% | 19.3% |
| drums | −21.0 | 80.8% | 12.1% | 7.0% |
| bass | −20.4 | 96.1% | 3.9% | 0.0% |
| other | −28.4 | 5.1% | 91.0% | 3.9% |

**What stems actually buy.** The frequency bass swap cuts at 200 Hz, which is
not a line between instruments — it takes the kick with it. Over the first half
of the same 16 bars: dry −14.5 dB in the low band, frequency swap −21.8,
stem swap −18.2. The stem swap keeps **3.6 dB more low end** because the kick
lives in the drums stem and is never touched. (Caveat: the two paths use
different gain laws for the non-bass content, so only the low-band number is a
controlled comparison — the mid/high columns are not.)

### Four bugs, all silent

1. **Wrong GPU stack.** `cuda_status()` proved a working onnxruntime CUDA
   session, so separation was expected to be fast. But backend is chosen by
   file extension: `.onnx` → onnxruntime, `.yaml`/`.th` → torch. The default
   model is Demucs = torch, and torch was a `+cpu` build. onnxruntime was never
   asked to do anything. One 4-minute track ran past **19 minutes** without
   finishing; 57 s once torch was the cu126 build. `htdemucs_ft` makes it worse
   — the *fine-tuned* variant is a bag of four checkpoints, 4× a plain
   `htdemucs`. Fixed: `device_status(model)` checks the backend that matters,
   and `separate()` refuses to start without a GPU.
2. **The two stacks cannot share PATH.** `enable_cuda()` prepends the
   `nvidia-*-cu12` wheels for onnxruntime; torch ships its own copies of the
   same libraries in `torch/lib`. With the wheels first, torch failed at import
   — `WinError 127 ... cudnn_cnn64_9.dll` — from a torch that imported fine in
   a clean process. Fixed: PATH surgery scoped per model in `enable_for()`,
   never at package import.
3. **librosa keyword removed.** audio-separator calls
   `librosa.get_duration(filename=...)`, deleted in librosa 1.0, from
   `write_audio` — so separation completes on the GPU and throws on the last
   step, writing nothing. A minute of GPU work lost to a rename. Downgrading
   librosa was not an option (every analyser here is on the 1.0 API), so it is
   shimmed in `_patch_librosa()`.
4. **Stem matched by substring.** `load_cached` did `if name in filename`, so
   `other` matched "Whigfield - An**other** Day_(Vocals)" and overwrote the
   real *other* stem. Four correct files on disk, two names resolving to one of
   them — every mashup would have quietly used the vocal twice, and nothing
   would have errored. Fixed: parse the `(Tag)` only.

**The lesson under all four:** every failure mode in this phase was silent.
Wrong device, wrong DLL, wrong stem — none of them raised. That is why
`device_status()`, `residual()` and the band table exist.

**Demos:** `stem_acapella`, `stem_instrumental`,
`mashup_whigfield_over_alban`, `swap_stem`, `swap_frequency`.

---

## Cut style — hard cuts on the 1

Prompted by a reference he chose: *"Megamix EuroDance 90's Vol 1 (Mr Vain,
Tonight Is The Night, Run To Me, Be My Lover) — DJ Mario Andretti"*. All four
tracks exist in his library, so `reference/` renders the identical source
material and the comparison is direct. (I cannot hear the video — the format
reasoning below comes from the title and from what the analysis says about
the set, not from listening.)

**The set breaks our rules.** Keys 8A, 3A, 3B, 10A: only 3A↔3B is legal by
`LEGAL_KEY_DISTANCE`. 8A→10A is 2 steps, 8A→3A and 10A→3A are 5. The blend
planner flagged 2 violations and was forced into a 4.75 BPM jump.

**Why a professional gets away with it.** In a megamix nothing overlaps. A hard
cut has both tracks audible for ~4 ms, so no interval ever forms and there is
nothing for Camelot distance to measure. The harmonic constraint is not a
universal law, it is a consequence of *long blends*.

So `style` selects a vocabulary, not a standard:

| | blend (default) | cut |
|---|---|---|
| joins | 16-bar overlapping EQ curves | hard cut on the 1 |
| both audible | ~16 bars | ~4 ms |
| key weight | full | ×0.15, key violations dropped |
| transitions | bass_swap, eq_blend, filter_sweep, echo_out, … | hard_cut, cut_with_echo, vocal_slam_drop |

**Result on the reference set** — freeing the key constraint let the planner
solve the tempo problem instead:

| | order | violations | plan cost | worst BPM jump |
|---|---|---|---|---|
| blend, 16 bars | Run To Me → Be My Lover → Mr. Vain → Tonight | 2 | 114.41 | **+4.75** |
| cut, 8 bars | Run To Me → Mr. Vain → Be My Lover → Tonight | **0** | **7.17** | **+2.84** |

**The de-clicker.** A sample-accurate jump between two non-zero samples is a
step discontinuity, and a step is broadband click. `hard_cut` crossfades
equal-power over 4 ms — long enough to kill the click, far too short to hear
as a fade. First attempt ran the two ramps back to back (A down, then B up)
and measured **summed power 0.0** at the join: an 8 ms hole. Overlapping them
holds `out^2 + in^2 == 1.0000` right through. Verified flat at 4 and 8 bars.

A hard cut is only safe because the beatgrid is right. Land it a few ms off a
downbeat and it reads as a dropout, not an edit.

**Files:** `output/reference_ourway.mp3` (blend) vs `output/reference_cut.mp3`
(cut) — same four tracks, same lengths, only the joins differ.

---

## Sound quality rebuild

He said it did not sound good. Measured against the source tracks themselves,
he was right, and it was not subtle.

| | LUFS | crest | notes |
|---|---|---|---|
| La Bouche (source master) | −10.59 | 10.63 dB | the yardstick |
| Culture Beat (source) | −12.68 | 11.61 dB | |
| Le Click (source) | −14.02 | 12.24 dB | |
| **our mix, before** | **−17.24** | 16.05 dB | 6.6 dB below its own source material |
| **our mix, after** | **−11.78** | 9.23 dB | |

**Why it was stuck.** `master()` never called `compress()` at all — the chain
was gain → limiter, with `MAX_PEAK_BEFORE_LIMIT = 1.6` capping it at ~4 dB of
limiting. With a 16 dB crest the target was unreachable by construction. The
earlier note "a loudness target you cannot reach makes you quieter" was a true
observation about *that chain*, and the wrong lesson: the fault was the missing
stages, not the target.

New chain, each stage making the next one's job smaller: multiband compress →
glue compress → gain → soft clip → limiter. Limiter gain reduction dropped from
0.54 to 0.03 — it now does touch-up instead of rescue.

**Tuned by sweep, not by taste.** Aggressive and gentle multiband settings
reached the *same* final loudness (−11.62 LUFS either way), because the soft
clipper and the ceiling set the end point. The aggressive version just did
10.2 dB of low-band gain reduction to get there instead of 6.4. Took the gentler
road to the identical number.

**Two file-format bugs, both silent:**

1. **MP3s were 156 kbps.** `audio.save` called `fh.set_compression_level()`,
   which does not exist on `SoundFile` (it is `_set_compression_level`), inside
   a bare `try/except: pass` — so `--quality` never did anything, ever. And the
   scale is **inverted**: measured, 0.0 → 154 kbps, 0.7 → **56 kbps**, 1.0 →
   raises. The silent exception was accidentally protecting the output. Now
   encodes through LAME (ffmpeg, already a stems dependency) at **320 kbps**.
2. **WAV masters were PCM_16.** They are re-encoded to MP3 afterwards, so
   quantising to 16 bits first adds a noise floor the encoder spends bits on.
   Now PCM_24.

**Gain staging.** Per-track `match_loudness` used the *master* target, so with
the new −11/4.0 settings tracks would have hit the bus up to 12 dB over full
scale — and the multiband thresholds are absolute dBFS. Split into `BUS_LUFS`
(−16, tracks agree with each other) and `TARGET_LUFS` (−11, the push, once, at
the end).

## Matching instruments, not just tempo

**BPM was already right, and now it is proven.** One constant-tempo grid fits
the entire 16-minute mix at 133.2390 BPM — error 0.0000 against what the
renderer used — and onset energy lands 1.6–2.8× stronger on-beat than off-beat
in every 90-second window, across all three joins. Nothing drifts.

**What was missing was *what is playing*.** `analysis/instruments.py` measures
per-bar lead-voice activity, from stems when cached (exact) or a mid-band
proxy otherwise. `collision()` uses the **product** of the two sides, so one
track singing over the other's groove scores zero — that is a good transition,
not a problem. Only both-at-once costs.

**Result: the entries were already optimal.** At every join, bar 0 beat the
alternatives — join 3 by 22× (0.0222 vs 0.4869 at bar 32). Mr. Vain and Tonight
Is The Night have vocal activity of exactly 0.00 through their first 14 bars.
Entering at a track's intro *is* entering where there is no vocal. The feature
is a safety net that confirmed the design rather than repairing it.

**The exception it exposed:** La Bouche's "Be My Lover" is vocal-forward from
bar 2 (activity 1.00 at bars 2–7), so join 1 collides at 0.0397 and *every*
alternative is worse. No entry point fixes that one — it needs the outgoing
vocal ducked or removed with stems during the overlap. Not built yet.

**A third silent bug, found on the way.** `Separator.output_dir` cannot be
reassigned between calls: the loaded model keeps the directory it was built
with. Separating several tracks in one process wrote track 2's stems into
track 1's cache folder — eight files in one directory, `load_cached` matching
the wrong `(Tag)`s, track 2 appearing to produce nothing while track 1 silently
began returning someone else's audio. Now every separation stages to a scratch
directory and moves its output into place. Verified: 4 files each, correct
track, no bleed.

## Vocal ducking with stems

The one case entry-shifting could not fix: La Bouche's "Be My Lover" sings from
bar 2, so *every* candidate entry collided. Fixed by subtracting a scaled vocal
stem from the outgoing track across the overlap:
`a - (1-g)*vocals` leaves the voice at gain `g` and everything else untouched,
because Demucs stems sum back to the source.

`g` is driven bar-by-bar by the *incoming* track's vocal activity, so it is not
a fade: where B is instrumental, A keeps its vocal at full level. Measured
across the three joins:

| join | incoming | max duck | mean duck |
|---|---|---|---|
| 1 | Be My Lover (sings from bar 2) | **−16.48 dB** | −4.06 dB |
| 2 | Mr. Vain (instrumental intro) | −5.83 dB | −0.43 dB |
| 3 | Tonight Is The Night (instrumental) | −8.37 dB | −0.35 dB |

**Why not just EQ.** For the same job, measured on real audio:

| method | vocal removed | instrumental damage |
|---|---|---|
| stem duck (depth 0.85) | **−16.48 dB** | **0.00 dB** |
| EQ cut 6 dB @ 300–3500 Hz | −4.96 dB | −0.65 dB |
| EQ cut 16 dB | −9.93 dB | −0.92 dB |
| EQ cut 24 dB | −11.12 dB | −0.97 dB |

EQ cannot reach it at any depth — the voice has energy outside the band, so
cutting harder stops helping the vocal and only costs more instrumental. The
stem path damages the instrumental by exactly zero, by construction.

`prepare()` and `prepare_vocal()` share `_retime_slice()` deliberately: the two
signals must go through an identical transform or the subtraction stops
cancelling and starts phasing.

## Swift transitions are now the default

He was explicit: transitions should be swift, not the outgoing track sliding
down in volume while the next one arrives. Measured around the first join, that
complaint is exactly right:

| | dip through the join |
|---|---|
| 16-bar blend | **−18.6 dB** |
| 4-bar cut | −10.1 dB (the incoming track's own dynamics, not a fade) |

A 16-bar transition at 133 BPM is **28.8 seconds**. That is not a transition,
it is a slow dissolve. Defaults changed to `--style cut --bars 4` (7.2 s
window, instantaneous handover at its midpoint). `--style blend --bars 16`
still gives the old behaviour.

---

## Measuring the reference mix

He downloaded the DJ Mario Andretti megamix, so it could be analysed instead of
guessed at. **Three things I had asserted turned out to be wrong.**

| | I claimed | Measured |
|---|---|---|
| tempo | sets ramp upward; our flat tempo was gap #1 | **136.00 BPM, dead flat** across all 16 windows (135.85–136.00) |
| format | megamix = short hooks, ~60–90 s per track | **12.74 min / 4 tracks ≈ 3.2 min each** — same order as ours |
| level | his joins must avoid the dips ours had | **his mix dips below median MORE than ours** (5.1% of seconds vs our 2.6%) |

His actual master: **−12.35 LUFS, crest 13.78 dB, peak 1.339** (clipped above
full scale), balance 57.5 / 31.2 / 11.3.

Ours had been pushed to −11.78 LUFS at 9.23 dB crest — *louder than the
reference and considerably more squashed*. Retuned to **−12.3 LUFS** with
gentler multiband, and added `tilt_to()`, a two-shelf correction toward his
spectral balance: our high band was 8.6% against his 11.3%, which is
measurably duller. Now 11.5%.

The lesson is the same one as the key-detection "bug" and the CPU separation:
**measure the thing before optimising against a story about it.** My top
recommendation — tempo ramping — was aimed at a behaviour the reference does
not have.

One honest limit: a frame-wise classifier to recover his exact track
boundaries fragmented into 8–47 s runs and is not trustworthy. All four tracks
are 90s eurodance with near-identical instrumentation and chord movement, so
chroma/MFCC cannot separate them confidently. No conclusions drawn from it.

## Cut alignment bug

He said "the swift cut still needs to match bpm". The tempo was already exact —
the *phrase* was not. `hard_cut` fires at the midpoint of its region, but
`segment_plan` still placed the incoming track as if it were fading in across
the whole region, so every cut landed **1–2 bars before the incoming track's
first full bar** — mid-phrase. Perfect beatmatching, wrong entry.

Fixed with `transitions.entry_lead(name, bars)`: a blend enters `bars` early, a
cut only `bars * CUT_AT`. Transition types are now chosen *before* the segment
plan, since the type determines the placement. Verified **+0.0 bars off** on
every join.

## Phase 6 — the GUI

`autodj/gui/`, PyQt6, dark-mode-first neo-minimalist per his brief.

- `theme.py` — every colour, space and type step as a token; the whole
  stylesheet is generated from them. Weights capped at 600 because Inter ships
  here as Regular/Medium/SemiBold only and 700 would be synthetic bold.
  `qfont()` applies an explicit fallback chain: Qt does *not* read a CSS font
  stack when a QFont is built in code.
- `widgets.py` — Card, StatTile, Sparkline, BarChart, Skeleton, EmptyState,
  Toast, Badge. Charts hand-painted with QPainter (QtCharts is not installed,
  and painting them ourselves keeps them inside the design system).
- `waveform.py` — peak envelope, beatgrid with beat/bar/phrase hierarchy,
  structure bands, cue markers, scrub. Reduced to one min/max per screen
  column so drawing is linear in widget width, not track length.
- `workers.py` — QThread jobs for analysis, render and separation. **Workers
  never touch widgets**; they emit data and the GUI thread draws it.
- `library_view.py` — QAbstractTableModel + search proxy, sorting on the raw
  value rather than the formatted string.
- `main_window.py` — collapsible sidebar, status bar, responsive bento grid
  that reflows to one column below 1180 px.

`python main.py --gui`

## Still NOT built

Of the four gaps, only tempo ramping is partly written (`tempo_ramp()` and a
sample-based `_place()`), and it is **not wired into `render()`**. Given the
reference runs dead flat, it is also no longer the obvious first choice.
Loop roll, double drop and risers are untouched.

---

## Sync, master deck, and the cross-dissolve

He asked for transitions that are "buttery smooth", then clarified: *"not like
a cut, like when you're in Adobe Premiere Pro"* -- a cross-dissolve. And,
separately, for a CDJ-style **Sync**: a master deck the others lock onto.

**`dissolve` / `smooth_swap`** (new default style `smooth`). Equal-power cross
of mids and highs so summed power is flat, with the low band still *handed
over* rather than crossed -- the one place a dissolve must not dissolve. No EQ
carving, unlike `bass_swap`: it should sound like one track becoming another,
not two tracks being processed.

**Master deck.** `mix_tempo` took the median, so *every* track was stretched,
including one already at the target. `master_deck()` picks the track nearest
the median and runs the set at **its** tempo, so that deck is never
time-stretched at all. Max stretch fell 2.483% -> 2.186%.

**The measurement that mattered.** Phase alignment measured separately either
side of each join:

| | worst | mean |
|---|---|---|
| median tempo, pad/truncate | 24.09 ms | 20.74 ms |
| master deck + exact-length resample | 9.53 ms | 6.69 ms |
| + working sync | 8.53 ms | 6.86 ms |
| **measurement noise floor** | **6.52 ms** | **3.73 ms** |

The noise floor is the control: the same metric measured across a point
*inside* a single track, where the true mismatch is zero by construction. So
the residual real misalignment is ~3 ms, about 0.7% of a beat.

`_retime_slice` had been cutting a 5%-padded window and trimming, which meant
the phase vocoder decided the segment's true length and the trim decided where
the beats landed. It now cuts exactly the bars wanted and resamples to the
exact target length -- a sub-cent pitch change for sample-exact phrasing.

### Two mistakes worth recording

1. **The first Sync was inert and I reported it as working.** `beat_phase_error`
   measures phase relative to the segment start, and the onset envelope carries
   a systematic offset from STFT framing -- every segment reads about +200 ms
   whether aligned or not. The code compared that against zero, concluded every
   track was 200 ms out, and hit its own `max_ms=90` sanity limit, so it
   returned the audio untouched. The improvement I attributed to Sync had
   actually come from the master deck and the resample fix.
   Fixed: phase is now corrected **relative to the master deck's** phase, so a
   shared measurement bias cancels -- which is what a Sync button does anyway.
2. **A single shift was the wrong shape.** Vocoder drift varies along a track
   (measured spread within one track: 48 ms), so `sync_to` corrects with an
   interpolated curve, median-filtered so one bad window cannot bend it.

Honest note: with the corrected Sync the numbers barely moved (6.69 -> 6.86 ms
mean), because master deck + exact resample had already taken the error down to
near the noise floor. Sync earns its place on material with a wider tempo
spread, not on this four-track set.

---

## Sections, not whole tracks (`--sections`)

His note: *"when you mix the songs it doesn't mean you have to play the whole
song -- you can mix and match by the beat, or however world-class DJs do it."*

`autodj/arrange.py` builds the set out of **blocks** instead of records. Each
track is cut into phrase-aligned candidate blocks (32 bars by default, offered
every 8), each scored on what is *inside* it -- mean energy, overlap with a
labelled `drop` or `chorus`, penalised for straying into the intro or outro. A
weak track with one great drop contributes that drop; a strong track's filler
does not get in on reputation.

Blocks are then chosen greedily along the energy arc. The output has the same
shape as a segment plan, so transitions, sync, vocal ducking and mastering all
work unchanged -- the renderer never learns whether it is playing records or
sections.

    python main.py --input reference --sections --minutes 8 --block-bars 32

**The degenerate optimum, again.** First run alternated Run To Me / Be My Lover
eight times and never played the other two records at all. They are 3A and 3B
-- relative major/minor, Camelot distance 1.0 -- so every other choice cost +6
for a key jump. Every individual step was optimal; the result was absurd.

A reuse *cost* was not enough: at 3.2 per prior use it took four repeats before
a fresh record could compete with +6 of harmonic penalty. Coverage had to be a
**constraint** -- until every record has played once, only unplayed records are
eligible. This is the third time this project has produced a degenerate
harmonic optimum (ten tracks in 8A; the same-key selector chain), and the
lesson repeats: *a cost function that is right about one dimension will
happily destroy every other one.*

Result on the four-track set -- all four records, each twice, energy following
the arc, both labelled drops landing at the peak:

| # | at | bars | key | energy | track |
|---|---|---|---|---|---|
| 1 | 0:00 | 76–108 | 3A | 0.00 | Run To Me |
| 2 | 0:57 | 9–41 | 3B | 0.40 | Be My Lover |
| 3 | 1:55 | 20–52 | 3A | 0.53 | Run To Me |
| 4 | 2:52 | 25–57 | 3B | 0.68 | Be My Lover |
| 5 | 3:50 | 51–83 | 10A | **0.95** | Tonight Is The Night **[drop]** |
| 6 | 4:48 | 148–180 | 8A | 0.85 | Mr. Vain |
| 7 | 5:45 | 91–123 | 10A | **0.89** | Tonight Is The Night **[drop]** |
| 8 | 6:43 | 124–156 | 8A | 0.27 | Mr. Vain |

`output/megamix.mp3`, 7.9 min.

---

## The five upgrades (2026-08-23)

Rated the project — idea 8, features 9, **user experience 5** — and the gap was
the whole story: an excellent engine you could not hear, could not edit, could
not correct, and could not interrogate. All five recommendations built, plus
djay's Automix vocabulary, which he supplied mid-session as a reference.

### 1. Playback — `gui/player.py`

There was no audio output anywhere in the codebase. No `QMediaPlayer`, no
`sounddevice`. You rendered a mix inside the app and left the app to hear it.

`QAudioSink` over a `QIODevice` that reads from the `(2, N)` float array we
already hold. No temp file, no re-decode, and seeking is assigning an integer —
which is exactly what "audition this transition" needs.

Two details that took thought:

- **Position comes from the device's read cursor, not a wall clock.** A clock
  drifts against the sink's buffer and the playhead slides away from the audio
  over a fifteen-minute mix. The read cursor is where the sink actually got to,
  minus one buffer of latency, so it stays locked to what you hear.
- **A fresh sink per start.** Reusing one across a seek drains a buffer still
  holding the *old* position's audio — you hear a burst of where you were
  before arriving where you asked for.
- `readData` past the end returns silence, not `b""`. An empty read makes a
  running sink underrun and stop, and a sink that stopped by itself cannot be
  resumed — it has to be rebuilt.

### 2. The GUI is now a superset of the CLI, in djay's layout

The old window could not reach `smooth` — the CLI default, and the style Sync
was written for. It offered `cut` and `blend` only, and knew nothing about
`--sections`.

Rebuilt on djay's Automix arrangement: decks left, a fixed Automix column right.
An equal-weight bento grid cannot express that a DJ app has two things on screen
— what is playing and what is coming — because it treats every panel as equally
important. Sidebar navigation now actually switches views; it used to call
`setFocus()` on a card in a grid where everything was already visible.

### 3. Editable timeline + join previews

Drag blocks to reorder, change any join's transition and length from its queue
row.

Then a measurement that redirected the design. Caching prepared segments took a
re-render from **54.4 s to 34.2 s** — and **29.1 s of the remaining 34 was the
master chain over the whole 15.6-minute mix**. No amount of caching upstream
touches that. So the edit loop does not render the mix; `render.preview_join`
renders the *join* — two slices, one transition, a master pass over about half a
minute. Change a transition, hear it in about two seconds.

**Lesson: caching the expensive part is not the same as making it fast.** Find
out where the time actually is before optimising the part you assumed.

### 4. Correction UI — `corrections.py`

Beat detection is the one failure that ruins a whole mix, and the only recourse
was `--force`, which re-runs the same algorithm and gets the same answer.

The design decision that matters: **corrections are inputs to analysis, not
patches on top of it.** Overriding a BPM on a finished analysis leaves the
energy curve, the segments and the cue points all still indexed against the grid
you just rejected — so they agree with each other and disagree with the music.
`analyse_file` applies grid overrides immediately after beat tracking and cue
overrides after structure, so everything downstream is recomputed.

The cache follows for free: the fingerprint includes a digest of a file's
corrections, so saving one invalidates exactly that track and reverting one
restores its old entry. Verified — halving a BPM took `n_bars` 187 → 93 and
re-indexed the energy curve 186 → 92 entries; clearing restored both the
analysis and the original fingerprint.

### 5. The move set finished, and the reasoning surfaced

- **Double drop** built. Its timing is dictated by the music, not the region:
  `segment_plan` sets A's exit and B's entry so the two drop bars land on the
  same mix bar. If that would truncate A too far the alignment is abandoned
  rather than forced — a double drop with the drops in different bars is worse
  than an ordinary overlap.
- **Tempo modes wired**: off / sync / blend / auto.
- **`explain.py`** turns every choice into a sentence: which rule fired, how
  long the join is and why, the harmonic relationship, the tempo movement in
  cents, where each track enters and leaves, and — with `--audition` — what each
  candidate measured.

### djay's vocabulary, and going past it

Added the three missing transitions: **Fade** (plain crossfader), **Tremolo**
(accelerating gate, raised-cosine so it swells rather than clicks, rate swept by
accumulating instantaneous frequency so it never restarts mid-cycle), and
**Neural Mix** as `stem_blend` — mixing by instrument instead of by frequency.
Drums hand over first, then harmony, then vocals, and A's vocal is *removed*
rather than ducked. No EQ setting can do that.

Then the three settings were taken past matching djay, using the one advantage
an offline renderer has: it can measure instead of guess.

- **Duration `auto`** is a fit, not a lookup. Every legal length is scored on
  where the region lands against both tracks' section boundaries, the measured
  vocal overlap, the energy difference across it, and the type's own character.
- **Tempo blend** is eased with smoothstep. A linear glide changes tempo at a
  constant rate, so the *rate* jumps from zero to full at each end — and the ear
  tracks tempo change, not tempo, so those two corners are the audible part.
  Above 30 cents the glide switches from resampling to a pitch-preserving
  block-wise stretch: 5% is 84 cents, most of a semitone, and no Camelot
  planning survives that.
- **`--audition`** renders each join's candidates and keeps whichever measures
  best on five faults: a hole in the middle, two basslines at once, low-mid mud,
  chroma-level harmonic clash (which a key label cannot see), and vocal overlap.

### One arithmetic trap

`_cumulative_seconds` integrates `4*60/T`, not `T`. A bar at tempo T lasts the
*reciprocal* of the tempo; integrating tempo directly makes an accelerating
passage get longer.

### Housekeeping

`reference/` was found empty mid-session — the four test tracks gone, `musica/`
gone entirely. No command run this session deletes audio, and the cause was not
identified. The originals were safe in
`Downloads/Disco.80s.90s.The.Greatest.Hits/` and the four were restored (and
verified in a separate call, per the sandbox-rollback rule).

### Measured, at the end

| Thing | Result |
|---|---|
| Join preview (the edit loop) | **1.15 s**, vs 34 s for a cached full re-render |
| Full render, 4 tracks / 15.6 min | 47–55 s depending on mode |
| Deck-to-deck lock, `sync` | **1.9 ms** mean across joins |
| Deck-to-deck lock, `blend` | **13.5 ms** mean, worst 17.4 ms |
| Body stretch, `sync` | 2.186% |
| Body stretch, `blend` | **0.0%** — only the joins are touched |
| Double drop alignment | **+0 bars** on both decks |
| Corrections round-trip | BPM ÷2 took n_bars 187 → 93, energy curve 186 → 92 |
| `--audition`, 3 joins | 25.8 s; ranked `filter_sweep` clearly first on both key-clash joins |

Two measurement lessons from that table.

**Measuring an accelerating passage against a fixed grid is meaningless.** The
first attempt at verifying tempo blend compared each side of a join against a
constant tempo and reported 141 ms of error. That number was an artefact of the
method: in blend mode both decks are gliding, so neither is at a constant tempo,
and the grid being measured against does not exist. The right question is
whether the two decks are locked to *each other*, measured by cross-correlating
their onset envelopes through the overlap.

**Then it found a real bug anyway.** Done properly, blend measured 65.8 ms mean
— driven entirely by one join reading -180 ms, which happened to be the only
4-bar region. A glide is a *rate*: moving 4.7 BPM across 4 bars is the same
change happening four times faster than across 16, and both decks have to track
it exactly. `glide_lengths` now gives any join that actually changes tempo at
least 8 bars. That took blend to 13.5 ms mean and the -180 ms case vanished,
which is what confirms it was the abruptness rather than the measurement.

`sync` is still tighter (1.9 vs 13.5 ms) and remains the default. `blend` buys
0% body stretch — no phase vocoder on any track's body at all — for about 12 ms
of alignment. That is a real trade, not a free upgrade, and the numbers are here
so it can be made deliberately.

### Still open

- `--audition` has not yet been shown to *change* a choice. It ranks sensibly
  (filter_sweep wins both key-clash joins by a clear margin) but on four
  well-behaved eurodance records with correct bass handovers there is nothing
  for it to catch. It needs a messier library to prove its worth.
- The stem cache was keyed on path+size+mtime, so restoring four identical
  files orphaned six separated tracks that were still sitting on disk. Now
  keyed on content (size + 1 MB from each end), and the six folders were
  relinked rather than re-separated.

### GUI: the black bars, and the text (2026-08-23)

Reported as "long black bars" and bad text. Both had single root causes.

**The bars were one stylesheet rule.** `QWidget { background: BG; }` is what
gives the app its ground colour, and Qt cascades it to *every* widget --
including every QLabel, which then fills its whole allocated width with the app
ground. Cards are `SURFACE`, which is lighter than `BG`, so every label sitting
on a card painted a dark rectangle behind its text: one under every field
caption, every metric row, every track title. `QLabel, QCheckBox { background:
transparent; }` removed all of them at once.

The same fault repeats at each level of the widget tree that a rule does not
reach: the reasoning panel's QScrollArea, its viewport, the body widget inside
it, and then the per-row containers inside that. Fixed with a descendant
selector plus one explicit call on the row containers.

**Lesson:** a global background rule on `QWidget` is not a background, it is a
default that every descendant inherits. Set the ground on the actual ground
widgets and let everything else be transparent.

**The text was three separate bugs, all silent:**

- `_select_join` sliced the reason to `[:96]` before handing it to a label.
  That chopped mid-word with no ellipsis, so it *looked* like the card was
  clipping text that had in fact been cut before it ever arrived. Removed --
  the label is the only place that decision can be made correctly.
- Eliding from a resize handler is a timing bug. The width you measure against
  is whatever the label had before the layout settled, and when it is wrong the
  failure is invisible: the text just runs off the edge. `ElidedLabel` now
  elides in `paintEvent`, where the width is by definition the real one.
- Compact controls in the queue rows had a fixed 26px height fighting the
  stylesheet's 10px input padding. That does not shrink a control, it clips the
  text inside it -- which read as a missing font rather than a layout error.

Also fixed while there: BPM shown as `13362` (an int spin box scaled by 100 --
obvious to whoever wrote it and to nobody else), the tempo chart drawn from a
zero baseline so 130 and 135 BPM were four identical bars, chart labels
truncated to six characters regardless of available width, cell borders drawn
through the selected table row, and cue tags sliced off the card edge -- which
is precisely where an OUT marker lives.

The GUI also no longer opens onto an error. A missing or empty input folder is
the state the app starts in, not a failure; it now lands on the Library empty
state and its "Choose folder" button instead of throwing a red toast over them.

### Icons, and three missing transitions (2026-08-23)

Asked for icons, and told "there's some transitions missing". Both were true.

**The missing transitions were literal.** `fade`, `tremolo` and `stem_blend`
were built, registered in the factory, listed in the GUI dropdown and fully
renderable -- but never added to `transitions.NAMES`. Anything iterating the
registry silently saw 13 of 16. Nothing failed; the three simply did not exist
as far as the rest of the program was concerned.

Auditing that also caught `NON_OVERLAPPING`, a hand-maintained tuple naming the
transitions across which a key clash cannot be heard. It listed `dissolve` and
`smooth_swap`, which measure **0.37** overlap -- both tracks plainly sounding
together for a third of the region. It was never read by anything, so the error
had been invisible. It is now computed from the curves:

    NON_OVERLAPPING = tuple(n for n in NAMES if overlap_of(n) < 0.05)

which yields the four real cuts and cannot go stale when a transition is added
or reshaped.

**Icons are generated from the DSP.** `gui/icons.py` draws each transition's
mark by building the real `Transition` and plotting its actual `out_mid` and
`in_mid` gain curves. A hard cut's icon is a step because the curve is a step.
A tremolo's shows the gate oscillating and accelerating because that is
literally the automation. Neural Mix shows four stem lanes entering at their
own times, read from `tr.stems`. Amber is the outgoing deck, cyan the incoming
one -- the same encoding as the deck waveforms.

The point is not that it is cheaper than drawing them. It is that a hand-drawn
icon is a *claim* about what a transition does, and claims drift -- this file
was written immediately after finding a drifted claim in the same module. A
picture computed from the curve cannot: change the DSP and the icon changes
with it, and if the icon looks wrong, the transition is wrong.

Transitions with identical automation get one distinguishing accent from the
thing the renderer adds rather than the curve: the echo's decaying tail, the
roll's accelerating ticks, the riser's sweep, the double drop's coincident bar.
Without it `hard_cut` and `cut_with_echo` would draw the same picture.

UI marks are drawn paths too, for duller reasons: no licence, no second palette
to keep in sync with the tokens, and no chance of a tofu box. The Unicode set
they replaced all existed in the font and all rendered, at body size, as
near-identical small grey squares -- a worse failure than a missing glyph
because it looks deliberate.

**Two bugs the contact sheet exposed immediately**, which is the argument for
rendering one: the back and forward arrows pointed the wrong way (a mirror-image
pair, both wrong, consistent enough to survive a glance), and `half`/`double`
were indistinguishable marks sitting beside buttons that already read "÷2" and
"×2" -- deleted, because a redundant ambiguous icon is worse than none.

**And one process lesson.** The deck header's icon never appeared: the patch
that added `setPixmap` was applied with a string replacement against text an
earlier edit had already changed, so it matched nothing and did nothing,
silently. Chained blind replacements have no failure mode. Use an edit that
errors on a miss.

## Design system pass, and a crash

### The crash: clicking the mix waveform

Reported traceback: `_bar()` → `self.meta["bpm"]` → `TypeError: 'NoneType' object
is not subscriptable`.

Every *paint* path in `waveform.py` guarded `self.meta`. `mousePressEvent` did
not. The mix view is the one waveform that legitimately has audio and no track
meta — it is the whole rendered set, not one track — so it drew perfectly and
died on the first click. Clicking the biggest thing on the Mix page to seek was
a hard crash, and it survived every screenshot because rendering a frame never
clicks it.

Fixed at the root rather than the call site: `_bar()` returns `None` when there
is no grid, `cue_marks()` returns `[]`, and both paint helpers bail. The same
guard covers a BPM of zero, which the Correct page produces the moment the
field is cleared mid-typing.

The regression test was checked against the *old* code first — a test that
passes on the broken version tests nothing. It also had a bug of its own worth
recording: the audio array was built `(samples, channels)`, and `set_track`
does `np.atleast_2d(audio).mean(0)`, so the transposed shape silently became a
2-sample file instead of raising. "Seek works" was measuring nothing until the
shape was fixed.

### Tokens split by role

There is no longer a single `BORDER`. `HAIRLINE` is decorative and has no
contrast obligation; `CONTROL_EDGE` is what makes a control identifiable and
needs 3:1; `FOCUS` is the keyboard ring and needs 3:1 against both the surface
and the fill it is drawn on. Collapsing those into one token is what forces the
usual bad trade — either an inaccessible focus ring or heavy boxes everywhere.

On a dark card the edge has to carry it: the card is dark enough that even pure
black against it measures 1.21:1, so no fill can identify a control on its own.

### The audit found my own bugs

`python -m autodj.gui.theme` measures every pair the app can produce, in both
modes. It failed 31 pairs on the first palette I wrote. Three were real defects
rather than tuning:

- The light-mode focus ring was the same indigo as the button fill it outlined
  — **1.00:1**, invisible to exactly the keyboard users it exists for. A ring
  drawn *on* a fill cannot be the accent colour; on filled buttons it is now
  `ON_ACCENT`, 6.29 against the fill in both modes.
- Structure segments were a hex table inside `waveform.py`, outside the palette
  and therefore never audited. Three of the seven measured 1.71 or below. They
  are `theme.SEGMENTS` now.
- `Badge` filled with `rgba(255,255,255,0.045)` — a lighten-on-dark assumption
  that is white-on-white in light mode, so the pill lost its fill and kept its
  border.

### The four blues

`#0D47A1 · #2196F3 · #90CAF9 · #E3F2FD`. They divide by role almost exactly
once measured, with one inversion worth knowing: **#2196F3 cannot carry white
text** (3.12) but takes near-black at 6.72. So on dark the primary button is
bright blue with a dark label — which is also the brightest thing on screen,
correctly. The pressed state stops at Blue 600; going to the brand deep blue
looks right and measures 1.93 against that dark label.

The data ramp deliberately does *not* collapse into the blues. Six blues would
encode six different things as six shades of one thing, and the decks are drawn
on top of each other at every join.

### Colour against data, not against tokens

The timeline paints one block per track from the data ramp, so no fixed token
is right for the label on top of it. `theme.on_colour()` picks the better
foreground, and `theme.blend()` resolves the translucent fill first — the block
faded from alpha 210 to 95, so the sub-label was sitting on almost pure card
colour, and in light mode that meant white on near-white.

Making the block opaque where the title sits was necessary because at alpha 210
the mid-blue reached 4.33 against black and *less* against white: no foreground
existed that passed.

### Restyle protocol

Switching palette live needs three things, and missing any one leaves the
window half-converted: regenerate the stylesheet, redraw every icon, repaint
every custom widget. Icons bake their colour at draw time and a widget holds
the QIcon it was given, so clearing the cache is not enough. `icons.apply()`
records how each icon was requested — storing the *token name*, not the colour,
so it resolves against whichever palette is live at replay time — and
`icons.restyle()` replays them. It returns a count, because a silent zero is
what a half-converted window looks like from outside.

`icons.OUT_COLOUR = T.SERIES[3]` at module level had the same fault in a purer
form: bound once at import, frozen forever. Now a function call.

### The app icon

The equal-power crossfade: two curves, one falling and one rising, crossing at
centre. Not decoration — it is the single rule the whole program is built on,
and the same picture the Dissolve transition icon draws from its own
automation. Fixed in the brand blues rather than following the live palette; an
app icon sits in a taskbar where the theme is not this app's to decide.

Drawn at each size rather than downscaled, and below 20px it drops to the bare
crossing, because the curves turn to mush there. The first version had both
curves at the same lightness and read as a generic X — the whole idea is that
one deck is leaving while the other arrives, so the incoming one is now white
and reads first.

`assets/autodj.ico` is assembled from the PNG payloads directly; Qt writes only
one size per ICO.

### Two more crashes, and a top bar that never fitted

**`_drag_cue` — every click on a waveform.** Assigned in one branch of
`mousePressEvent` (the branch where the press lands on a draggable cue) and
read unconditionally in `mouseReleaseEvent`, so any ordinary click raised
`AttributeError` when the button came back up. Now initialised in `__init__`.

The click regression test had missed it because it only ever pressed. It does
whole press → move → release cycles now, and drives the four edit modes.

**The status text overlapping the progress bar.** Reported as text overlap
during a job; the cause was the whole row.

Measured, the top bar needed roughly 1200px of a 1080px row *at full width* —
search 300 + status + progress 170 + theme + render + GPU badge. So items
overlapped by construction: the progress bar sat across the theme and render
buttons, and at 1280 the search field ran 153px into the status text. On top of
that the status was a plain `QLabel` sized to its own text, and
`"Analysing 1/4 · <title>"` is a different width for every track — so the whole
row re-laid out on every progress tick, and the label needed up to 431px in a
slot giving it 257 and painted past its own edge.

Three separate fixes:

- The status region is now the one *expanding* element and everything else is
  fixed, so slack has somewhere to go and the right-hand controls never move.
- The status is an `ElidedLabel`, so an over-long message truncates with an
  ellipsis instead of overflowing.
- `TopBar.fit(width)` drops what the row cannot hold, in order of what is worth
  least mid-task: the GPU badge below 1500, a narrower search below 1400, a
  narrower progress bar below 1400. Choosing nothing to sacrifice is what
  produced the overlap.

One trap worth recording: `ElidedLabel` sets `QSizePolicy.Ignored` horizontally
so it can shrink below its text. Give it a `setFixedWidth` as well and a box
layout allocates it **zero** width when positioning the next item while still
honouring the fixed width when it sets the geometry — the two disagree, and the
progress bar landed 8px from the label's *left* edge, on top of the text. The
fix is an explicit policy, not a fixed width.

`scratchpad/topbar.py` measures every pair of controls in the row, in the busy
state, at eight window widths. The first overlap sweep found nothing because it
only ran with the bar idle at "Ready" — the one state the bug does not occur in.

### "Only three options in transition style"

Two different controls, and the label was mine to blame. The Automix panel's
selector is a *set-wide planner policy* — smooth / cut / blend, fed to
`planner.choose_transition` and `two_opt` to bias which moves get reached for
and how hard key compatibility is weighted. The sixteen transitions are in each
queue row, one per join.

It was labelled "Transition style", borrowing djay's name for its
transition-type menu, so three entries where sixteen were expected read as
fourteen missing ones. Renamed to "Mixing style", with a hint saying where the
per-join control actually is.

### QFont::setPointSize warning

`qfont()` uses `setPixelSize`, so `pointSize()` returns -1; Qt emits the warning
whenever an internal path round-trips the application font through
`setPointSize`. `theme.app_font()` sets family and hinting only and leaves the
point size alone. Sizes still come from the stylesheet, which sets `font-size`
in px on QWidget.

### Saving, a set-wide transition, Roboto — and two type mistakes

**The GUI could not save.** `RenderWorker` has always taken an `out_path` and
written WAV, tracklist, cue sheet and JSON, but the window only ever called it
with `out_path=None`. The only way to get a file out of this program was the
command line. `save_mix` (Ctrl+S, or the export mark in the top bar) writes the
audio already in memory rather than re-rendering — the mix is right there, and
re-rendering to save would be a minute of work to produce it again.

**A set-wide Transition control.** The 3-option planner policy stays, renamed
"Mixing style"; above it sits a Transition combo with the same sixteen moves
the rows offer. "Automatic" hands each join back to the planner, anything else
applies to every join, and a row can still override afterwards. When a
transition is pinned the Mixing style control is disabled — it biases *which*
move the planner reaches for, and there is nothing left for it to decide.

**Roboto**, checked before switching: it is installed here with Regular through
Black, including a real Bold, which the previous stack did not have (Inter
shipped only three weights). `python -m autodj.gui.theme` now prints what each
type step actually resolved to, so a machine without the family is told rather
than silently re-rendering in a fallback. Roboto Mono is absent here and the
audit says so.

**Two mistakes of the same kind, both mine.** `render()` returns `joins` as
built `Transition` objects, not names. I passed `self.join_names` into
`queue.set_plan` and `timeline.set_plan`, where `joins[i].name` raised — inside
a signal handler, which in PyQt takes the whole process down rather than
printing anything. The render simply stopped, silently. Then the same error in
`save_mix`, where the WAV wrote and the tracklist, cue and JSON did not, with
the failure swallowed by the error toast.

Both are fixed at the source: `_refresh_queue` rebuilds real `Transition`
objects from the names, and `set_plan` in the queue now accepts either — the
tolerance `TimelineView` already had.

### `pullFromQIODeviceToRingbuffer cannot read from QIODevice`

`_Stream.bytesAvailable()` and `_Stream.readData()` disagreed. Past the end of
the mix, `bytesAvailable` returned 0 while `readData` kept handing back silence
to stop the sink underrunning — so Qt read at a moment its own accounting said
was empty, and logged that. Worse, `set_frame` did not clamp, so seeking past
the end (which `set_audio` does whenever it keeps the position across a shorter
mix) left `_pos` beyond the buffer and `bytesAvailable` **negative**.

The tail of silence is now a declared quantity the device reports, and
`set_frame` clamps to the buffer. `scratchpad/player_test.py` drives the device
the way a sink does, past the end and across seeks, with a Qt message handler
installed to catch the warning rather than trusting that it is gone.

### The four libraries, checked rather than assumed

**librosa** — already the analysis backend, 1.0.0, nothing to do.

**demucs** — the standing rule "never install demucs, it downgrades torch" was
true of 4.0.x, which required `torchaudio`. **4.1.0 moved torchaudio into the
`train` extra**, so inference needs only `torch>=2.1`. Installed and verified:
torch still `2.13.0+cu126`, `cuda.is_available()` True, torchaudio still
absent, `htdemucs_ft` loads with its four sources. The primary stem path is
still `audio-separator` running the same model; demucs is now available
natively as well.

**pedalboard** — installed from a `cp313-win_amd64` wheel, no build, torch
untouched. `dsp/pedal.py` wraps it, off by default.

**essentia** — not viable here, and this is a property of the package rather
than of the machine: PyPI has only a 2021 `2.1b5` **source** tarball with no
Windows wheel for any Python version, so pip would attempt a C++ build needing
FFTW, TagLib and Eigen. Not attempted.

### The measurement said don't swap the master chain

`dsp/pedal.py` exists so the question could be settled with numbers instead of
with the observation that JUCE is very good. On 45 s of *Mr. Vain*, target
-12.3 LUFS, ceiling 0.97:

| chain | LUFS | error | peak | crest |
|---|---|---|---|---|
| built-in | -12.44 | **-0.14** | **0.97** | 10.86 |
| pedalboard | -13.04 | -0.74 | **1.00** | **11.84** |

The pedalboard chain misses the loudness target by more than five times as
much and **overshoots the ceiling to full scale** -- `Limiter.threshold_db` is
a threshold, not a brickwall, and it lets transients through. It does keep a
decibel more crest, which is a real advantage and the wrong trade when the
result is clipping.

So the master chain stays as it is. What pedalboard is kept for is the two
things we genuinely do not have: a JUCE reverb and a real resonant ladder
filter, both in `transition_fx`. Speed was not a reason either way -- its
limiter ran 13 ms against our 14 ms on 8 s of audio.

One trap on the way in: every stage in `master.py` returns
`(audio, something-for-the-report)`. Feeding a whole tuple to the next stage
does not raise anything readable -- numpy builds a ragged object array and
fails several frames later with "inhomogeneous shape".

And `transition_fx` clamps: reverb at amount 0.8 took a peak of 0.97 up to
**1.566**. On the render path the master chain would catch it, but a join
preview is never mastered, and a float sink handed 1.5 produces full-scale
noise rather than a loud reverb.

### Making the libraries earn their place

**Two transitions that use pedalboard**, so the dependency is called rather
than merely installed:

- **Reverb wash** — the outgoing track dissolves into its own reverb tail. The
  move an echo-out cannot make: an echo *repeats* what was just played, so the
  outgoing track stays recognisable and keeps competing; a reverb tail smears
  it into unpitched texture. The safest thing to do across a hard key clash.
- **Ladder filter** — `filter_sweep` with a resonant 24 dB/oct ladder instead
  of a Butterworth. The band curves are byte-identical to `filter_sweep` on
  purpose, so if it sounds better that is the filter and not a different fade.

`Transition` gained an `fx` field the renderer reads next to `echo`/`roll`/
`riser`. When pedalboard is absent the effect is skipped and the transition
still works from its curves — quieter than intended, never broken.

Verified as actually applied rather than assumed: against `filter_sweep` on the
same two tracks, `ladder_sweep` differs on **76.6%** of samples and
`reverb_wash` on **87.4%**, both with peak still at the 0.97 ceiling.

### A native Demucs backend, and why it needs its own process

`AUTODJ_STEM_BACKEND=demucs` (or `backend="demucs"`) runs Demucs directly on
torch instead of through audio-separator. Measured on *Be My Lover*:

| | audio-separator | demucs |
|---|---|---|
| seconds | 62.2 | **42.3** |
| stems sum back to source (r) | 0.9941 | **0.9966** |
| stems produced | 4 | 4 |

**32% faster and marginally cleaner**, so it is worth having — but the default
stays `audio-separator` until that holds over more than one track on one run.

The backend runs in a **subprocess**, and that is necessity rather than
caution. `audio_separator` prepends its own vendored copy of Demucs to
`sys.path[0]` when it loads a model:

    sys.path[0] = .../audio_separator/separator/architectures/../uvr_lib_v5

From that point `import demucs` resolves to the fork for the rest of the
process, and `get_model` goes looking for the fork's `remote/files.txt`. That
is exactly how the first attempt failed. No import ordering fixes it — the path
is rewritten at model-load time, after any import we control. Separation
already costs a minute and communicates through files, so a fresh interpreter
costs nothing measurable. The worker asserts it did not get the fork.

Both backends write the same `_(Tag)_` filenames into the same content-keyed
cache, so `load_cached` cannot tell which produced a folder.

### A measurement bug worth remembering

The first backend comparison reported a reconstruction correlation of
**0.0006** — stems that do not sum to their source at all. The stems were fine;
the benchmark loaded them at their native rate and the source at *its* native
rate, compared two differently-sampled arrays, and got noise. Loading both at a
matched rate gives 0.9941. A metric that disagrees with a previously measured
0.984 by three orders of magnitude is measuring the wrong thing, not
discovering a catastrophe.
