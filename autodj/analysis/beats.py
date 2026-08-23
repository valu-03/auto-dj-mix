"""Tempo and beatgrid.

The core idea: do NOT trust a list of detected beat times. Instead fit a
*constant-tempo grid* directly against the onset envelope, by asking
"which (bpm, phase) puts the most onset energy on grid lines, and the least
between them?"

Dance music from a drum machine really is constant-tempo, so a two-parameter
model beats any amount of per-beat tracking, and it extrapolates cleanly into
the silent intro and the fading outro where beat trackers fall apart.
"""

import numpy as np

from .. import spectral
from . import onsets

BPM_MIN = 70.0
BPM_MAX = 200.0
PRIOR_BPM = 128.0
PRIOR_WIDTH = 0.45      # in octaves; wide enough not to force the answer
PHASE_BINS = 128        # phase resolution: period/128 ~ 3.4 ms at 136 BPM


def _autocorr(env):
    """Normalised autocorrelation of the onset envelope."""
    x = env - env.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    return ac / (ac[0] + 1e-12)


def tempo_candidates(env, sample_rate, hop=spectral.HOP, top=6):
    """Plausible tempos, best first, from autocorrelation peaks."""
    frame_rate = sample_rate / hop
    ac = _autocorr(env)

    lags = np.arange(1, len(ac))
    bpms = 60.0 * frame_rate / lags
    keep = (bpms >= BPM_MIN) & (bpms <= BPM_MAX)
    bpms, vals = bpms[keep], ac[1:][keep]

    # A log-Gaussian prior around 128 BPM. It does not decide the tempo; it
    # only breaks ties, and `score_grid` below is what really rules on octaves.
    prior = np.exp(-0.5 * (np.log2(bpms / PRIOR_BPM) / PRIOR_WIDTH) ** 2)
    score = vals * prior

    out, seen = [], []
    for i in np.argsort(score)[::-1]:
        bpm = float(bpms[i])
        if any(abs(bpm - s) < 2.0 for s in seen):
            continue
        seen.append(bpm)
        out.append((bpm, float(score[i])))
        if len(out) >= top:
            break
    return out


def score_grid(env, period, phase_bins=PHASE_BINS, contrast=False):
    """For one beat period, score every phase.

    Default score is the mean envelope value ON the grid. Sampling by
    interpolation (rather than binning frames into phase buckets) is essential:
    bucketing concentrates energy whenever the period happens to be a whole
    number of frames, which biases any search towards those periods by ~16x.

    `contrast=True` subtracts the mean at the midpoints between grid lines. That
    is only meaningful as a *quality* measure, not as a search objective --
    eurodance basslines play offbeat eighths, so the midpoints carry real
    low-band energy and the contrast at the true tempo is not the maximum.
    """
    x = np.arange(len(env), dtype=np.float64)
    n = int((len(env) - 1 - period) / period)
    if n < 8:
        return None, None
    k = np.arange(n, dtype=np.float64)
    phases = np.linspace(0.0, period, phase_bins, endpoint=False)
    pos = phases[:, None] + k[None, :] * period
    on = np.interp(pos, x, env).mean(axis=1)
    if not contrast:
        return phases, on
    off = np.interp(pos + 0.5 * period, x, env).mean(axis=1)
    return phases, on - off


def _search(env, frame_rate, lo_bpm, hi_bpm, steps):
    """Best (bpm, phase_frames, score) over a tempo range."""
    best = None
    for bpm in np.linspace(lo_bpm, hi_bpm, steps):
        period = 60.0 / bpm * frame_rate
        if period < 4:
            continue
        phases, score = score_grid(env, period)
        if phases is None:
            continue
        j = int(np.argmax(score))
        if best is None or score[j] > best[2]:
            best = (float(bpm), float(phases[j]), float(score[j]))
    return best


def fit_grid(env, sample_rate, hop=spectral.HOP, bpm_hint=None,
             span=0.025, coarse=81, fine=81):
    """Fit (bpm, offset_seconds, score) to the onset envelope.

    Two stages. Coarse locates the tempo to ~0.15 BPM; fine refines to ~0.003
    BPM, which matters more than it looks: over a 4-minute track a 0.05 BPM
    error accumulates into ~0.1 s of drift by the outro, and the outro is
    exactly where we mix.
    """
    frame_rate = sample_rate / hop
    if bpm_hint is None:
        cands = tempo_candidates(env, sample_rate, hop)
        bpm_hint = cands[0][0] if cands else PRIOR_BPM

    rough = _search(env, frame_rate, bpm_hint * (1 - span), bpm_hint * (1 + span), coarse)
    if rough is None:
        return float(bpm_hint), 0.0, 0.0

    best = _search(env, frame_rate, rough[0] * 0.9985, rough[0] * 1.0015, fine)
    if best is None or best[2] < rough[2]:
        best = rough
    bpm, phase_frames, score = best
    return bpm, phase_frames * hop / sample_rate, score


def choose_tempo(env, sample_rate, hop=spectral.HOP):
    """Autocorrelation picks the tempo; grid fitting refines it.

    Splitting the job this way matters. Autocorrelation with an octave prior is
    reliable at choosing *which* tempo (it was right on all 10 test tracks by a
    wide margin), but its resolution is limited by integer lags -- every one of
    those tracks reported exactly 136.00 because they all landed on lag 19.
    Grid fitting then refines inside a narrow window, where there is no octave
    ambiguity left to get wrong.
    """
    cands = tempo_candidates(env, sample_rate, hop)
    hint = cands[0][0] if cands else PRIOR_BPM
    return fit_grid(env, sample_rate, hop, hint)


def grid_confidence(env, sample_rate, bpm, offset, hop=spectral.HOP):
    """How much more onset energy sits on the grid than between it."""
    frame_rate = sample_rate / hop
    period = 60.0 / bpm * frame_rate
    x = np.arange(len(env), dtype=np.float64)
    n = int((len(env) - 1 - period) / period)
    if n < 8:
        return 0.0
    pos = offset * frame_rate + np.arange(n) * period
    on = float(np.interp(pos, x, env).mean())
    off = float(np.interp(pos + 0.5 * period, x, env).mean())
    return on / (off + 1e-9)


def beat_times(bpm, offset, duration):
    """Every beat from the start of the track to the end."""
    period = 60.0 / bpm
    first = offset - period * np.floor(offset / period)   # wrap back towards 0
    n = int(np.floor((duration - first) / period)) + 1
    return first + np.arange(max(0, n)) * period


def find_downbeat(beats, mag, sample_rate, hop=spectral.HOP, beats_per_bar=4):
    """Which of the 4 beat positions is the '1'?

    The kick lands on the downbeat, so score each candidate phase by low-band
    energy on its beats and take the strongest.
    """
    low = spectral.band_energy(mag, sample_rate, spectral.LOW_BAND)
    frames = np.clip((beats * sample_rate / hop).astype(int), 0, len(low) - 1)
    if len(frames) < beats_per_bar * 2:
        return 0
    scores = [float(low[frames[p::beats_per_bar]].mean()) for p in range(beats_per_bar)]
    return int(np.argmax(scores))


def analyse(audio_array, sample_rate, mag=None):
    """Full beat analysis for one track."""
    if mag is None:
        mag = spectral.magnitude(audio_array)

    # Fit the grid on the LOW band: in dance music the kick defines the beat,
    # and ignoring hats and vocals removes most of what could mislead us.
    env = onsets.envelope(mag, sample_rate, band=spectral.LOW_BAND)
    if env.max() <= 0:
        env = onsets.envelope(mag, sample_rate)

    bpm, offset, score = choose_tempo(env, sample_rate)
    confidence = grid_confidence(env, sample_rate, bpm, offset)

    duration = spectral.as_mono_1d(audio_array).shape[0] / sample_rate
    beats = beat_times(bpm, offset, duration)
    phase = find_downbeat(beats, mag, sample_rate)
    downbeats = beats[phase::4] if len(beats) else beats

    return {
        "bpm": round(float(bpm), 4),
        "beat_offset": round(float(beats[0]) if len(beats) else 0.0, 6),
        "downbeat_phase": int(phase),
        "first_downbeat": round(float(downbeats[0]) if len(downbeats) else 0.0, 6),
        "beat_period": round(60.0 / float(bpm), 6),
        "grid_score": round(float(score), 6),
        "grid_confidence": round(float(confidence), 3),
        "n_beats": int(len(beats)),
        "n_bars": int(len(downbeats)),
    }
