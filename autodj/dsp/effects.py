"""Transition effects: echo tails and loop rolls."""

import numpy as np

from . import filters


def echo(audio, sample_rate, delay_s, feedback=0.55, mix=0.7, taps=10,
         damp_hz=6000.0):
    """Feedback delay, expanded into a finite sum of taps.

    A true feedback loop is recursive and has to be run sample by sample. But
    with feedback < 1 each repeat is quieter than the last, so after ~10 taps
    the rest is inaudible -- and a finite sum of delayed copies is one
    vectorised operation instead of millions of Python iterations.
    """
    a = np.atleast_2d(audio).astype(np.float64)
    d = int(round(delay_s * sample_rate))
    if d < 1:
        return a.astype(np.float32)

    wet = np.zeros((a.shape[0], a.shape[1] + d * taps))
    for k in range(1, taps + 1):
        g = feedback ** k
        if g < 1e-4:
            break
        wet[:, d * k:d * k + a.shape[1]] += a * g
    if damp_hz:
        # Each repeat of a real delay loses top end. Damping the whole wet path
        # is a cheap approximation that reads as "further away".
        wet = filters.lowpass(wet, sample_rate, damp_hz)

    out = np.zeros((a.shape[0], wet.shape[1]))
    out[:, :a.shape[1]] += a
    out += wet * mix
    return out.astype(np.float32)


def echo_tail(audio, sample_rate, delay_s, feedback=0.55, taps=10,
              damp_hz=6000.0):
    """Just the repeats, with no dry signal -- for echoing a track out."""
    a = np.atleast_2d(audio).astype(np.float64)
    d = int(round(delay_s * sample_rate))
    if d < 1:
        return np.zeros_like(a, dtype=np.float32)
    wet = np.zeros((a.shape[0], a.shape[1] + d * taps))
    for k in range(1, taps + 1):
        g = feedback ** k
        if g < 1e-4:
            break
        wet[:, d * k:d * k + a.shape[1]] += a * g
    if damp_hz:
        wet = filters.lowpass(wet, sample_rate, damp_hz)
    return wet.astype(np.float32)


def loop_roll(audio, sample_rate, loop_s, total_s, halve_every=None):
    """Repeat a slice, optionally halving its length as it goes.

    The classic build into a drop: a one-bar loop becomes half a bar, then a
    beat, then half a beat, so the repeats accelerate while the pitch stays put.
    """
    a = np.atleast_2d(audio).astype(np.float64)
    total = int(round(total_s * sample_rate))
    out = np.zeros((a.shape[0], total))
    pos, cur = 0, float(loop_s)
    reps = 0
    while pos < total and cur * sample_rate >= 64:
        n = int(round(cur * sample_rate))
        chunk = a[:, :n]
        if chunk.shape[1] < n:
            chunk = np.pad(chunk, ((0, 0), (0, n - chunk.shape[1])))
        take = min(n, total - pos)
        # 3 ms fade at each end: looping mid-waveform otherwise clicks.
        fade = max(1, int(0.003 * sample_rate))
        piece = chunk[:, :take].copy()
        if take > 2 * fade:
            piece[:, :fade] *= np.linspace(0, 1, fade)
            piece[:, -fade:] *= np.linspace(1, 0, fade)
        out[:, pos:pos + take] += piece
        pos += take
        reps += 1
        if halve_every and reps % halve_every == 0:
            cur /= 2.0
    return out.astype(np.float32)


def riser(sample_rate, length_s, f0=300.0, f1=9000.0, q=1.6, shape="ease_in",
          seed=0):
    """White noise swept upward through a resonant bandpass.

    Synthesised, not sampled -- so it is always exactly the right length for
    the bars it has to fill, always at the mix tempo, and never a stock sample
    someone else's mix also uses. The rising resonant peak is what the ear
    tracks; the noise underneath is just the fuel for it.

    Written block-by-block because the filter coefficients change continuously.
    Recomputing them per sample would be correct and unusably slow; per block
    of ~10 ms the sweep is smooth and the cost is trivial.
    """
    from scipy.signal import butter, sosfilt, sosfilt_zi

    n = int(round(length_s * sample_rate))
    if n < 64:
        return np.zeros((2, max(0, n)), dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n) * 0.5

    block = max(64, int(0.010 * sample_rate))
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    if shape == "ease_in":
        curve = t ** 2.2
    elif shape == "ease_out":
        curve = 1.0 - (1.0 - t) ** 2.2
    else:
        curve = t
    # Sweep in log frequency: linear in Hz spends most of its time in the top
    # octave, where the ear hears almost no movement.
    freqs = f0 * (f1 / f0) ** curve

    out = np.zeros(n)
    zi = None
    nyq = sample_rate / 2.0
    for s in range(0, n, block):
        e = min(n, s + block)
        fc = float(np.clip(freqs[s], 20.0, nyq * 0.92))
        bw = fc / q
        lo = max(10.0, fc - bw / 2) / nyq
        hi = min(0.97, (fc + bw / 2) / nyq)
        if hi <= lo:
            hi = min(0.97, lo * 1.15 + 1e-4)
        sos = butter(2, [lo, hi], btype="band", output="sos")
        if zi is None or zi.shape[0] != sos.shape[0]:
            zi = sosfilt_zi(sos) * 0.0
        chunk, zi = sosfilt(sos, noise[s:e], zi=zi)
        out[s:e] = chunk

    # Amplitude rises with the sweep, so it arrives rather than just existing.
    env = curve ** 1.4
    peak = float(np.max(np.abs(out))) or 1.0
    out = out / peak * env
    return np.vstack([out, out]).astype(np.float32)


def downlifter(sample_rate, length_s, f0=6000.0, f1=120.0, seed=1):
    """The mirror image: a falling sweep to sit under a drop's first bar."""
    return riser(sample_rate, length_s, f0=f0, f1=f1, q=1.2,
                 shape="ease_out", seed=seed)
