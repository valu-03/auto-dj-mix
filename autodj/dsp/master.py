"""The master chain: loudness matching, limiting, dither."""

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import lfilter

# -10.5 LUFS, matched to the source material rather than to a streaming spec.
#
# History, because it reversed twice. -9 was the first choice and failed: the
# chain was gain -> limiter with nothing in between, so reaching it meant ~10 dB
# of pure limiting, which measured 69% gain reduction and came out at -16 LUFS.
# The conclusion drawn then -- "a target you cannot reach makes you quieter" --
# was correct about that chain, and the target was dropped to -14.
#
# But the real fault was the chain, not the target. Measured against the actual
# sources: La Bouche's own master is -10.59 LUFS at 11.1 dB crest, Le Click
# -14.02 at 12.6 dB, while our output sat at -17.2 LUFS and 16.0 dB crest --
# 6.6 dB quieter than the material it was built from, which is why it sounded
# small and distant next to anything commercial. With multiband compression and
# soft clipping ahead of the gain stage, -10.5 is reachable with the limiter
# doing touch-up, not rescue.
#
# MAX_PEAK_BEFORE_LIMIT is 4.0 rather than 1.6 for the same reason: the cap
# exists to stop the limiter being handed an impossible job, and once the
# compressor has taken the crest down, that much headroom is no longer needed.
TARGET_LUFS = -12.3
CEILING = 0.97          # ~ -0.26 dBFS, leaving headroom for encoder overshoot
MAX_PEAK_BEFORE_LIMIT = 4.0     # see below

# Level each track is matched to *on the way into the mix bus*. Deliberately
# NOT the master target: per-track matching exists to make the tracks agree
# with each other, and the loudness push belongs at the end, once. Feeding the
# bus at -11 with a 4.0 peak cap would let material arrive up to 12 dB over
# full scale, and the multiband thresholds are absolute dBFS -- the compressor
# would then crush everything before the gain stage ever ran. Leaving headroom
# here costs nothing: it is float, and the master makes it up.
BUS_LUFS = -16.0


def robust_peak(audio, quantile=99.9, block=1024):
    """Peak level ignoring isolated outliers.

    Using the absolute maximum to set gain is a trap on long material: a single
    freak transient anywhere in a 35-minute mix would hold the gain of the whole
    thing down. Taking a high quantile of per-block maxima describes the level
    the music actually sits at, and leaves the true outliers to the limiter --
    which is what a limiter is for.
    """
    a = np.atleast_2d(audio)
    if a.size == 0:
        return 0.0
    env = np.max(np.abs(a), axis=0)
    n = (len(env) // block) * block
    if n < block:
        return float(np.max(env))
    peaks = env[:n].reshape(-1, block).max(axis=1)
    return float(np.percentile(peaks, quantile))


def measure_lufs(audio, sample_rate):
    """EBU R128 integrated loudness."""
    a = np.atleast_2d(audio)
    y = a.mean(axis=0).astype(np.float64) if a.shape[0] > 1 else a[0].astype(np.float64)
    try:
        import pyloudnorm
        return float(pyloudnorm.Meter(sample_rate).integrated_loudness(y))
    except Exception:
        return 20.0 * np.log10(float(np.sqrt(np.mean(y ** 2))) + 1e-12)


def gain_for(current_lufs, target=TARGET_LUFS):
    """Linear gain that moves `current_lufs` to `target`."""
    if not np.isfinite(current_lufs) or current_lufs < -70:
        return 1.0
    return float(10.0 ** ((target - current_lufs) / 20.0))


def match_loudness(audio, sample_rate, target=TARGET_LUFS,
                   max_peak=MAX_PEAK_BEFORE_LIMIT):
    """Scale a track towards a target loudness, without setting up the limiter
    to fail.

    Loudness and peak are independent: a lightly-compressed track can be 8 dB
    quieter in LUFS while already peaking at full scale. Applying the full LUFS
    gain then hands the limiter several dB of work on every transient, and the
    result comes out *quieter* than the target as well as squashed. Capping the
    gain so peaks stay under `max_peak` keeps the limiter doing touch-up.
    """
    a = np.atleast_2d(audio)
    g = gain_for(measure_lufs(a, sample_rate), target)
    peak = robust_peak(a)
    if peak > 0 and peak * g > max_peak:
        g = max_peak / peak
    return (a * g).astype(np.float32), g


def limit(audio, sample_rate, ceiling=CEILING, lookahead_ms=3.0,
          release_ms=60.0):
    """Look-ahead peak limiter.

    Three steps, and the ordering is what makes it safe:

    1. `need` -- the gain each sample would require on its own.
    2. Running *minimum* over the release window. Gain now starts coming down
       before a peak arrives, instead of clamping the instant it hits (which is
       what makes a plain clipper sound like distortion).
    3. Smooth with a moving average narrower than the running-min window.

    Because step 2 already took the minimum over a *wider* span than step 3
    averages over, every value inside the smoothing window is <= `need` at the
    centre -- so the smoothed gain can never exceed what was needed. The
    ceiling is guaranteed without a final clip.
    """
    a = np.atleast_2d(audio).astype(np.float64)
    peak = np.max(np.abs(a), axis=0)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))

    w = max(3, int(lookahead_ms * 1e-3 * sample_rate))
    # The running-min window controls how long a peak keeps the gain down. Set
    # it to the full release time and one transient ducks a third of a second of
    # music. A quarter of the release is enough to stay smooth while letting the
    # gain recover between kicks.
    r = max(w, int(release_ms * 1e-3 * sample_rate) // 4)
    g = minimum_filter1d(need, size=2 * r + 1, mode="nearest")
    g = uniform_filter1d(g, size=2 * w + 1, mode="nearest")
    return (a * g).astype(np.float32), float(1.0 - g.min())


def compress(audio, sample_rate, threshold_db=-2.0, ratio=1.5,
             attack_ms=20.0, release_ms=150.0, knee_db=6.0, makeup=True):
    """Soft-knee compressor.

    A true compressor's envelope is recursive -- each sample's gain depends on
    the last -- which in Python would mean millions of iterations. Two tricks
    make it vectorised: a maximum filter gives the fast attack (the envelope
    must rise *before* the transient), and a one-pole IIR run by `lfilter`
    gives the slow release. Attack and release therefore get separate,
    correctly asymmetric behaviour without a sample loop.

    This sits before the limiter and does the gentle, musical part of the gain
    reduction, so the limiter is only catching what escapes.
    """
    a = np.atleast_2d(audio).astype(np.float64)
    if ratio <= 1.0 or a.size == 0:
        return a.astype(np.float32), 0.0

    level = np.max(np.abs(a), axis=0)
    att = max(1, int(attack_ms * 1e-3 * sample_rate))
    env = maximum_filter1d(level, size=2 * att + 1, mode="nearest")
    rel = float(np.exp(-1.0 / max(1.0, release_ms * 1e-3 * sample_rate)))
    env = lfilter([1.0 - rel], [1.0, -rel], env)

    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))
    over = env_db - threshold_db
    # Soft knee: ease into compression over `knee_db` instead of switching on.
    gr = np.zeros_like(over)
    half = knee_db / 2.0
    knee = (over > -half) & (over <= half)
    gr[knee] = (1.0 / ratio - 1.0) * (over[knee] + half) ** 2 / (2.0 * knee_db)
    above = over > half
    gr[above] = (1.0 / ratio - 1.0) * over[above]

    # Measure the reduction BEFORE makeup: makeup shifts the whole curve up, so
    # measuring afterwards reported 0.00 dB on a compressor that was visibly
    # working.
    reduction = float(-gr.min())
    if makeup:
        gr -= np.median(gr[gr < -0.01]) if np.any(gr < -0.01) else 0.0
    g = 10.0 ** (gr / 20.0)
    return (a * g).astype(np.float32), reduction


MULTIBAND = (
    # (name, threshold_db, ratio, attack_ms, release_ms)
    # The low band gets the most work and the fastest attack: in eurodance the
    # kick owns the peaks, and every dB of kick peak taken off here is a dB the
    # whole mix can come up. Mids are treated gently -- that is where the vocal
    # lives and over-compressing it is what makes a master sound squashed.
    # Highs get a slow attack so hats keep their snap.
    # Tuned by sweep, not by taste. Aggressive settings (low -20/3.0) and these
    # reached the SAME final loudness -- -11.62 LUFS either way -- because the
    # soft clipper and limiter ceiling set the end point, not the compressor.
    # The aggressive version simply did 10.2 dB of low-band gain reduction to
    # get there instead of 6.4. Same destination, more damage: take the less
    # compressed road to the identical number.
    ("low",  -13.0, 1.8, 10.0, 160.0),
    ("mid",  -12.0, 1.4, 25.0, 200.0),
    ("high", -15.0, 1.3, 30.0, 160.0),
)


def multiband_compress(audio, sample_rate, bands=MULTIBAND):
    """Compress low, mid and high independently, then sum.

    This is the stage that was missing, and it is why the mix could not get
    loud. Broadband compression is governed by whatever is loudest, which in
    this genre is always the kick -- so the kick pumps the vocal and the cymbals
    every time it lands, and the limiter still sees a 16 dB crest. Splitting
    first means the kick only ducks the low band. The mid and high bands keep
    their own levels, the crest factor comes down where it actually matters,
    and the result is denser without sounding pumped.

    Uses the same perfect-reconstruction split as the DJ EQ (`mid = x - low -
    high`), so with all three compressors bypassed this is exactly the identity.
    """
    from . import filters
    a = np.atleast_2d(audio).astype(np.float64)
    split = dict(zip(("low", "mid", "high"), filters.split(a, sample_rate)))
    out = np.zeros_like(a)
    report = {}
    for name, thr, ratio, att, rel in bands:
        band, gr = compress(split[name], sample_rate, threshold_db=thr,
                            ratio=ratio, attack_ms=att, release_ms=rel,
                            knee_db=6.0, makeup=False)
        out += band
        report[f"gr_{name}"] = round(gr, 2)
    return out.astype(np.float32), report


# Spectral balance measured off DJ Mario Andretti's "Megamix EuroDance 90's
# Vol 1": low / mid / high as a fraction of total power. Our own render came
# out at 61.3 / 30.1 / 8.6 -- bassier and noticeably duller. That gap is not
# only a matter of taste: LUFS is K-weighted, which de-emphasises bass and
# lifts treble, so a bass-heavy mix has to run a *higher* RMS to reach the same
# loudness. Correcting the tilt makes it brighter AND lets it hit the target
# with less compression.
REFERENCE_TILT = (0.575, 0.312, 0.113)


def tilt_to(audio, sample_rate, target=REFERENCE_TILT, max_db=1.5):
    """Nudge the low/high balance toward a reference profile with two shelves.

    Deliberately gentle and deliberately broad. This is not corrective EQ for a
    specific problem -- it is the difference between a mix that sounds like it
    was mastered and one that sounds like a render. Shelves rather than bells
    because the change has to be inaudible as an *event*: no corner you can
    point at, just a mix that sits differently.

    Capped at `max_db` because a large tilt here means something upstream is
    wrong (a bad source, a broken EQ curve) and quietly papering over it would
    hide the real fault.
    """
    from .. import spectral
    from . import filters
    from scipy.signal import sosfilt

    a = np.atleast_2d(audio).astype(np.float32)
    mag = spectral.magnitude(a.mean(axis=0))
    p = np.array([spectral.band_power(mag, sample_rate, b) for b in
                  (spectral.LOW_BAND, spectral.MID_BAND, spectral.HIGH_BAND)])
    have = p / (p.sum() + 1e-20)
    want = np.asarray(target, dtype=float)

    # Ratio of ratios, in dB, relative to the mid band -- the mid is the anchor
    # because it carries the vocal and must not move.
    low_db = float(np.clip(10 * np.log10((want[0] / have[0]) *
                                         (have[1] / want[1])), -max_db, max_db))
    high_db = float(np.clip(10 * np.log10((want[2] / have[2]) *
                                          (have[1] / want[1])), -max_db, max_db))

    out = a.astype(np.float64)
    if abs(low_db) > 0.1:
        out = sosfilt(filters.shelf_sos(sample_rate, 200.0, low_db, "low"), out,
                      axis=-1)
    if abs(high_db) > 0.1:
        out = sosfilt(filters.shelf_sos(sample_rate, 4000.0, high_db, "high"),
                      out, axis=-1)
    return out.astype(np.float32), {"tilt_low_db": round(low_db, 2),
                                    "tilt_high_db": round(high_db, 2)}


def soft_clip(audio, knee=0.80):
    """Round off peaks instead of letting the limiter chase them.

    Above `knee` the transfer curve bends smoothly to an asymptote at 1.0, so a
    transient is shortened rather than gain-reduced. The limiter that follows
    then has far less to do, which is what keeps it from pumping at high
    density -- and the low-order harmonics this adds are the "glue" and warmth
    that a purely clean chain lacks.

    Below the knee it is exactly unity: quiet material passes untouched.
    """
    a = np.atleast_2d(audio).astype(np.float64)
    mag = np.abs(a)
    over = mag > knee
    if not np.any(over):
        return a.astype(np.float32), 0.0
    room = 1.0 - knee
    shaped = knee + room * np.tanh((mag[over] - knee) / room)
    out = a.copy()
    out[over] = np.sign(a[over]) * shaped
    return out.astype(np.float32), float(np.mean(over) * 100.0)


def dither(audio, bits=16):
    """TPDF dither: decorrelates quantisation error so it sounds like hiss
    rather than distortion on quiet fades."""
    lsb = 2.0 ** -(bits - 1)
    rng = np.random.default_rng(0)
    a = np.atleast_2d(audio).astype(np.float64)
    noise = (rng.random(a.shape) - rng.random(a.shape)) * lsb
    return (a + noise).astype(np.float32)


def master(audio, sample_rate, target=TARGET_LUFS, ceiling=CEILING,
           add_dither=False, multiband=True, glue=True, clip_knee=0.95,
           bands=None, tilt=True):
    """Full master chain. Returns (audio, report).

    Order matters, and each stage exists to make the next one's job smaller:

    1. multiband compression -- brings the crest factor down where it is caused
       (the kick), so the gain stage has something it can actually lift.
    2. glue compression -- gentle, broadband, ties the bands back together.
    3. gain to target -- now reachable, because 1 and 2 removed the peaks that
       were blocking it.
    4. soft clip -- rounds the remaining transients instead of gain-reducing
       them.
    5. limiter -- catches only what escapes 4, so it never has to pump.

    The earlier chain was 3 then 5 with nothing else, and that is why it was
    stuck 6 dB below the source material: with a 16 dB crest and a hard cap of
    4 dB of limiting, the loudness simply was not reachable. Adding stages
    before the gain is what makes the target achievable rather than aspirational.

    `add_dither` defaults to False: dither belongs only at the final
    quantisation to a fixed-point format. Adding it to a float master that is
    then MP3-encoded just adds noise for the encoder to spend bits on.
    """
    a = np.atleast_2d(audio).astype(np.float32)
    before = measure_lufs(a, sample_rate)
    rep = {"lufs_in": round(before, 2)}

    if tilt:
        # Before compression, so the compressor reacts to the corrected
        # spectrum rather than to a balance we are about to change.
        a, ti = tilt_to(a, sample_rate)
        rep.update(ti)
    if multiband:
        a, mb = multiband_compress(a, sample_rate, bands or MULTIBAND)
        rep.update(mb)
    if glue:
        a, gr = compress(a, sample_rate, threshold_db=-12.0, ratio=1.8,
                         attack_ms=30.0, release_ms=200.0, makeup=False)
        rep["gr_glue"] = round(gr, 2)

    rep["lufs_compressed"] = round(measure_lufs(a, sample_rate), 2)
    rep["crest_before"] = round(_crest(a), 2)

    g = gain_for(measure_lufs(a, sample_rate), target)
    peak = robust_peak(a)
    if peak > 0 and peak * g > MAX_PEAK_BEFORE_LIMIT:
        g = MAX_PEAK_BEFORE_LIMIT / peak
    a = (a * g).astype(np.float32)
    peak_before = float(np.max(np.abs(a))) if a.size else 0.0

    a, pct = soft_clip(a, knee=clip_knee * ceiling / CEILING) if clip_knee \
        else (a, 0.0)
    rep["soft_clipped_pct"] = round(pct, 3)

    a, reduction = limit(a, sample_rate, ceiling)
    if add_dither:
        a = dither(a)

    rep.update({
        "gain_db": round(20 * np.log10(g + 1e-12), 2),
        "peak_before_limiter": round(peak_before, 3),
        "max_gain_reduction": round(reduction, 4),
        "lufs_out": round(measure_lufs(a, sample_rate), 2),
        "crest_out": round(_crest(a), 2),
        "peak_out": round(float(np.max(np.abs(a))) if a.size else 0.0, 4),
    })
    return a, rep


def _crest(audio):
    """Peak-to-RMS in dB. The number that says how dense a master is.

    Commercial dance masters sit around 11-13 dB. Much above that and the mix
    sounds distant next to anything else; much below and it is squashed flat.
    """
    a = np.atleast_2d(audio).astype(np.float64)
    if a.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(a ** 2)))
    return float(20 * np.log10((float(np.max(np.abs(a))) + 1e-12) / (rms + 1e-12)))
