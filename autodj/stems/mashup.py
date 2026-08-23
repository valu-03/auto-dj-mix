"""Building things out of stems: real bass swaps, and acapella mashups.

Phase 3 already swaps bass by *frequency* -- cut everything under 200 Hz on the
outgoing track, open it on the incoming one. That works, but 200 Hz is not a
line between instruments. It runs straight through the kick, the low end of the
vocal, and the body of every synth stab. Cutting there thins the whole track,
not just the bassline.

A stem swap cuts by *source* instead. The bass stem leaves and the other three
stay exactly as they were -- full-bandwidth kick, full-bodied vocal, nothing
else touched. That is the difference this phase buys.

Everything here works in bars, never seconds. A mashup that is a beat out is
worse than no mashup at all.
"""

import numpy as np

from .. import audio as audio_mod
from ..analysis import track as track_mod
from ..dsp import automation, master, stretch
from . import separate

STEM_NAMES = separate.STEM_NAMES


def load_stems(path, meta=None, sample_rate=None, names=STEM_NAMES, **kw):
    """Separate if needed, then decode the stems we asked for.

    Returns {name: (channels, frames)} all at the same rate and length. Demucs
    reconstructs each stem at the original length, but FLAC round-tripping can
    leave a sample or two of difference, and every operation downstream assumes
    the four arrays line up. So they are trimmed to a common length here rather
    than defensively everywhere else.
    """
    sample_rate = sample_rate or audio_mod.RENDER_RATE
    paths, _ = separate.separate(path, **kw)
    out = {}
    for name in names:
        if name not in paths:
            raise KeyError(f"no {name!r} stem for {path}")
        out[name], _ = audio_mod.load(paths[name], sample_rate)
    n = min(a.shape[1] for a in out.values())
    return {k: v[:, :n] for k, v in out.items()}


def recombine(stems, gains=None):
    """Sum stems back together, optionally with a gain per stem.

    With no gains this is the identity: Demucs is trained so the four stems add
    back to the input. Worth checking once (`residual` below) -- if it does not
    hold, something upstream resampled or clipped a stem and every mashup built
    on it will be subtly wrong.
    """
    gains = gains or {}
    items = list(stems.items())
    n = min(a.shape[1] for _, a in items)
    out = np.zeros((items[0][1].shape[0], n), dtype=np.float32)
    for name, a in items:
        g = gains.get(name, 1.0)
        if g:
            out += np.asarray(a[:, :n], dtype=np.float32) * np.float32(g)
    return out


def residual(original, stems):
    """How far the stems are from summing back to the source, in dB."""
    mix = recombine(stems)
    n = min(mix.shape[1], original.shape[1])
    err = mix[:, :n] - original[:, :n]
    ref = float(np.sqrt(np.mean(original[:, :n] ** 2))) + 1e-12
    return round(float(20 * np.log10(np.sqrt(np.mean(err ** 2)) / ref)), 2)


def drop(stems, *names):
    """Everything except the named stems -- `drop(s, 'vocals')` is the instrumental."""
    return recombine({k: v for k, v in stems.items() if k not in names})


def only(stems, *names):
    """Just the named stems -- `only(s, 'vocals')` is the acapella."""
    return recombine({k: v for k, v in stems.items() if k in names})


def bar_slice(audio, meta, sample_rate, start_bar, bars):
    """Cut a whole number of bars out of a track, using its fitted grid.

    Slicing by bar rather than by time is the whole reason the beatgrid was
    worth getting right. `bar_time` extrapolates, so this is safe past the last
    detected downbeat; short reads are zero-padded so the caller always gets
    exactly the length it asked for.
    """
    t0 = track_mod.bar_time(meta, start_bar)
    t1 = track_mod.bar_time(meta, start_bar + bars)
    want = int(round((t1 - t0) * sample_rate))
    a = audio_mod.clip(audio, sample_rate, max(0.0, t0), max(0.0, t1))
    if a.shape[1] < want:
        a = np.pad(a, ((0, 0), (0, want - a.shape[1])))
    return a[:, :want]


def align_to_bpm(audio, meta, to_bpm, backend="phase_vocoder"):
    """Stretch a stem to the mix tempo. Same rate the full track would get.

    `stretch_to_bpm` returns (audio, rate); only the audio is wanted here, and
    every stem of one track gets the identical rate because they share a
    beatgrid -- so stretching them separately cannot pull them apart.
    """
    out, _ = stretch.stretch_to_bpm(audio, meta["bpm"], to_bpm, backend=backend)
    return out


def acapella_over(vocal_meta, vocal_stems, bed_meta, bed_audio, sample_rate,
                  mix_bpm, vocal_bar, bed_bar, bars=16, duck_db=-3.0,
                  drop_bed_vocals=True):
    """A's vocal riding B's instrumental for `bars` bars.

    Three things have to be true at once or it sounds like an accident:

    tempo    both sides stretched to the same mix BPM, from their own fitted
             grids -- not resampled to each other, which would detune one.
    phase    both cut on a downbeat, so bar 1 of the vocal lands on bar 1 of
             the bed. This is what `bar_slice` is for.
    space    the bed's own vocal is removed, otherwise two singers overlap.
             That is only possible with stems, and it is the reason this
             function exists rather than a filter-based version.

    Key still has to match -- that is the planner's job, and it is why the
    Camelot distance is a hard constraint there. Two compatible keys sound
    intentional; two incompatible ones sound broken no matter how tight the
    timing is.
    """
    voc = only(vocal_stems, "vocals")
    voc = bar_slice(voc, vocal_meta, sample_rate, vocal_bar, bars)
    voc = align_to_bpm(voc, vocal_meta, mix_bpm)

    bed = (drop(bed_audio, "vocals") if isinstance(bed_audio, dict)
           else bed_audio)
    if drop_bed_vocals and not isinstance(bed_audio, dict):
        raise ValueError("bed must be a stem dict to remove its vocals")
    bed = bar_slice(bed, bed_meta, sample_rate, bed_bar, bars)
    bed = align_to_bpm(bed, bed_meta, mix_bpm)

    n = min(voc.shape[1], bed.shape[1])
    voc, bed = voc[:, :n], bed[:, :n]

    # Match the vocal to the bed by loudness before mixing. A vocal stem is
    # much quieter than a full track, so a straight sum buries it; matching
    # first means `duck_db` means the same thing on every pair.
    voc, _ = master.match_loudness(
        voc, sample_rate, master.measure_lufs(bed, sample_rate) + duck_db)
    return np.asarray(bed + voc, dtype=np.float32)


def stem_bass_swap(out_stems, in_stems, sample_rate, spb, bars=16, at=0.5,
                   width=0.02):
    """The bass-swap transition, cutting by source instead of by frequency.

    Same handover shape as `transitions.bass_swap` -- outgoing bass reaches
    zero *before* incoming bass rises, never a crossfade -- but applied to the
    bass stem alone. The kick stays in the drums stem and is untouched, so the
    outgoing track keeps its punch right up to the handover instead of being
    hollowed out under 200 Hz for sixteen bars.
    """
    n = int(round(bars * spb))
    pts_out = [(0, 1), (at - width, 1), (at, 0)]
    pts_in = [(0, 0), (at, 0), (at + width, 1)]
    g_out = automation.fit(
        automation.by_bar(bars, pts_out, spb, "ease_out"), n)
    g_in = automation.fit(
        automation.by_bar(bars, pts_in, spb, "ease_in"), n)

    fade_out, fade_in = automation.equal_power(n)

    def side(stems, bass_gain, rest_gain):
        rest = drop(stems, "bass")[:, :n]
        bass = only(stems, "bass")[:, :n]
        m = min(rest.shape[1], bass.shape[1], len(bass_gain))
        return rest[:, :m] * rest_gain[:m] + bass[:, :m] * bass_gain[:m]

    a = side(out_stems, g_out, fade_out)
    b = side(in_stems, g_in, fade_in)
    m = min(a.shape[1], b.shape[1])
    return np.asarray(a[:, :m] + b[:, :m], dtype=np.float32)
