"""Time-frequency analysis: the STFT and what we read off it."""

import librosa
import numpy as np

N_FFT = 2048
HOP = 512

LOW_BAND = (20.0, 200.0)
MID_BAND = (200.0, 2000.0)
HIGH_BAND = (2000.0, 11000.0)


def as_mono_1d(audio):
    """Accept (frames,) or (channels, frames); return a 1-D mono array."""
    a = np.asarray(audio)
    if a.ndim == 1:
        return a
    if a.shape[0] == 1:
        return a[0]
    return a.mean(axis=0)


def stft(audio, n_fft=N_FFT, hop=HOP):
    """Complex spectrogram, shape (freq_bins, frames)."""
    return librosa.stft(as_mono_1d(audio), n_fft=n_fft, hop_length=hop)


def magnitude(audio, n_fft=N_FFT, hop=HOP):
    """How much of each frequency is present. Phase discarded."""
    return np.abs(stft(audio, n_fft, hop))


def to_db(mag, floor_db=-80.0):
    """Convert magnitude to decibels, relative to the loudest point."""
    return librosa.amplitude_to_db(mag, ref=np.max, top_db=-floor_db)


def frame_times(n_frames, sample_rate, hop=HOP):
    """The time in seconds of each spectrogram column."""
    return librosa.frames_to_time(np.arange(n_frames), sr=sample_rate, hop_length=hop)


def bin_freqs(sample_rate, n_fft=N_FFT):
    """The centre frequency in Hz of each spectrogram row."""
    return librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)


def band_power(mag, sample_rate, band, n_fft=N_FFT):
    """Total energy inside a band, as a single number.

    Not interchangeable with `band_energy`: that returns RMS per frame, and RMS
    values do NOT sum across bands. Dividing two RMS figures to get an "energy
    fraction" gives a meaningless number -- it read 26% of a track's energy as
    sitting above 8 kHz. Power (sum of squares) is additive, so fractions
    computed from it are real fractions.
    """
    freqs = bin_freqs(sample_rate, n_fft)
    low, high = band
    rows = (freqs >= low) & (freqs < high)
    if not rows.any():
        return 0.0
    return float(np.sum(mag[rows].astype(np.float64) ** 2))


def band_energy(mag, sample_rate, band, n_fft=N_FFT):
    """RMS energy per frame inside a (low, high) Hz range."""
    freqs = bin_freqs(sample_rate, n_fft)
    low, high = band
    rows = (freqs >= low) & (freqs < high)
    if not rows.any():
        return np.zeros(mag.shape[1])
    return np.sqrt(np.mean(mag[rows] ** 2, axis=0))