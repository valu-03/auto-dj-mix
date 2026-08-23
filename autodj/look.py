import matplotlib.pyplot as plt
import numpy as np

from autodj import audio, spectral

a, sr = audio.load("musica/06 - Dr Alban - Let The Beat Go On.mp3",
                   audio.ANALYSIS_RATE, mono=True)
seg = audio.clip(a, sr, 60, 68)              # 8 seconds from the middle

mag = spectral.magnitude(seg)
db = spectral.to_db(mag)
t = spectral.frame_times(mag.shape[1], sr)

print("spectrogram shape:", mag.shape, "= (freq bins, frames)")
print("seconds per frame:", round(t[1] - t[0], 4))

fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1]})

ax[0].imshow(db, origin="lower", aspect="auto", cmap="magma",
             extent=[t[0], t[-1], 0, sr / 2])
ax[0].set_ylim(0, 4000)
ax[0].set_ylabel("Hz")
ax[0].set_title("Dr Alban - Let The Beat Go On   (60-68s)")

for band, name in ((spectral.LOW_BAND, "low (kick/bass)"),
                   (spectral.MID_BAND, "mid (vocal/snare)"),
                   (spectral.HIGH_BAND, "high (hats)")):
    e = spectral.band_energy(mag, sr, band)
    ax[1].plot(t, e / e.max(), label=name, linewidth=1.2)

ax[1].legend(loc="upper right", fontsize=8)
ax[1].set_xlabel("seconds")
ax[1].set_ylabel("energy")
plt.tight_layout()
plt.savefig("spectrogram.png", dpi=110)
print("wrote spectrogram.png")