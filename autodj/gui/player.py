"""Playing the rendered mix inside the application.

Everything else in this project is offline: analyse, plan, render, write a file.
That was a deliberate choice -- it buys a phase vocoder and a 4-stem separation
pass per track that no realtime engine could afford -- but it left the program
unable to do the one thing a DJ tool exists for, which is let you *hear* it. You
rendered a mix and then left to play it somewhere else, which means every
judgement about whether the mix was any good happened outside the app.

So this is playback of an array we already hold in memory, not a media player.
No decoding, no file handle, no seeking through a container: the mix is a
`(2, N)` float32 array, and playing it is pushing interleaved frames at an
audio sink. That makes seeking exact and instant, which is what an "audition
this transition" button needs -- it is a change to one integer.

QAudioSink rather than QMediaPlayer for the same reason. QMediaPlayer wants a
URL and would mean writing a temporary file on every render and re-decoding it;
QAudioSink takes raw frames from a QIODevice, which is precisely what we have.
"""

import numpy as np
from PyQt6.QtCore import QIODevice, QObject, QTimer, pyqtSignal

try:
    from PyQt6.QtMultimedia import (QAudioFormat, QAudioSink, QMediaDevices)
    AVAILABLE = True
except ImportError:                       # pragma: no cover - depends on build
    AVAILABLE = False
    QAudioFormat = QAudioSink = QMediaDevices = None


def interleave(audio):
    """(channels, n) float32 -> flat interleaved float32, clipped to [-1, 1].

    Clipping here rather than trusting the master chain: the limiter guarantees
    the *file* is in range, but a caller may hand us an unmastered preview, and
    a float sink given 1.4 does not wrap politely -- it produces a burst of
    full-scale noise straight into whatever the user is wearing.
    """
    a = np.atleast_2d(np.asarray(audio, dtype=np.float32))
    if a.shape[0] == 1:
        a = np.vstack([a[0], a[0]])
    return np.clip(a[:2].T.reshape(-1), -1.0, 1.0).astype(np.float32)


class _Stream(QIODevice):
    """A read-only device over one numpy buffer.

    Sequential on purpose. Qt's own seek machinery would have to agree with our
    frame counter about where "here" is, and there is no benefit: we own the
    buffer, so moving the playhead is assigning an integer.
    """

    # How much silence the device is always willing to hand over once the mix
    # has run out. It has to be a real number rather than zero, because
    # `bytesAvailable` and `readData` must agree: the sink asks the first how
    # much there is and then calls the second. Saying "nothing left" and then
    # supplying silence anyway is what produced
    #   pullFromQIODeviceToRingbuffer cannot read from QIODevice: "Unknown error"
    # -- Qt read at a moment its own accounting said was empty.
    SILENCE_TAIL = 1 << 16

    def __init__(self, frames, parent=None):
        super().__init__(parent)
        self._buf = frames.tobytes()
        self._bytes_per_frame = 8          # 2 channels x float32
        self._pos = 0

    def isSequential(self):
        return True

    def _remaining(self):
        return max(0, len(self._buf) - self._pos)

    def bytesAvailable(self):
        # Never negative and never zero. `_pos` past the end used to make this
        # return a negative count, which Qt has no defined behaviour for.
        return self._remaining() + self.SILENCE_TAIL + super().bytesAvailable()

    def readData(self, maxlen):
        if maxlen <= 0:
            return b""
        if self._pos >= len(self._buf):
            # Silence rather than b"": handing an empty read to a running sink
            # makes it underrun and stop, and a sink that stopped by itself
            # cannot be resumed cleanly -- it has to be rebuilt.
            return bytes(min(int(maxlen), 4096))
        chunk = self._buf[self._pos:self._pos + int(maxlen)]
        self._pos += len(chunk)
        return chunk

    def writeData(self, _):
        return 0

    def frames_total(self):
        return len(self._buf) // self._bytes_per_frame

    def frame(self):
        return self._pos // self._bytes_per_frame

    def set_frame(self, frame):
        # Clamped to the buffer. Seeking past the end -- which `set_audio`
        # does whenever it keeps the position across a shorter mix -- left
        # `_pos` beyond `len(_buf)` and every later size calculation negative.
        total = self.frames_total()
        self._pos = int(min(max(0, int(frame)), total)) * self._bytes_per_frame

    def finished(self):
        return self._pos >= len(self._buf)


class Player(QObject):
    """Transport for one in-memory mix. Position is reported in seconds.

    Position comes from the device's read cursor, not from a wall clock. A
    clock drifts against the sink's buffer and the playhead slides away from
    the audio over a long mix; the read cursor is where the sink has actually
    got to, give or take the buffer, so it stays locked to what you can hear.
    """

    position = pyqtSignal(float)          # seconds
    state_changed = pyqtSignal(bool)      # True while playing
    ended = pyqtSignal()

    TICK_MS = 40                          # 25 fps: smooth playhead, cheap

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rate = 44100
        self.audio = None
        self._sink = None
        self._stream = None
        self._playing = False
        self._latency_frames = 0
        self._pending = 0.0
        self._volume = 0.9

        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------- source ---
    def set_audio(self, audio, rate):
        """Load a mix. Keeps the current position if the length is unchanged."""
        was = self._playing
        at = self.seconds()
        self.stop()
        self.audio = interleave(audio)
        self.rate = int(rate)
        if was:
            self.play(at)

    def duration(self):
        if self.audio is None or not self.rate:
            return 0.0
        return len(self.audio) / 2 / self.rate

    def has_audio(self):
        return self.audio is not None and len(self.audio) > 0

    # ---------------------------------------------------------- transport ---
    def _make_sink(self):
        fmt = QAudioFormat()
        fmt.setSampleRate(self.rate)
        fmt.setChannelCount(2)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
        device = QMediaDevices.defaultAudioOutput()
        if device is None or not device.isFormatSupported(fmt):
            raise RuntimeError("no audio device accepts 44.1 kHz stereo float")
        sink = QAudioSink(device, fmt, self)
        sink.setVolume(getattr(self, "_volume", 0.9))
        return sink

    def play(self, seconds=None):
        if not AVAILABLE:
            raise RuntimeError("PyQt6.QtMultimedia is not installed")
        if not self.has_audio():
            return
        at = self.seconds() if seconds is None else float(seconds)
        # A fresh sink per start. Reusing one across a seek means draining a
        # buffer that still holds the old position's audio, which is heard as a
        # short burst of the previous location before the new one arrives.
        self._teardown()
        self._sink = self._make_sink()
        # Parented to the sink, not to the Player: the device must outlive
        # every read the sink's own thread will make, and die with it. Parented
        # to the Player it accumulated one dead device per seek for the life of
        # the session, all of them still holding a copy of the mix.
        self._stream = _Stream(self.audio, self._sink)
        # The return value is checked. `QAudioSink.start` on a device that
        # failed to open does not fail -- it starts, and the backend then logs
        # "QIODevice::read (_Stream): device not open" once per pull, forever,
        # while playing silence. Better to say so once, here.
        if not self._stream.open(QIODevice.OpenModeFlag.ReadOnly):
            self._teardown()
            raise RuntimeError("could not open the playback stream")
        self._stream.set_frame(at * self.rate)
        self._sink.start(self._stream)
        # The sink reads ahead; the read cursor is therefore ahead of what is
        # audible by one buffer. Measuring it once and subtracting keeps the
        # playhead where the sound is instead of where the reader is.
        self._latency_frames = int(self._sink.bufferSize() / 8)
        self._playing = True
        self._timer.start()
        self.state_changed.emit(True)

    def pause(self):
        if self._sink is not None and self._playing:
            self._sink.suspend()
            self._playing = False
            self._timer.stop()
            self.state_changed.emit(False)

    def resume(self):
        if self._sink is not None and not self._playing:
            self._sink.resume()
            self._playing = True
            self._timer.start()
            self.state_changed.emit(True)
        elif self._sink is None:
            self.play()

    def toggle(self):
        if self._playing:
            self.pause()
        elif self._sink is not None:
            self.resume()
        else:
            self.play()

    def stop(self):
        at_end = self._stream.finished() if self._stream else False
        self._teardown()
        self._playing = False
        self._pending = 0.0
        self._timer.stop()
        self.state_changed.emit(False)
        self.position.emit(0.0)
        if at_end:
            self.ended.emit()

    def _teardown(self):
        """Stop playback and let the sink take its device down with it.

        The device is deliberately NOT closed here. `QAudioSink.stop()` returns
        before the FFmpeg backend's pull thread has finished with it, and that
        thread issues at least one more read; closing the device synchronously
        produced a burst of

            QIODevice::read (_Stream): device not open
            pullFromQIODeviceToRingbuffer cannot read from QIODevice

        on every seek, every audition and every re-render -- one pair per
        late pull. The stream is parented to its sink instead, so it stays
        open and valid until the sink is actually destroyed, which
        `deleteLater` schedules for a point where nothing is reading it.
        """
        sink, self._sink = self._sink, None
        self._stream = None
        if sink is not None:
            sink.stop()
            sink.deleteLater()

    def seek(self, seconds):
        """Move the playhead. Restarts the sink so nothing buffered survives."""
        at = float(np.clip(seconds, 0.0, max(0.0, self.duration() - 0.01)))
        if self._playing:
            self.play(at)
        else:
            self._teardown()
            self._pending = at
            self.position.emit(at)

    def set_volume(self, value):
        self._volume = float(np.clip(value, 0.0, 1.0))
        if self._sink is not None:
            self._sink.setVolume(self._volume)

    def seconds(self):
        if self._stream is not None:
            f = max(0, self._stream.frame() - self._latency_frames)
            return f / self.rate
        return float(getattr(self, "_pending", 0.0))

    def is_playing(self):
        return self._playing

    def _tick(self):
        self.position.emit(self.seconds())
        if self._stream is not None and self._stream.finished():
            self.stop()
