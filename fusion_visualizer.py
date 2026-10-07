# -*- coding: utf-8 -*-
"""
[FUSION] - real-time system-audio reactive overlay (Windows)

* Captures global system audio through WASAPI loopback (PyAudioWPatch)
* Translucent, frameless, draggable and edge-resizable window
* Dot-matrix spectrum driven by the live FFT of whatever your PC is playing
* Hover menu (colour / hue / speed / gain / opacity) that shrinks + fades when
  the mouse leaves the window
* Right click for: Always on top, Reconnect audio, Reset, Quit

Run with  --demo  to see the visual with synthetic data (no audio needed).
"""
import sys
import os
import json
import math
import random
import time
import threading
import functools
import operator
import ctypes

import numpy as np

try:
    import pyaudiowpatch as pyaudio
except Exception:  # pragma: no cover - lets the UI still start without audio lib
    pyaudio = None

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QPoint
from PySide6.QtGui import (QPainter, QColor, QFont, QFontMetricsF, QPen, QBrush,
                           QPainterPath, QLinearGradient, QRadialGradient,
                           QCursor, QFontDatabase, QImage, QPixmap, QGuiApplication)
from PySide6.QtWidgets import QApplication, QWidget, QMenu

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
COLS = 25            # spectrum columns
ROWS = 8             # spectrum rows
FFT_N = 4096
VW, VH = 640.0, 304.0   # virtual design canvas (taken from the reference image)

PW, PH, MENU_M = 214.0, 216.0, 14.0   # menu panel width / height / margin

FLOOR_DB, TOP_DB, AGC_TARGET_DB = -62.0, -8.0, -14.0

PRESETS = [  # name, hue(deg), sat, val
    ("ROSE", 7.0, 0.26, 0.77),
    ("ICE", 190.0, 0.45, 0.92),
    ("AMBER", 38.0, 0.70, 0.96),
    ("LIME", 95.0, 0.55, 0.88),
    ("VIOLET", 265.0, 0.45, 0.92),
    ("MAGENTA", 322.0, 0.55, 0.94),
    ("WHITE", 0.0, 0.00, 0.92),
]

SLIDERS = [  # key, label, min, max
    ("hue", "HUE", 0.0, 360.0),
    ("speed", "SPEED", 0.25, 3.0),
    ("gain", "GAIN", 0.0, 1.0),
    ("opacity", "OPACITY", 0.10, 1.0),
]

DEFAULTS = dict(hue=7.0, sat=0.26, val=0.77, speed=1.0, gain=0.5, opacity=0.82,
                topmost=True, win_x=None, win_y=None, win_w=720, win_h=620)


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def ease_out(t):
    return 1.0 - (1.0 - t) ** 3


def ease_io(t):
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
def settings_path():
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "FusionVisualizer")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return os.path.join(d, "settings.json")


def load_settings():
    cfg = dict(DEFAULTS)
    try:
        with open(settings_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        for k in DEFAULTS:
            if k in data:
                cfg[k] = data[k]
    except Exception:
        pass
    return cfg


def save_settings(cfg):
    try:
        with open(settings_path(), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# audio capture (WASAPI loopback = everything the PC is playing)
# --------------------------------------------------------------------------- #
def _find_loopback(p):
    try:
        return p.get_default_wasapi_loopback()
    except Exception:
        pass
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    speakers = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if speakers.get("isLoopbackDevice"):
        return speakers
    for lb in p.get_loopback_device_info_generator():
        if speakers["name"] in lb["name"]:
            return lb
    raise RuntimeError("No WASAPI loopback device found")


class AudioCapture:
    def __init__(self):
        self.size = FFT_N
        self.ring = np.zeros(self.size, np.float32)
        self.lock = threading.Lock()
        self.wpos = 0
        self.rate = 48000
        self.channels = 2
        self.last_data = 0.0
        self.running = True
        self.force_reconnect = False
        self.status = "starting..."
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False

    def reconnect(self):
        self.force_reconnect = True

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            a = np.frombuffer(in_data, dtype=np.int16)
            ch = self.channels
            if ch > 1:
                n = (a.size // ch) * ch
                a = a[:n].reshape(-1, ch).mean(axis=1)
            a = a.astype(np.float32) / 32768.0
            n = a.size
            if n >= self.size:
                a = a[-self.size:]
                n = self.size
            with self.lock:
                end = self.wpos + n
                if end <= self.size:
                    self.ring[self.wpos:end] = a
                else:
                    k = self.size - self.wpos
                    self.ring[self.wpos:] = a[:k]
                    self.ring[:n - k] = a[k:]
                self.wpos = end % self.size
            self.last_data = time.perf_counter()
        except Exception:
            pass
        return (in_data, pyaudio.paContinue)

    def snapshot(self):
        """Samples ordered oldest -> newest."""
        with self.lock:
            return np.roll(self.ring, -self.wpos).copy()

    def _run(self):
        while self.running:
            p = None
            stream = None
            try:
                p = pyaudio.PyAudio()
                info = _find_loopback(p)
                self.channels = max(1, int(info["maxInputChannels"]))
                self.rate = int(info["defaultSampleRate"])
                stream = p.open(format=pyaudio.paInt16, channels=self.channels,
                                rate=self.rate, input=True,
                                input_device_index=info["index"],
                                frames_per_buffer=1024,
                                stream_callback=self._callback)
                self.status = "live: " + str(info["name"])
                opened = time.perf_counter()
                self.last_data = opened
                self.force_reconnect = False
                while self.running and not self.force_reconnect:
                    time.sleep(0.25)
                    now = time.perf_counter()
                    if not stream.is_active():
                        break
                    # Windows delivers no packets while silent. Re-scan the default
                    # device every few seconds of silence so output switches are picked up.
                    if now - self.last_data > 2.0 and now - opened > 3.0:
                        break
            except Exception as e:
                self.status = "audio error: %s" % e
                time.sleep(2.0)
            finally:
                try:
                    if stream is not None:
                        stream.stop_stream()
                        stream.close()
                except Exception:
                    pass
                try:
                    if p is not None:
                        p.terminate()
                except Exception:
                    pass
            time.sleep(0.25)


# --------------------------------------------------------------------------- #
# spectrum analysis
# --------------------------------------------------------------------------- #
class Analyzer:
    def __init__(self, cols):
        self.cols = cols
        self.win = np.hanning(FFT_N).astype(np.float32)
        self.agc = -40.0
        self.rate = 0
        self._setup(48000)

    def _setup(self, rate):
        self.rate = rate
        freqs = np.fft.rfftfreq(FFT_N, 1.0 / rate)
        top = min(16000.0, rate / 2.0 * 0.92)
        edges = np.geomspace(45.0, top, self.cols + 1)
        self.bands = []
        for i in range(self.cols):
            lo = int(np.searchsorted(freqs, edges[i]))
            hi = int(np.searchsorted(freqs, edges[i + 1]))
            self.bands.append((lo, max(hi, lo + 1)))
        centers = np.sqrt(edges[:-1] * edges[1:])
        self.tilt = np.where(centers > 200.0, 3.0 * np.log2(np.maximum(centers, 1.0) / 200.0),
                             0.0).astype(np.float32)

    def process(self, samples, rate, gain_db, dt):
        if rate != self.rate:
            self._setup(rate)
        x = samples[-FFT_N:]
        if x.size < FFT_N:
            x = np.pad(x, (FFT_N - x.size, 0))
        spec = np.abs(np.fft.rfft(x * self.win)) / (FFT_N / 4.0)
        mags = np.fromiter((spec[lo:hi].max() for lo, hi in self.bands),
                           dtype=np.float32, count=self.cols)
        db = 20.0 * np.log10(mags + 1e-7) + self.tilt
        peak = float(db.max())
        if peak > -72.0:
            if peak > self.agc:
                self.agc += min(peak - self.agc, 60.0 * dt)
            else:
                self.agc -= 2.0 * dt
            self.agc = max(self.agc, -70.0)
        off = clamp(AGC_TARGET_DB - self.agc, 0.0, 36.0)
        norm = np.clip((db + off + gain_db - FLOOR_DB) / (TOP_DB - FLOOR_DB), 0.0, 1.0)
        return (norm ** 1.35).astype(np.float32)


# --------------------------------------------------------------------------- #
# text scramble effect
# --------------------------------------------------------------------------- #
class Scramble:
    G = "0123456789ABCDEF#%/<>+=*"

    def __init__(self, text, delay=0.0, dur=0.7):
        self.set(text, delay, dur)

    def set(self, text, delay=0.0, dur=0.7):
        self.target = text
        self.t = -delay
        self.dur = dur

    def update(self, dt):
        self.t += dt

    @property
    def done(self):
        return self.t >= self.dur

    def text(self):
        if self.t < 0:
            return ""
        if self.t >= self.dur:
            return self.target
        n = len(self.target)
        k = int(self.t / self.dur * n)
        return "".join(ch if (i < k or ch == " ") else random.choice(self.G)
                       for i, ch in enumerate(self.target))


# --------------------------------------------------------------------------- #
# main widget
# --------------------------------------------------------------------------- #
class FusionWidget(QWidget):
    def __init__(self, demo=False):
        super().__init__()
        self.cfg = load_settings()
        self.setWindowTitle("FUSION")
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window
        if self.cfg["topmost"]:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setMinimumSize(420, 400)
        self.resize(max(420, int(self.cfg["win_w"])), max(400, int(self.cfg["win_h"])))
        self._place_window()

        # ---- fonts
        fams = set(QFontDatabase.families())
        family = "monospace"
        for f in ("JetBrains Mono", "Space Mono", "Cascadia Mono", "Consolas", "Courier New"):
            if f in fams:
                family = f
                break
        self.family = family
        self.f_title = QFont(family)
        self.f_title.setPixelSize(self._fit_px("FUSION", 247.0, True))
        self.f_title.setBold(True)
        self.f_body = QFont(family)
        self.f_body.setPixelSize(self._fit_px("RESTART REQUIRED", 185.0, False))
        self.f_body.setBold(True)
        self.fm_body = QFontMetricsF(self.f_body)
        self.f_small = QFont(family)
        self.f_small.setPixelSize(11)
        self.f_small_b = QFont(family)
        self.f_small_b.setPixelSize(12)
        self.f_small_b.setBold(True)
        self.f_tiny = QFont(family)
        self.f_tiny.setPixelSize(10)

        # ---- audio
        self.audio = None
        if not demo and pyaudio is not None:
            self.audio = AudioCapture()
            self.audio.start()
        self.an = Analyzer(COLS)

        # ---- state
        self.t = 0.0
        self.boot = 0.0
        self.last = time.perf_counter()
        self.hover_t = 1.0
        self.outside = 0.0
        self.drag = None
        self.disp = np.zeros(COLS, np.float32)
        self.peak = np.zeros(COLS, np.float32)
        self.hold = np.zeros(COLS, np.float32)
        self.energy = 0.0
        self.bass_avg = 0.0
        self.beat = 0.0
        self.beat_count = 0
        self.last_beat = 0.0
        self.last_code = 0.0
        self.last_glitch = 0.0
        self.glitch = 0.0
        self.next_idle_glitch = time.perf_counter() + 6.0
        self.last_sound = time.perf_counter()
        self.idle_mix = 0.0
        self.status = "ACTIVE"
        self.temp = 64.0
        self.spin = 0.0
        self.ripples = []
        self.plus_ang = [0.0, 0.0]
        self.plus_tgt = [0.0, 0.0]
        self.plus_flash = [0.0, 0.0]
        self.flick = 1.0
        self.hex_acc = 0.0
        self.hex_counter = random.randint(0, 0xFFFF)
        self.hex_lines = ["%04X  %s" % (random.randint(0, 0xFFFF),
                                        " ".join("%02X" % random.randint(0, 255) for _ in range(9)))
                          for _ in range(6)]
        self.ch, self.cs, self.cv = self.cfg["hue"], self.cfg["sat"], self.cfg["val"]
        col = QColor.fromHsvF((self.ch % 360.0) / 360.0, clamp(self.cs), clamp(self.cv))
        self.rgb = (col.red(), col.green(), col.blue())

        self.scr = {
            "title": Scramble("FUSION", 0.25, 0.8),
            "l1": Scramble("CORE.BIN", 0.50, 0.6),
            "l2": Scramble("RESTORE", 0.60, 0.6),
            "l3": Scramble("ACTIVE", 0.70, 0.6),
            "r1": Scramble("FIRMWARE_02", 0.55, 0.7),
            "r2": Scramble("INSTALL", 0.65, 0.6),
            "r3": Scramble("RESTART REQUIRED", 0.75, 0.8),
            "code": Scramble("0x0A9F", 0.95, 0.7),
            "tag": Scramble("9F-88", 1.10, 0.6),
        }

        self.tile = self._make_tile()

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    # ------------------------------------------------------------------ setup
    def _fit_px(self, text, width, bold):
        f = QFont(self.family if hasattr(self, "family") else "monospace")
        f.setPixelSize(100)
        f.setBold(bold)
        w = QFontMetricsF(f).horizontalAdvance(text)
        return max(6, int(round(100.0 * width / max(w, 1.0))))

    def _place_window(self):
        x, y = self.cfg["win_x"], self.cfg["win_y"]
        ok = False
        if x is not None and y is not None:
            for s in QGuiApplication.screens():
                if s.availableGeometry().contains(QPoint(int(x) + 40, int(y) + 40)):
                    ok = True
        if ok:
            self.move(int(x), int(y))
        else:
            geo = QGuiApplication.primaryScreen().availableGeometry()
            self.move(geo.center().x() - self.width() // 2, geo.center().y() - self.height() // 2)

    def _make_tile(self):
        rng = np.random.default_rng(7)
        arr = np.zeros((120, 120, 4), np.uint8)
        arr[..., 0:3] = 255
        arr[..., 3] = rng.integers(0, 24, (120, 120), dtype=np.uint8)
        arr[::3, :, 0:3] = 0          # scanlines
        arr[::3, :, 3] = 34
        img = QImage(arr.data, 120, 120, 120 * 4, QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(img)

    # ---------------------------------------------------------------- helpers
    def C(self, a=1.0, light=0.0):
        r, g, b = self.rgb
        if light > 0:
            r += (255 - r) * light
            g += (255 - g) * light
            b += (255 - b) * light
        return QColor(int(clamp(r, 0, 255)), int(clamp(g, 0, 255)), int(clamp(b, 0, 255)),
                      int(clamp(a) * 255))

    def rv(self, t0, d):
        return ease_out(clamp((self.boot - t0) / d))

    def demo_levels(self):
        i = np.arange(COLS)
        t = self.t
        lv = 0.45 + 0.35 * np.sin(t * 1.7 + i * 0.38) * np.sin(t * 0.9 + i * 0.21) \
            + 0.15 * np.sin(t * 5.0 + i)
        pulse = max(0.0, math.sin(t * 2 * math.pi * 1.8)) ** 6
        lv[:5] += 0.45 * pulse * (1 - i[:5] / 6.0)
        lv *= np.linspace(1.0, 0.55, COLS)
        return np.clip(lv, 0, 1).astype(np.float32)

    # ------------------------------------------------------------------- tick
    def tick(self):
        now = time.perf_counter()
        dt = clamp(now - self.last, 0.001, 0.05)
        self.last = now
        sp = self.cfg["speed"]
        self.t += dt
        self.boot += dt

        # hover (menu + window morph)
        inside = self.frameGeometry().contains(QCursor.pos()) or self.drag is not None
        self.outside = 0.0 if inside else self.outside + dt
        want = 0.0 if self.outside > 0.4 else 1.0
        self.hover_t += (want - self.hover_t) * (1 - math.exp(-dt * 8))

        # accent colour easing (shortest hue arc)
        k = 1 - math.exp(-dt * 6)
        dh = ((self.cfg["hue"] - self.ch + 540.0) % 360.0) - 180.0
        self.ch = (self.ch + dh * k) % 360.0
        self.cs += (self.cfg["sat"] - self.cs) * k
        self.cv += (self.cfg["val"] - self.cv) * k
        col = QColor.fromHsvF((self.ch % 360.0) / 360.0, clamp(self.cs), clamp(self.cv))
        self.rgb = (col.red(), col.green(), col.blue())

        # audio -> band targets
        gain_db = (self.cfg["gain"] - 0.5) * 20.0
        if self.audio is None:
            raw = self.demo_levels()
        else:
            silent = (now - self.audio.last_data) > 0.25
            if silent:
                raw = np.zeros(COLS, np.float32)
            else:
                raw = self.an.process(self.audio.snapshot(), self.audio.rate, gain_db, dt)
        if float(raw.max()) > 0.06:
            self.last_sound = now
        quiet = now - self.last_sound

        # idle "breathing" when nothing is playing
        want_idle = 1.0 if (quiet > 0.8 and self.audio is not None) else 0.0
        self.idle_mix += (want_idle - self.idle_mix) * (1 - math.exp(-dt * 2.0))
        target = raw
        if self.idle_mix > 0.01:
            i = np.arange(COLS)
            idle = 0.12 + 0.07 * np.sin(i * 0.45 - self.t * 1.3 * sp) \
                + 0.05 * np.sin(i * 0.17 + self.t * 0.7 * sp)
            target = np.maximum(raw, np.clip(idle, 0, 1).astype(np.float32) * self.idle_mix)

        # smoothing (speed scales attack / release)
        att = 1 - math.exp(-dt * 38.0 * sp)
        rel = 1 - math.exp(-dt * 6.5 * sp)
        up = target > self.disp
        self.disp = np.where(up, self.disp + (target - self.disp) * att,
                             self.disp + (target - self.disp) * rel).astype(np.float32)
        newp = self.disp >= self.peak
        self.peak = np.where(newp, self.disp, self.peak).astype(np.float32)
        self.hold = np.where(newp, 0.35 / sp, self.hold - dt).astype(np.float32)
        fall = self.hold <= 0
        self.peak = np.where(fall, np.maximum(self.disp, self.peak - dt * 0.55 * sp),
                             self.peak).astype(np.float32)

        self.energy += (float(self.disp.mean()) - self.energy) * (1 - math.exp(-dt * 8))

        # beat detection (bass)
        bass = float(raw[:4].mean())
        self.bass_avg += (bass - self.bass_avg) * (1 - math.exp(-dt * 2.5))
        if bass > self.bass_avg * 1.28 + 0.06 and bass > 0.2 and now - self.last_beat > 0.2:
            self.last_beat = now
            self.beat = 1.0
            self.beat_count += 1
            idx = self.beat_count % 2
            self.plus_tgt[idx] += 90.0
            self.plus_flash[idx] = 1.0
            self.ripples.append(0.0)
            if bass > 0.55 and now - self.last_glitch > 0.5:
                self.last_glitch = now
                self.glitch = 1.0
            if now - self.last_code > 1.5 and self.beat_count % 3 == 0:
                self.last_code = now
                self.scr["code"].set("0x%04X" % random.randint(0, 0xFFFF), 0.0, 0.5)
                self.scr["tag"].set("%02X-%02X" % (random.randint(0, 255), random.randint(0, 255)),
                                    0.0, 0.5)
        self.beat *= math.exp(-dt * 4.5 * sp)
        self.glitch *= math.exp(-dt * 8.0)
        if now >= self.next_idle_glitch:
            self.glitch = max(self.glitch, 0.55)
            self.next_idle_glitch = now + random.uniform(7.0, 14.0)

        # status text
        want_status = "ACTIVE" if (quiet < 1.5 or self.audio is None) else "STANDBY"
        if want_status != self.status:
            self.status = want_status
            self.scr["l3"].set(want_status, 0.0, 0.5)

        # misc animation state
        self.temp += ((64.0 + self.energy * 32.0) - self.temp) * (1 - math.exp(-dt * 3.0))
        self.spin = (self.spin + dt * (60.0 + 260.0 * self.energy) * sp) % 360.0
        for j in (0, 1):
            self.plus_ang[j] += (self.plus_tgt[j] - self.plus_ang[j]) * (1 - math.exp(-dt * 10.0))
            self.plus_flash[j] *= math.exp(-dt * 6.0)
        self.ripples = [r + dt * 1.6 * sp for r in self.ripples if r + dt * 1.6 * sp < 1.0]
        for s in self.scr.values():
            s.update(dt)

        # rare CRT flicker
        if random.random() < dt * 0.25:
            self.flick = 0.78
        self.flick += (1.0 - self.flick) * (1 - math.exp(-dt * 14.0))

        # live hex readout
        self.hex_acc += dt * sp * 4.0
        while self.hex_acc >= 1.0:
            self.hex_acc -= 1.0
            self.hex_counter = (self.hex_counter + 1 + random.randint(0, 3)) & 0xFFFF
            vals = " ".join("%02X" % int(clamp(float(v)) * 255) for v in self.disp[::3])
            self.hex_lines.append("%04X  %s" % (self.hex_counter, vals))
            self.hex_lines = self.hex_lines[-6:]

        self.update()

    # ------------------------------------------------------------------ paint
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
                         | QPainter.RenderHint.SmoothPixmapTransform)
        W, H = self.width(), self.height()
        self.draw_bg(p, W, H)

        s = max(0.05, min((W - 24.0) / VW, (H - 24.0) / VH))
        ox = (W - VW * s) / 2.0
        oy = H - 12.0 - VH * s
        p.save()
        p.translate(ox, oy)
        p.scale(s, s)
        p.setOpacity(self.flick)
        self.draw_content(p)
        p.restore()

        if oy > 118:
            self.draw_hex(p)
        self.draw_menu(p, W, H)
        self.draw_frame(p, W, H)
        p.end()

    def draw_bg(self, p, W, H):
        a = clamp(self.cfg["opacity"] * (0.72 + 0.28 * self.hover_t), 0.04, 1.0)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0.5, 0.5, W - 1.0, H - 1.0), 8, 8)
        g = QLinearGradient(0, 0, 0, H)
        g.setColorAt(0.0, QColor(18, 16, 18, int(a * 255)))
        g.setColorAt(1.0, QColor(7, 7, 9, int(a * 255)))
        p.fillPath(path, QBrush(g))
        p.save()
        p.setClipPath(path)
        p.setOpacity(0.55 * math.sqrt(a))
        p.drawTiledPixmap(0, 0, W, H, self.tile)
        # sweeping scan beam
        p.setOpacity(1.0)
        sp = self.cfg["speed"]
        y = (self.t * sp * 70.0) % (H + 160.0) - 80.0
        bg = QLinearGradient(0, y - 40, 0, y + 40)
        bg.setColorAt(0.0, self.C(0.0))
        bg.setColorAt(0.5, self.C(0.085 + 0.08 * self.energy))
        bg.setColorAt(1.0, self.C(0.0))
        p.fillRect(QRectF(0, y - 40, W, 80), QBrush(bg))
        p.restore()

    def draw_frame(self, p, W, H):
        path = QPainterPath()
        path.addRoundedRect(QRectF(0.5, 0.5, W - 1.0, H - 1.0), 8, 8)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(self.C(0.18 + 0.22 * self.hover_t), 1.0))
        p.drawPath(path)
        # corner reticles
        p.setPen(QPen(self.C(0.35 + 0.35 * self.hover_t), 1.4))
        m, L = 6.0, 14.0
        for (x, y, dx, dy) in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)):
            p.drawLine(QPointF(x, y), QPointF(x + dx * L, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + dy * L))
        # resize grip hint
        p.setPen(QPen(self.C(0.25 + 0.45 * self.hover_t), 1.2))
        for i in range(3):
            o = 5 + i * 4
            p.drawLine(QPointF(W - 4, H - o), QPointF(W - o, H - 4))

    def draw_hex(self, p):
        p.save()
        p.setFont(self.f_tiny)
        st = self.audio.status if self.audio is not None else "demo signal"
        if len(st) > 34:
            st = st[:33] + "~"
        r = self.rv(0.3, 0.6)
        p.setPen(self.C(0.55 * r))
        p.drawText(QPointF(22, 24), "// " + st.upper())
        n = len(self.hex_lines)
        for i, ln in enumerate(self.hex_lines):
            a = (0.10 + 0.30 * (i + 1) / n) * r
            p.setPen(self.C(a, light=0.25 if i == n - 1 else 0.0))
            p.drawText(QPointF(22, 40 + i * 12), ln)
        p.restore()

    # ------------------------------------------------------- virtual content
    def draw_content(self, p):
        sp = self.cfg["speed"]
        beat = self.beat
        NP, NB = Qt.PenStyle.NoPen, Qt.BrushStyle.NoBrush
        rv = self.rv

        # ---- bloom behind the spectrum
        bl = clamp(self.energy * 1.3) * rv(0.8, 0.8)
        if bl > 0.01:
            rg = QRadialGradient(QPointF(155, 150), 190)
            rg.setColorAt(0.0, self.C(0.16 * bl))
            rg.setColorAt(1.0, self.C(0.0))
            p.setPen(NP)
            p.setBrush(QBrush(rg))
            p.drawRect(QRectF(-40, 0, 400, 304))

        # ---- [ ] brackets
        r = rv(0.10, 0.6)
        if r > 0:
            top, bot = 15.0, 84.0
            mid = (top + bot) / 2.0
            ext = (bot - top) / 2.0 * r
            t0, t1 = mid - ext, mid + ext
            ins = beat * 2.5
            pen = QPen(self.C(r, light=beat * 0.4), 3.2)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            p.setPen(pen)
            p.setBrush(NB)
            pl = QPainterPath()
            pl.moveTo(23 + ins, t0)
            pl.lineTo(11.6 + ins, t0)
            pl.lineTo(11.6 + ins, t1)
            pl.lineTo(23 + ins, t1)
            pr = QPainterPath()
            pr.moveTo(297 - ins, t0)
            pr.lineTo(308.4 - ins, t0)
            pr.lineTo(308.4 - ins, t1)
            pr.lineTo(297 - ins, t1)
            p.drawPath(pl)
            p.drawPath(pr)

        # ---- title (with beat-driven chromatic glitch)
        txt = self.scr["title"].text()
        if txt:
            p.setFont(self.f_title)
            g = self.glitch
            if g > 0.03:
                off = g * 4.5
                p.setPen(QColor(255, 60, 110, int(110 * g)))
                p.drawText(QPointF(38 - off, 74), txt)
                p.setPen(QColor(70, 220, 255, int(110 * g)))
                p.drawText(QPointF(38 + off, 74), txt)
            p.setPen(self.C(1.0, light=beat * 0.35))
            p.drawText(QPointF(38, 74), txt)
            if g > 0.25:
                sy = random.uniform(18, 70)
                p.save()
                p.setClipRect(QRectF(0, sy, 320, random.uniform(3, 9)))
                p.setPen(self.C(1.0, light=0.5))
                p.drawText(QPointF(38 + random.uniform(-9, 9) * g, 74), txt)
                p.restore()

        # ---- top rule lines (+ travelling highlight)
        r = rv(0.25, 0.5)
        if r > 0:
            p.setPen(QPen(self.C(0.9), 1.6))
            p.drawLine(QPointF(328, 14), QPointF(328 + 92 * r, 14))
            p.drawLine(QPointF(442, 14), QPointF(442 + 185 * r, 14))
            if r >= 0.999:
                hx = 442 + (self.t * sp * 60.0) % 215.0 - 15.0
                a0, a1 = max(442.0, hx), min(627.0, hx + 30.0)
                if a1 > a0:
                    p.setPen(QPen(self.C(1.0, light=0.55), 1.9))
                    p.drawLine(QPointF(a0, 14), QPointF(a1, 14))

        # ---- info columns
        p.setFont(self.f_body)
        rows = (("l1", 328, 47.0), ("l2", 328, 65.5), ("l3", 328, 84.0),
                ("r1", 442, 47.0), ("r2", 442, 65.5), ("r3", 442, 84.0))
        for key, x, y in rows:
            t = self.scr[key].text()
            if t:
                hot = (key == "l3" and self.status == "ACTIVE")
                p.setPen(self.C(0.95, light=(0.25 * self.energy if hot else 0.0)))
                p.drawText(QPointF(x, y), t)

        # ---- capsules
        r = rv(0.70, 0.6)
        if r > 0:
            w = 306.0 * r
            x = 322.0 + 153.0 * (1 - r)
            for top in (105.0, 151.0):
                p.setPen(QPen(self.C(0.92), 1.6))
                p.setBrush(self.C(0.045 + 0.05 * beat))
                p.drawRoundedRect(QRectF(x, top, w, 39.0), 19.5, 19.5)
            if r > 0.9:
                # capsule 1: knob, rotating plus signs, code
                self._knob(p, 345.0, 124.5, 13.0, self.spin, 60 + 220 * self.energy, beat)
                for j, cx in enumerate((382.0, 421.0)):
                    fl = self.plus_flash[j]
                    p.save()
                    p.translate(cx, 124.5)
                    p.rotate(self.plus_ang[j])
                    p.setPen(QPen(self.C(0.8 + 0.2 * fl, light=fl * 0.5), 1.6))
                    p.drawLine(QPointF(-13, 0), QPointF(13, 0))
                    p.drawLine(QPointF(0, -13), QPointF(0, 13))
                    p.restore()
                p.setFont(self.f_body)
                code = "CODE: " + self.scr["code"].text()
                p.setPen(self.C(0.95))
                p.drawText(QPointF(626.0 - self.fm_body.horizontalAdvance(code), 131.0), code)
                # capsule 2: tag + temp, ripple circle
                tag = self.scr["tag"]
                sub = tag.text().ljust(5) + " \u2014 TEMP: %d\u00b0C" % int(round(self.temp))
                p.setPen(self.C(0.95))
                p.drawText(QPointF(334.0, 177.0), sub)
                for age in self.ripples:
                    rad = 13.0 + age * 28.0
                    p.setPen(QPen(self.C((1 - age) * 0.6), 1.3))
                    p.setBrush(NB)
                    p.drawEllipse(QPointF(607.0, 170.5), rad, rad)
                self._knob(p, 607.0, 170.5, 13.0, -self.spin * 0.7, 40 + 160 * self.energy, beat)

        # ---- dot-matrix spectrum
        self._spectrum(p)

        # ---- bottom ruler + scanning marker
        r = rv(1.0, 0.7)
        if r > 0:
            xe = 10.0 + 618.0 * r
            p.setPen(QPen(self.C(0.9), 1.6))
            p.drawLine(QPointF(10, 290), QPointF(xe, 290))
            for i in range(0, 51):
                x = 10.0 + i * 12.36
                if x > xe:
                    break
                if i % 5 == 0:
                    p.setPen(QPen(self.C(0.9), 1.6))
                    p.drawLine(QPointF(x, 290), QPointF(x, 279))
                else:
                    p.setPen(QPen(self.C(0.30), 1.0))
                    p.drawLine(QPointF(x, 290), QPointF(x, 286))
            if r >= 0.999:
                mx = 10.0 + (self.t * sp * 150.0) % 618.0
                tr = QLinearGradient(mx - 60, 0, mx, 0)
                tr.setColorAt(0.0, self.C(0.0))
                tr.setColorAt(1.0, self.C(0.55 + 0.4 * beat, light=0.3))
                p.setPen(NP)
                p.setBrush(QBrush(tr))
                p.drawRect(QRectF(max(10.0, mx - 60), 281, min(60.0, mx - 10.0), 9))
                p.setPen(QPen(self.C(1.0, light=0.6), 2.0))
                p.drawLine(QPointF(mx, 276), QPointF(mx, 292))

    def _knob(self, p, cx, cy, r, spin, span, pulse):
        rr = r + pulse * 2.0
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.C(1.0, light=pulse * 0.35))
        p.drawEllipse(QPointF(cx, cy), rr, rr)
        pen = QPen(QColor(10, 10, 12, 235), 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        rect = QRectF(cx - r * 0.55, cy - r * 0.55, r * 1.1, r * 1.1)
        p.drawArc(rect, int(spin * 16), int(span * 16))

    def _spectrum(self, p):
        NP, NB = Qt.PenStyle.NoPen, Qt.BrushStyle.NoBrush
        sp = self.cfg["speed"]
        x0, pitch, cell = 10.0, 11.8, 10.0
        ry, rp = 183.0, 11.5
        tq = int(self.t * sp * 5.0)
        beat = self.beat
        for c in range(COLS):
            ramp = clamp((self.boot - 0.8 - 0.035 * c) / 0.5)
            if ramp <= 0:
                continue
            e = ease_out(ramp)
            h = float(self.disp[c]) * ROWS * e
            k = min(ROWS, int(h + 0.35))
            pk = int(float(self.peak[c]) * ROWS * e) if self.peak[c] > 0.03 else -1
            pk = min(pk, ROWS - 1)
            cx = x0 + cell / 2.0 + c * pitch
            if k > 0:      # soft glow behind the filled stack
                gt = ry - (k - 1) * rp - cell / 2.0 - 3.0
                gb = ry + cell / 2.0 + 3.0
                p.setPen(NP)
                p.setBrush(self.C(0.07 * ramp + 0.05 * beat))
                p.drawRoundedRect(QRectF(cx - cell / 2.0 - 3.0, gt, cell + 6.0, gb - gt), 4, 4)
            for r in range(ROWS):
                cy = ry - r * rp
                if r < k:
                    top = (r == k - 1)
                    if (not top) and r > 0 and ((c * 7 + r * 13 + tq) % 17 == 0):
                        p.setPen(QPen(self.C(0.8 * ramp), 1.3))
                        p.setBrush(NB)
                        p.drawRoundedRect(QRectF(cx - 4.5, cy - 4.5, 9, 9), 2.2, 2.2)
                    else:
                        a = (0.80 + 0.20 * r / ROWS) * ramp
                        p.setPen(NP)
                        p.setBrush(self.C(a, light=(0.35 if top else beat * 0.15)))
                        p.drawRoundedRect(QRectF(cx - 5, cy - 5, 10, 10), 2.4, 2.4)
                elif r == pk:
                    p.setPen(QPen(self.C(1.0 * ramp, light=0.45), 1.5))
                    p.setBrush(NB)
                    p.drawRoundedRect(QRectF(cx - 4.5, cy - 4.5, 9, 9), 2.2, 2.2)
                elif r < k + 2:
                    a = (0.85 if r == k else 0.40) * ramp
                    p.setPen(QPen(self.C(a), 1.3))
                    p.setBrush(NB)
                    p.drawRoundedRect(QRectF(cx - 4.5, cy - 4.5, 9, 9), 2.2, 2.2)
                else:
                    rad = lerp(2.3, 1.0, (r - k) / max(1.0, float(ROWS - k)))
                    a = (0.58 - 0.30 * r / ROWS) * clamp(ramp * 2.0)
                    p.setPen(NP)
                    p.setBrush(self.C(a))
                    p.drawEllipse(QPointF(cx, cy), rad, rad)

    # ------------------------------------------------------------------- menu
    def menu_rect(self):
        e = ease_io(self.hover_t)
        W = self.width()
        cw = lerp(PW * 0.62, PW, e)
        ch = lerp(30.0, PH, e)
        return QRectF(W - MENU_M - cw, MENU_M, cw, ch)

    def menu_interactive(self):
        return self.hover_t > 0.8

    def menu_layout(self, W):
        px0 = W - MENU_M - PW
        py0 = float(MENU_M)
        L = {"px0": px0,
             "close": QRectF(px0 + PW - 28, py0 + 5, 20, 20),
             "pin": QRectF(px0 + PW - 62, py0 + 5, 30, 20)}
        y = py0 + 38
        L["color_y"] = y
        L["swatches"] = [QRectF(px0 + 14 + i * 26, y + 16, 18, 18) for i in range(len(PRESETS))]
        y += 44
        L["sliders"] = {}
        for key, _lab, _lo, _hi in SLIDERS:
            L["sliders"][key] = {"y": y,
                                 "track": QRectF(px0 + 14, y + 19, PW - 28, 4),
                                 "hit": QRectF(px0 + 8, y + 8, PW - 16, 26)}
            y += 34
        return L

    def draw_menu(self, p, W, H):
        reveal = ease_out(clamp((self.boot - 1.2) / 0.6))
        if reveal <= 0:
            return
        e = ease_io(self.hover_t)
        rect = self.menu_rect()
        base_op = lerp(0.32, 1.0, e) * reveal
        NB = Qt.BrushStyle.NoBrush
        p.save()
        p.setOpacity(base_op)
        path = QPainterPath()
        path.addRoundedRect(rect, 6, 6)
        p.fillPath(path, QColor(8, 8, 10, int(lerp(150, 235, e))))
        p.setPen(QPen(self.C(lerp(0.45, 0.85, e)), 1.0))
        p.setBrush(NB)
        p.drawPath(path)
        p.setClipPath(path)
        p.setFont(self.f_small_b)
        p.setPen(self.C(1.0, light=0.2))
        p.drawText(QPointF(rect.left() + 12, rect.top() + 19), "[ CONFIG ]")
        if e < 0.6:  # blinking status pip when collapsed
            blink = 0.5 + 0.5 * math.sin(self.t * 4.0)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(self.C(0.4 + 0.6 * blink))
            p.drawEllipse(QPointF(rect.right() - 12, rect.top() + 15), 2.6, 2.6)

        if e > 0.03:
            p.setOpacity(base_op * (e ** 1.6))
            L = self.menu_layout(W)
            px0 = L["px0"]
            # header buttons
            p.setFont(self.f_small_b)
            pin_on = self.cfg["topmost"]
            p.setPen(self.C(1.0 if pin_on else 0.4, light=0.2 if pin_on else 0.0))
            tw = QFontMetricsF(self.f_small_b).horizontalAdvance("TOP")
            p.drawText(QPointF(L["pin"].center().x() - tw / 2.0, L["pin"].center().y() + 4.0), "TOP")
            if pin_on:
                p.setPen(QPen(self.C(0.9), 1.2))
                p.drawLine(QPointF(L["pin"].left() + 3, L["pin"].bottom() - 2),
                           QPointF(L["pin"].right() - 3, L["pin"].bottom() - 2))
            cr = L["close"]
            p.setPen(QPen(self.C(0.9), 1.5))
            p.drawLine(QPointF(cr.left() + 5, cr.top() + 5), QPointF(cr.right() - 5, cr.bottom() - 5))
            p.drawLine(QPointF(cr.right() - 5, cr.top() + 5), QPointF(cr.left() + 5, cr.bottom() - 5))
            p.setPen(QPen(self.C(0.25), 1.0))
            p.drawLine(QPointF(px0 + 10, MENU_M + 31), QPointF(px0 + PW - 10, MENU_M + 31))

            # colour swatches
            p.setFont(self.f_small)
            p.setPen(self.C(0.8))
            p.drawText(QPointF(px0 + 14, L["color_y"] + 10), "COLOR")
            for i, r in enumerate(L["swatches"]):
                name, h, s, v = PRESETS[i]
                c = QColor.fromHsvF(min(h / 360.0, 0.999), s, v)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(c)
                p.drawEllipse(r)
                if abs(((self.cfg["hue"] - h + 180) % 360) - 180) < 2 and abs(self.cfg["sat"] - s) < 0.02:
                    p.setPen(QPen(self.C(1.0, light=0.3), 1.4))
                    p.setBrush(NB)
                    p.drawEllipse(r.adjusted(-3, -3, 3, 3))

            # sliders
            for key, label, lo, hi in SLIDERS:
                v = self.cfg[key]
                ratio = clamp((v - lo) / (hi - lo))
                S = L["sliders"][key]
                tr = S["track"]
                y = S["y"]
                p.setFont(self.f_small)
                p.setPen(self.C(0.8))
                p.drawText(QPointF(px0 + 14, y + 12), label)
                vt = self._fmt(key, v)
                p.setPen(self.C(1.0, light=0.2))
                p.drawText(QPointF(px0 + PW - 14 - QFontMetricsF(self.f_small).horizontalAdvance(vt), y + 12), vt)
                p.setPen(Qt.PenStyle.NoPen)
                if key == "hue":
                    gr = QLinearGradient(tr.left(), 0, tr.right(), 0)
                    for q in range(7):
                        gr.setColorAt(q / 6.0, QColor.fromHsvF(min(q / 6.0, 0.999), 0.55, 0.95, 0.85))
                    p.setBrush(QBrush(gr))
                    p.drawRoundedRect(tr, 2, 2)
                else:
                    p.setBrush(self.C(0.22))
                    p.drawRoundedRect(tr, 2, 2)
                    p.setBrush(self.C(0.95))
                    p.drawRoundedRect(QRectF(tr.left(), tr.top(), tr.width() * ratio, tr.height()), 2, 2)
                kx = tr.left() + tr.width() * ratio
                p.setBrush(self.C(1.0, light=0.35))
                p.setPen(QPen(QColor(8, 8, 10, 220), 1.0))
                p.drawRoundedRect(QRectF(kx - 3.5, tr.top() - 5, 7, 14), 2, 2)
        p.restore()

    @staticmethod
    def _fmt(key, v):
        if key == "hue":
            return "%d\u00b0" % int(v)
        if key == "speed":
            return "%.2fx" % v
        if key == "gain":
            return "%+.0f dB" % ((v - 0.5) * 20.0)
        return "%d%%" % int(round(v * 100))

    # ------------------------------------------------------------ interaction
    def edges_at(self, pos):
        m = 8
        flags = []
        if pos.x() < m:
            flags.append(Qt.Edge.LeftEdge)
        if pos.x() > self.width() - m:
            flags.append(Qt.Edge.RightEdge)
        if pos.y() < m:
            flags.append(Qt.Edge.TopEdge)
        if pos.y() > self.height() - m:
            flags.append(Qt.Edge.BottomEdge)
        if not flags:
            return None
        return functools.reduce(operator.or_, flags)

    def _cursor_for(self, edges):
        if edges is None:
            return Qt.CursorShape.ArrowCursor
        L, R = Qt.Edge.LeftEdge, Qt.Edge.RightEdge
        T, B = Qt.Edge.TopEdge, Qt.Edge.BottomEdge
        has = lambda f: bool(edges & f)
        if (has(L) and has(T)) or (has(R) and has(B)):
            return Qt.CursorShape.SizeFDiagCursor
        if (has(R) and has(T)) or (has(L) and has(B)):
            return Qt.CursorShape.SizeBDiagCursor
        if has(L) or has(R):
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.SizeVerCursor

    def update_slider(self, x):
        key = self.drag
        if key is None:
            return
        spec = next(s for s in SLIDERS if s[0] == key)
        tr = self.menu_layout(self.width())["sliders"][key]["track"]
        ratio = clamp((x - tr.left()) / tr.width())
        self.cfg[key] = spec[2] + (spec[3] - spec[2]) * ratio
        if key == "hue":
            self.cfg["sat"] = max(self.cfg["sat"], 0.40)
            self.cfg["val"] = max(self.cfg["val"], 0.80)

    def menu_press(self, pos):
        L = self.menu_layout(self.width())
        if L["close"].contains(pos):
            self.close()
            return
        if L["pin"].contains(pos):
            self.set_topmost(not self.cfg["topmost"])
            return
        for i, r in enumerate(L["swatches"]):
            if r.adjusted(-3, -3, 3, 3).contains(pos):
                _n, h, s, v = PRESETS[i]
                self.cfg["hue"], self.cfg["sat"], self.cfg["val"] = h, s, v
                save_settings(self.cfg)
                return
        for key, S in L["sliders"].items():
            if S["hit"].contains(pos):
                self.drag = key
                self.update_slider(pos.x())
                return

    def mousePressEvent(self, ev):
        pos = ev.position()
        if ev.button() == Qt.MouseButton.LeftButton:
            if self.menu_interactive() and self.menu_rect().contains(pos):
                self.menu_press(pos)
                return
            wh = self.windowHandle()
            if wh is None:
                return
            edges = self.edges_at(pos)
            if edges is not None:
                wh.startSystemResize(edges)
            else:
                wh.startSystemMove()
        elif ev.button() == Qt.MouseButton.RightButton:
            self.show_context(ev.globalPosition().toPoint())

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self.drag is not None:
            self.update_slider(pos.x())
            return
        if self.menu_interactive() and self.menu_rect().contains(pos):
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(self._cursor_for(self.edges_at(pos)))

    def mouseReleaseEvent(self, ev):
        if self.drag is not None:
            self.drag = None
            save_settings(self.cfg)

    def show_context(self, gpos):
        m = QMenu(self)
        m.setStyleSheet("QMenu{background:#0b0b0d;color:#d8b9b3;border:1px solid #4a3a38;padding:4px;}"
                        "QMenu::item{padding:5px 22px;}QMenu::item:selected{background:#2a1f1e;}"
                        "QMenu::separator{height:1px;background:#3a2c2a;margin:4px 6px;}")
        a_top = m.addAction("Always on top")
        a_top.setCheckable(True)
        a_top.setChecked(self.cfg["topmost"])
        a_rec = m.addAction("Reconnect audio")
        a_rst = m.addAction("Reset settings")
        m.addSeparator()
        a_quit = m.addAction("Quit")
        act = m.exec(gpos)
        if act is a_top:
            self.set_topmost(a_top.isChecked())
        elif act is a_rec and self.audio is not None:
            self.audio.reconnect()
        elif act is a_rst:
            for k in ("hue", "sat", "val", "speed", "gain", "opacity"):
                self.cfg[k] = DEFAULTS[k]
            save_settings(self.cfg)
        elif act is a_quit:
            self.close()

    def set_topmost(self, on):
        self.cfg["topmost"] = bool(on)
        geo = self.geometry()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, bool(on))
        self.show()
        self.setGeometry(geo)
        save_settings(self.cfg)

    def closeEvent(self, ev):
        g = self.geometry()
        self.cfg.update(win_x=g.x(), win_y=g.y(), win_w=g.width(), win_h=g.height())
        save_settings(self.cfg)
        if self.audio is not None:
            self.audio.stop()
        ev.accept()
        QApplication.quit()


def main():
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Fusion.AudioVisualizer")
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setApplicationName("FUSION Visualizer")
    w = FusionWidget(demo=("--demo" in sys.argv))
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
