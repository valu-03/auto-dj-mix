"""The design system: one source of truth for colour, type, space and motion.

Everything visual is defined here as tokens and the stylesheet is generated
from them. Nothing anywhere else in the GUI hard-codes a hex value -- if a
colour appears twice in two files it will drift, and a design that drifts stops
reading as one product.

Two things in here are less obvious than they look.

**Tokens are split by role, not by appearance.** There is no single "border"
colour. A decorative card hairline, the edge that tells you a control *is* a
control, and a focus ring are three different jobs with three different
accessibility obligations, and collapsing them into one token forces a choice
between an inaccessible focus ring and a design shouting at you through heavy
boxes everywhere. WCAG only requires 3:1 of the second and third. So the
hairline stays whisper-quiet and the ones that carry meaning are loud enough to
see, which is both more accessible and better looking than either compromise.

**Both palettes are solved, not picked.** Every colour here was moved along
lightness only -- hue and saturation held -- until it measured its target
ratio, so the ramp still reads as one family. `python -m autodj.gui.theme`
re-runs the audit and prints the measured ratio of every pair the app can
actually produce, in both modes. If a token is edited to something that fails,
that command says so.
"""

# ---------------------------------------------------------------- colour ----
# Neutrals are blue-tinted rather than pure grey: a true neutral next to a
# saturated accent reads as dirty. The accent is the only saturated hue in the
# chrome, so it never has to compete to mean "this is the action".
#
# Contrast targets, from WCAG 2.1:
#   text                        4.5:1  (1.4.3)
#   text >=18.66px semibold     3.0:1  (1.4.3 large text)
#   control edges, focus rings,
#   and meaningful graphics     3.0:1  (1.4.11, 2.4.11)
#   decorative hairlines        none

# The brand ramp is four blues:
#
#   #0D47A1  deep     #2196F3  mid     #90CAF9  light     #E3F2FD  palest
#
# They divide by role almost exactly, once measured. #0D47A1 carries white
# text at 8.63 and is the primary fill on light. #2196F3 cannot carry white
# text at all -- 3.12, a clear fail -- but takes near-black at 6.72, so on
# dark it becomes the primary fill with a dark label, which is also the
# brightest thing on the screen and therefore correctly the primary action.
# #90CAF9 is the focus ring and accent text on dark. #E3F2FD is body text on
# dark and the app ground on light.
#
# The neutrals are not from the ramp -- four blues cannot furnish surfaces,
# hairlines and three tiers of text. They are mixed toward #0D47A1's hue so
# the greys read as belonging to the blues rather than sitting beside them.
_DARK = dict(
    BG="#0A1929",          # app ground, the deep blue taken almost to black
    SURFACE="#0F2137",     # cards
    SURFACE_2="#152C46",   # raised: inputs, rows, hover
    SURFACE_3="#1B3655",   # popovers, menus, tooltips

    HAIRLINE="#1B3350",    # decorative only -- separates, never informs
    HAIRLINE_HI="#26456A",
    CONTROL_EDGE="#5B84B8",   # the edge that makes a control identifiable
    CONTROL_EDGE_HI="#7BA4D6",
    FOCUS="#90CAF9",          # keyboard focus ring -- the palette's light blue

    TEXT="#E3F2FD",           # the palette's palest, straight from the ramp
    TEXT_DIM="#A8C4DD",
    TEXT_MUTED="#7E9CBA",

    # Primary fill is the mid blue, and its label is dark. White on #2196F3
    # measures 3.12 and fails; near-black measures 6.72 and passes.
    ACCENT="#2196F3",
    ACCENT_HI="#48A9F5",
    # Pressed darkens one step -- Material Blue 600 against Blue 500 -- and
    # stops there. Going all the way to the brand deep blue looks right and
    # measures 1.93 against the dark label, so the pressed state would be the
    # one state of the primary button nobody can read.
    ACCENT_DIM="#1E88E5",
    ACCENT_TEXT="#90CAF9",    # accent used AS text, on a dark ground
    ACCENT_WASH="rgba(33, 150, 243, 0.16)",
    GRAPHIC="#2196F3",        # progress fills, meters, slider value
    ON_ACCENT="#04141F",      # text and icons sitting on an accent fill

    SUCCESS="#3DD9A0",
    WARNING="#F5B841",
    DANGER="#FF6B81",

    # Data-visualisation ramp. This one deliberately does NOT collapse into
    # the brand blues. The chrome should be one family; a data ramp encodes
    # things that are different from each other, and six blues would encode
    # them as six shades of the same thing. The decks are the sharpest case:
    # SERIES[1] is always the incoming deck and SERIES[3] always the outgoing
    # one, drawn on top of each other at every join, and the whole reason that
    # reads at a glance is that they are not the same hue.
    #
    # SERIES[1] is the brand's mid blue, so the incoming deck is brand-coloured
    # and the ramp still starts from the palette.
    SERIES=("#7C9EF8", "#2196F3", "#3DD9A0", "#F5B841", "#FF6B81", "#B69CFF"),

    # Structure segments drawn under a waveform. These lived in waveform.py as
    # raw hex until this rewrite, which is exactly the drift the module
    # docstring claims does not happen.
    #
    # Painted opaque, and lightened from the originals until each clears 3:1
    # on every surface. The old set was painted at alpha 200 over the card,
    # which made the band a *lighter* version of an already-failing colour --
    # the intro and outro blocks measured 1.71 and were, honestly, invisible.
    SEGMENTS={
        "intro": "#5B7C9E", "build": "#4F9BE8", "drop": "#90CAF9",
        "breakdown": "#B69CFF", "outro": "#5B7C9E", "verse": "#3E7FC1",
        "chorus": "#4F9BE8",
    },
)

# Light mode is not the dark palette inverted. Inversion fails twice: the
# saturated ramp loses all its contrast against white (cyan #22D3EE measures
# 1.58 on white and is simply invisible), and dark-mode text weights look
# heavy. So the ramp is darkened per-colour and the neutrals are re-solved.
_LIGHT = dict(
    # #E3F2FD is the ground here rather than a grey: it is the palette's
    # palest blue, so the light mode is brand-tinted at the root and cards sit
    # on it as plain white.
    BG="#E3F2FD",
    SURFACE="#FFFFFF",
    SURFACE_2="#F2F8FE",
    SURFACE_3="#FFFFFF",

    HAIRLINE="#C3DFF7",
    HAIRLINE_HI="#90CAF9",     # the palette's light blue, as a hover hairline
    CONTROL_EDGE="#5A87B5",
    CONTROL_EDGE_HI="#3A6A9C",
    FOCUS="#0D47A1",           # the deep blue: 7.56 on the ground, 8.63 on white

    TEXT="#0A1929",
    TEXT_DIM="#2E4A66",
    TEXT_MUTED="#4C6A87",

    # The deep blue is the fill on light, and it carries white at 8.63.
    ACCENT="#0D47A1",
    ACCENT_HI="#0A3A85",       # hover goes *darker* on light, not lighter
    ACCENT_DIM="#082D68",
    ACCENT_TEXT="#0D47A1",
    ACCENT_WASH="rgba(13, 71, 161, 0.10)",
    GRAPHIC="#0D47A1",
    ON_ACCENT="#FFFFFF",

    # Status colours have to clear 4.5 here, not 3.0: on light they are used
    # as text far more than as fills, and the saturated dark-mode versions all
    # measure around 3.2 on white.
    SUCCESS="#0A7250",
    WARNING="#8C5A05",
    DANGER="#C8183F",

    # Darkened per-colour rather than inverted. #2196F3 measures 3.12 on white
    # and 2.74 on the #E3F2FD ground -- it cannot be text here, so the light
    # ramp's blue is the deep one.
    # Two of these are darker than the 3.0 chart requirement needs. The
    # timeline paints a track label directly on the block, and at the
    # lighter values no foreground cleared 4.5 -- black reached 4.36 on
    # the emerald and white was worse. Darkening lets the white label
    # pass and costs the chart nothing.
    SERIES=("#3F5FD0", "#1565C0", "#0A7250", "#8C5A05", "#C8183F", "#6D4FD6"),

    SEGMENTS={
        "intro": "#63819E", "build": "#1565C0", "drop": "#0D47A1",
        "breakdown": "#6D4FD6", "outro": "#63819E", "verse": "#4C7BA8",
        "chorus": "#1565C0",
    },
)

PALETTES = {"dark": _DARK, "light": _LIGHT}
MODE = "dark"

# The tokens are also bound as module attributes so call sites stay `T.BG`
# rather than `T.palette()["BG"]`. Widgets read them inside paintEvent, so
# rebinding these and re-polishing is enough to switch the whole app live.
globals().update(_DARK)


def set_mode(mode):
    """Switch palette. Returns the stylesheet for the new mode."""
    global MODE
    if mode not in PALETTES:
        raise ValueError(f"unknown mode: {mode!r}")
    MODE = mode
    globals().update(PALETTES[mode])
    return stylesheet()


def palette(mode=None):
    return PALETTES[mode or MODE]


# --------------------------------------------------------------- spacing ----
# A strict 4px base with an 8px rhythm. Named rather than numbered so intent
# survives edits: changing "card padding" is a decision, changing "24" is a
# typo waiting to happen.
SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "2xl": 32, "3xl": 48}

UNIT = 8                # the rhythm everything else is a multiple of
GAP = SPACE["lg"]       # 16 -- between cells
PAD_CARD = SPACE["xl"]  # 24 -- inside a card
PAD_VIEW = SPACE["2xl"] # 32 -- around the main view

RADIUS = 14             # cards
RADIUS_SM = 9           # controls
RADIUS_XS = 6           # chips, tags, indicators

SIDEBAR_W = 248
SIDEBAR_MIN = 76
TOPBAR_H = 64
QUEUE_W = 380

# Breakpoints, in window width. The app is a desktop tool and does not reflow
# to phone widths, but it does have three real states: below COMPACT the
# sidebar collapses to icons, and below NARROW the Automix queue hides behind
# a toggle rather than shrinking to uselessness.
BP_NARROW = 1100
BP_COMPACT = 1400

# Minimum hit target. 44px is the WCAG 2.5.5 enhanced figure; 24px is the AA
# floor from 2.5.8. Dense desktop rows use the AA floor, primary actions the
# enhanced one.
HIT_MIN = 24
HIT_COMFORTABLE = 44

# ------------------------------------------------------------------ type ----
FONT_STACK = '"Roboto", "Inter", "Segoe UI Variable Text", "Segoe UI", sans-serif'
MONO_STACK = '"Roboto Mono", "JetBrains Mono", "Cascadia Mono", Consolas, monospace'

# Fallback chain, best first. Qt does not read the CSS stack in FONT_STACK
# when a QFont is built in code, so setFamilies has to be given the list
# explicitly -- otherwise a missing Inter silently becomes the system default
# and every measurement in the layout shifts.
FAMILIES = ["Roboto", "Inter", "Segoe UI Variable Text", "Segoe UI",
            "Helvetica Neue", "Arial"]
MONO_FAMILIES = ["Roboto Mono", "JetBrains Mono", "Cascadia Mono", "Consolas",
                 "Courier New"]

# (px, weight, letter-spacing px, line-height px)
#
# Sizes step by roughly 1.25x. Weight and colour carry as much of the
# hierarchy as size does, which is what keeps a dense screen from growing.
#
# Weights are checked against what the family actually ships. Roboto is
# installed here with Regular, Medium, SemiBold and a real Bold, so 600 is a
# genuine cut rather than a synthesised one -- which was not true of the
# previous stack: Inter shipped only Regular/Medium/SemiBold, and any request
# for 700 was quietly clamped or faked by smearing the outline.
# `python -m autodj.gui.theme` re-checks this, so a machine without Roboto
# gets told rather than silently re-rendering in a fallback.
#
# Letter-spacing runs negative at display sizes and positive at caption sizes.
# Tracking that looks correct at 28px is too tight at 11px -- the optical size
# compensation a typeface would do for you if this were a real optical family.
#
# Line heights are the tight end of comfortable (1.2 at display, 1.45 at body)
# because most text here is one line inside a control. Qt has no CSS
# line-height, so `line_height` is applied through layout spacing and, for
# wrapping labels, through QTextBlockFormat -- see widgets.set_line_height.
TYPE = {
    "display": (28, 600, -0.4, 34),
    "title":   (19, 600, -0.2, 25),
    "heading": (15, 600, -0.1, 21),
    "body":    (13, 400,  0.0, 19),
    "label":   (12, 500,  0.1, 16),
    "caption": (11, 500,  0.2, 15),
    "metric":  (30, 600, -0.6, 34),
    "mono":    (12, 400,  0.0, 17),
}


def qfont(name, family=None):
    """A QFont for a type-scale entry, with the fallback chain applied."""
    from PyQt6.QtGui import QFont
    size, weight, spacing, _ = TYPE[name]
    f = QFont()
    fams = MONO_FAMILIES if name == "mono" else FAMILIES
    f.setFamilies([family] + fams if family else fams)
    f.setPixelSize(size)
    f.setWeight(QFont.Weight(weight))
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    f.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    return f


def app_font():
    """The application-wide font: family and hinting only, no size.

    Deliberately leaves the size alone. `qfont()` uses `setPixelSize`, which
    makes `pointSize()` return -1, and Qt emits "QFont::setPointSize: Point
    size <= 0 (-1)" whenever an internal path round-trips the application font
    through `setPointSize`. The stylesheet sets `font-size` in px on QWidget,
    so every widget still gets the scale -- this only stops the app default
    from carrying an invalid point size around.
    """
    from PyQt6.QtGui import QFont
    f = QFont()
    f.setFamilies(FAMILIES)
    f.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    return f


def font_css(name):
    size, weight, spacing, _ = TYPE[name]
    return (f"font-size:{size}px; font-weight:{weight}; "
            f"letter-spacing:{spacing}px;")


def line_height(name):
    return TYPE[name][3]


# ---------------------------------------------------------------- motion ----
# Durations. Qt's stylesheet engine has no `transition`, so these are consumed
# by QPropertyAnimation in gui/motion.py rather than by the stylesheet.
#
# The scale is short on purpose. A state change the user caused should feel
# instant -- past about 200ms a hover reads as lag rather than as polish -- and
# only movement that has to be *followed* by the eye (a panel sliding, a toast
# arriving) gets long enough to track.
DUR_INSTANT = 90     # colour and opacity on hover/press
DUR_FAST = 150       # small transforms, checkbox, chip
DUR_BASE = 200       # panels, page changes
DUR_SLOW = 280       # sidebar collapse, toast arrival
ANIM_MS = DUR_BASE   # kept: existing call sites use this name

# Cubic-bezier control points, matched to the CSS names they came from so the
# motion vocabulary is portable if any of this is ever documented as a spec.
EASE_STANDARD = (0.20, 0.00, 0.00, 1.00)   # most things: decisive, settles soft
EASE_OUT = (0.16, 1.00, 0.30, 1.00)        # things arriving
EASE_IN = (0.70, 0.00, 0.84, 0.00)         # things leaving
EASE_SPRING = (0.34, 1.56, 0.64, 1.00)     # overshoots: only for confirmations


def stylesheet():
    """The whole application stylesheet, generated from the tokens above.

    State is expressed through :hover, :pressed, :focus and :disabled, which
    Qt does support and which is what makes a control feel alive under the
    cursor. Actual movement lives in motion.py.
    """
    p = palette()
    return f"""
    QWidget {{
        background: {p['BG']};
        color: {p['TEXT']};
        font-family: {FONT_STACK};
        {font_css("body")}
    }}

    /* Labels must not paint a background. The rule above is what gives the
       app its ground colour, and Qt cascades it to every widget -- including
       every QLabel, which then fills its whole allocated width with the app
       ground. Inside a card, whose surface is lighter than the ground, that
       reads as a long dark bar behind the text: one under every field
       caption, every metric row and every track title. Transparent labels
       simply show whatever they are sitting on. */
    QLabel, QCheckBox {{ background: transparent; }}

    /* Same fault one level up the tree: a scroll area, its viewport and the
       widget it holds each inherit the app ground, which paints a block over
       the card behind them. */
    QScrollArea#Reasons, QScrollArea#Reasons QWidget {{
        background: transparent;
    }}

    /* ---- structure ---- */
    #Sidebar {{ background: {p['SURFACE']};
                border-right: 1px solid {p['HAIRLINE']}; }}
    #TopBar  {{ background: {p['BG']};
                border-bottom: 1px solid {p['HAIRLINE']}; }}
    #Canvas  {{ background: {p['BG']}; }}

    QFrame#Card {{
        background: {p['SURFACE']};
        border: 1px solid {p['HAIRLINE']};
        border-radius: {RADIUS}px;
    }}
    QFrame#Card:hover {{ border-color: {p['HAIRLINE_HI']}; }}

    QLabel#CardTitle   {{ color: {p['TEXT']};       {font_css("heading")} }}
    QLabel#CardHint    {{ color: {p['TEXT_MUTED']}; {font_css("caption")} }}
    QLabel#Metric      {{ color: {p['TEXT']};       {font_css("metric")} }}
    QLabel#MetricLabel {{ color: {p['TEXT_DIM']};   {font_css("label")}
                          text-transform: uppercase; }}
    QLabel#Display     {{ color: {p['TEXT']};       {font_css("display")} }}
    QLabel#Dim         {{ color: {p['TEXT_DIM']}; }}
    QLabel#Muted       {{ color: {p['TEXT_MUTED']}; }}
    QLabel#Accent      {{ color: {p['ACCENT_TEXT']}; }}

    /* ---- navigation ---- */
    QPushButton#NavItem {{
        background: transparent;
        border: none;
        border-radius: {RADIUS_SM}px;
        padding: 11px 14px;
        text-align: left;
        color: {p['TEXT_DIM']};
        {font_css("body")}
    }}
    QPushButton#NavItem:hover   {{ background: {p['SURFACE_2']};
                                   color: {p['TEXT']}; }}
    QPushButton#NavItem:checked {{ background: {p['ACCENT_WASH']};
                                   color: {p['TEXT']}; }}
    QPushButton#NavItem:focus   {{ outline: none;
                                   border: 2px solid {p['FOCUS']}; }}

    /* ---- buttons ---- */
    QPushButton#Primary {{
        background: {p['ACCENT']}; color: {p['ON_ACCENT']}; border: none;
        border-radius: {RADIUS_SM}px; padding: 11px 20px;
        {font_css("label")}
    }}
    QPushButton#Primary:hover    {{ background: {p['ACCENT_HI']}; }}
    QPushButton#Primary:pressed  {{ background: {p['ACCENT_DIM']}; }}
    QPushButton#Primary:disabled {{ background: {p['SURFACE_2']};
                                    color: {p['TEXT_MUTED']}; }}
    /* The focus ring on a filled button is drawn ON the fill, so it cannot be
       the accent colour -- in light mode FOCUS and ACCENT are the same indigo
       and the ring measured 1.00:1 against the thing it was outlining, which
       is to say it was invisible to exactly the keyboard users it exists for.
       ON_ACCENT measures 6.29 against the fill in both modes. */
    QPushButton#Primary:focus    {{ border: 2px solid {p['ON_ACCENT']};
                                    padding: 9px 18px; }}

    QPushButton#Ghost {{
        background: transparent; color: {p['TEXT_DIM']};
        border: 1px solid {p['CONTROL_EDGE']}; border-radius: {RADIUS_SM}px;
        padding: 10px 16px; {font_css("label")}
    }}
    QPushButton#Ghost:hover    {{ border-color: {p['CONTROL_EDGE_HI']};
                                  color: {p['TEXT']};
                                  background: {p['SURFACE_2']}; }}
    QPushButton#Ghost:pressed  {{ background: {p['HAIRLINE']}; }}
    QPushButton#Ghost:focus    {{ border: 2px solid {p['FOCUS']}; }}
    QPushButton#Ghost:disabled {{ color: {p['TEXT_MUTED']};
                                  border-color: {p['HAIRLINE']}; }}

    QPushButton#Danger {{
        background: transparent; color: {p['DANGER']};
        border: 1px solid {p['DANGER']}; border-radius: {RADIUS_SM}px;
        padding: 10px 16px; {font_css("label")}
    }}
    QPushButton#Danger:hover  {{ background: {p['DANGER']};
                                 color: {p['ON_ACCENT']}; }}
    QPushButton#Danger:focus  {{ border: 2px solid {p['FOCUS']}; }}

    QPushButton#IconBtn {{
        background: transparent; border: 1px solid transparent;
        border-radius: {RADIUS_SM}px; padding: 8px; color: {p['TEXT_DIM']};
        min-width: {HIT_MIN}px; min-height: {HIT_MIN}px;
    }}
    QPushButton#IconBtn:hover {{ background: {p['SURFACE_2']};
                                 color: {p['TEXT']};
                                 border-color: {p['CONTROL_EDGE']}; }}
    QPushButton#IconBtn:focus {{ border: 2px solid {p['FOCUS']}; }}

    /* ---- inputs ---- */
    /* The edge here is CONTROL_EDGE, not the decorative hairline. On a dark
       card no fill can identify a control on its own: the card is already so
       dark that even pure black against it measures 1.21:1, so the boundary
       has to carry it. */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {p['SURFACE_2']};
        border: 1px solid {p['CONTROL_EDGE']};
        border-radius: {RADIUS_SM}px;
        padding: 10px 13px;
        color: {p['TEXT']};
        selection-background-color: {p['ACCENT']};
        selection-color: {p['ON_ACCENT']};
        min-height: {HIT_MIN}px;
    }}
    QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {p['CONTROL_EDGE_HI']};
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 2px solid {p['FOCUS']};
        background: {p['SURFACE']};
        padding: 9px 12px;   /* keep the box the same size as the 1px state */
    }}
    QLineEdit:disabled, QComboBox:disabled {{ color: {p['TEXT_MUTED']};
                                              border-color: {p['HAIRLINE']}; }}
    QComboBox::drop-down {{ border: none; width: 26px; }}
    QComboBox QAbstractItemView {{
        background: {p['SURFACE_3']};
        border: 1px solid {p['HAIRLINE_HI']};
        border-radius: {RADIUS_SM}px;
        selection-background-color: {p['ACCENT_WASH']};
        selection-color: {p['TEXT']};
        padding: 4px;
        outline: none;
    }}

    QCheckBox {{ color: {p['TEXT_DIM']}; spacing: 9px; }}
    QCheckBox:hover {{ color: {p['TEXT']}; }}
    QCheckBox::indicator {{
        width: 17px; height: 17px; border-radius: 5px;
        border: 1px solid {p['CONTROL_EDGE']}; background: {p['SURFACE_2']};
    }}
    QCheckBox::indicator:checked {{ background: {p['ACCENT']};
                                    border-color: {p['ACCENT']}; }}
    QCheckBox::indicator:hover   {{ border-color: {p['FOCUS']}; }}
    QCheckBox:focus {{ color: {p['TEXT']}; }}

    /* ---- table ---- */
    QTableView {{
        background: transparent;
        border: none;
        gridline-color: transparent;
        selection-background-color: {p['ACCENT_WASH']};
        selection-color: {p['TEXT']};
        outline: none;
    }}
    QTableView::item {{
        padding: 9px 10px;
        border: none;
        border-bottom: 1px solid {p['HAIRLINE']};
    }}
    QTableView::item:hover {{ background: {p['SURFACE_2']}; }}
    /* Border repeated on the selected state. Without it Qt falls back to its
       own cell borders for the selected row, which draws a vertical rule at
       every column edge -- a row of pipes through the text that appears only
       when a row is selected. */
    QTableView::item:selected {{
        background: {p['ACCENT_WASH']};
        color: {p['TEXT']};
        border: none;
        border-bottom: 1px solid {p['HAIRLINE']};
    }}
    QTableView::item:focus {{ border: none;
                              border-bottom: 1px solid {p['HAIRLINE']}; }}
    QHeaderView::section {{
        background: transparent;
        color: {p['TEXT_MUTED']};
        border: none;
        border-bottom: 1px solid {p['HAIRLINE']};
        padding: 10px;
        {font_css("caption")}
        text-transform: uppercase;
    }}

    /* ---- progress ---- */
    QProgressBar {{
        background: {p['SURFACE_2']}; border: 1px solid {p['HAIRLINE']};
        border-radius: 4px; height: 6px;
        text-align: center; color: transparent;
    }}
    QProgressBar::chunk {{ background: {p['GRAPHIC']}; border-radius: 3px; }}

    /* ---- scrollbars: present, but never the loudest thing on screen ---- */
    QScrollBar:vertical   {{ background: transparent; width: 11px;
                             margin: 2px; }}
    QScrollBar:horizontal {{ background: transparent; height: 11px;
                             margin: 2px; }}
    QScrollBar::handle {{ background: {p['CONTROL_EDGE']}; border-radius: 5px; }}
    QScrollBar::handle:hover {{ background: {p['CONTROL_EDGE_HI']}; }}
    QScrollBar::handle:vertical   {{ min-height: 32px; }}
    QScrollBar::handle:horizontal {{ min-width: 32px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    /* ---- the Automix column ---- */
    QFrame#QueuePanel {{
        background: {p['SURFACE']};
        border-left: 1px solid {p['HAIRLINE']};
    }}
    QListWidget#Queue {{ background: transparent; border: none; }}
    QListWidget#Queue::item {{ border: none; padding: 0; }}

    /* Compact inputs, for controls that sit inside a dense row. The default
       10px vertical padding is right for a form and wrong here: combined with
       a short fixed height it does not shrink the control, it clips the text
       inside it, which looked like a missing font rather than a layout bug. */
    QComboBox#Compact, QSpinBox#Compact, QDoubleSpinBox#Compact {{
        padding: 3px 8px;
        {font_css("caption")}
        border-radius: {RADIUS_XS}px;
        min-height: 0px;
    }}
    QComboBox#Compact:focus, QSpinBox#Compact:focus,
    QDoubleSpinBox#Compact:focus {{ padding: 2px 7px; }}
    QComboBox#Compact::drop-down {{ width: 18px; }}
    QSpinBox#Compact::up-button, QSpinBox#Compact::down-button,
    QDoubleSpinBox#Compact::up-button,
    QDoubleSpinBox#Compact::down-button {{ width: 12px; }}

    /* One row per track, with the join that follows it. The active row is
       marked with an accent edge rather than a fill: a filled row competes
       with the transition controls sitting inside it. */
    QFrame#QueueRow {{
        background: {p['SURFACE_2']};
        border: 1px solid {p['HAIRLINE']};
        border-left: 3px solid transparent;
        border-radius: {RADIUS_SM}px;
        margin: 3px 0;
    }}
    QFrame#QueueRow:hover {{ border-color: {p['HAIRLINE_HI']};
                             border-left-color: {p['HAIRLINE_HI']}; }}
    QFrame#QueueRow[active="true"] {{
        border-left: 3px solid {p['GRAPHIC']};
        background: {p['ACCENT_WASH']};
    }}

    /* ---- transport ---- */
    QFrame#Transport {{
        background: {p['SURFACE']};
        border: 1px solid {p['HAIRLINE']};
        border-radius: {RADIUS}px;
    }}
    QSlider {{ min-height: {HIT_MIN}px; }}
    QSlider::groove:horizontal {{
        height: 4px; background: {p['SURFACE_2']};
        border: 1px solid {p['HAIRLINE_HI']}; border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{ background: {p['GRAPHIC']};
                                    border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: {p['TEXT']}; width: 13px; height: 13px;
        margin: -6px 0; border-radius: 6px;
    }}
    QSlider::handle:horizontal:hover {{ background: {p['GRAPHIC']}; }}
    QSlider:focus::handle:horizontal {{ background: {p['FOCUS']};
                                        width: 15px; height: 15px;
                                        margin: -7px 0; border-radius: 7px; }}

    QToolTip {{
        background: {p['SURFACE_3']}; color: {p['TEXT']};
        border: 1px solid {p['HAIRLINE_HI']}; border-radius: 7px;
        padding: 7px 10px;
    }}

    QSplitter::handle {{ background: transparent; }}
    QSplitter::handle:hover {{ background: {p['ACCENT_WASH']}; }}
    """


# ------------------------------------------------------------- self-audit ---
def _lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_):
    h = hex_.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(fg, bg):
    """WCAG 2.1 contrast ratio between two opaque colours."""
    a, b = luminance(fg), luminance(bg)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def blend(top, alpha, bottom):
    """Composite `top` at `alpha` (0-255) over opaque `bottom`, as hex.

    Contrast is a property of what is actually on the screen, so a translucent
    fill has to be resolved before it can be measured. The timeline's track
    blocks fade from alpha 210 to alpha 95 down their height, which means the
    bottom of a block is mostly the card behind it -- and a label chosen
    against the block's nominal colour is being chosen against a colour that
    is not there.
    """
    a = max(0.0, min(1.0, alpha / 255.0))
    t, b = top.lstrip("#"), bottom.lstrip("#")
    out = []
    for i in (0, 2, 4):
        tv, bv = int(t[i:i + 2], 16), int(b[i:i + 2], 16)
        out.append(round(tv * a + bv * (1 - a)))
    return "#%02X%02X%02X" % tuple(out)


def on_colour(background, dark=None, light=None):
    """The better of a dark or light foreground for an arbitrary background.

    Needed wherever the background is data rather than design: the timeline
    paints one block per track from the SERIES ramp, so no fixed token can be
    right for the label on top of it. A white label is correct on the deep
    blue and close to unreadable on the amber, and picking one token for both
    means picking which of the two is broken.
    """
    dark = dark or "#0A1929"
    light = light or "#FFFFFF"
    return dark if contrast(dark, background) >= contrast(light, background) \
        else light


def audit(mode):
    """Every (foreground, background, requirement) pair the app can produce.

    Returns a list of (label, fg, bg, ratio, required, ok). Decorative
    hairlines are deliberately absent: WCAG 1.4.11 exempts them, and asserting
    a requirement that does not exist would force the design to shout.
    """
    p = palette(mode)
    grounds = [("ground", p["BG"]), ("card", p["SURFACE"]),
               ("raised", p["SURFACE_2"])]
    rows = []

    def add(label, fg, bg, need):
        r = contrast(fg, bg)
        rows.append((label, fg, bg, r, need, r >= need - 1e-9))

    for gname, g in grounds:
        add(f"text on {gname}", p["TEXT"], g, 4.5)
        add(f"dim text on {gname}", p["TEXT_DIM"], g, 4.5)
        add(f"muted text on {gname}", p["TEXT_MUTED"], g, 4.5)
        add(f"accent text on {gname}", p["ACCENT_TEXT"], g, 4.5)
        add(f"success on {gname}", p["SUCCESS"], g, 4.5)
        add(f"warning on {gname}", p["WARNING"], g, 4.5)
        add(f"danger on {gname}", p["DANGER"], g, 4.5)
        add(f"control edge on {gname}", p["CONTROL_EDGE"], g, 3.0)
        add(f"focus ring on {gname}", p["FOCUS"], g, 3.0)
        add(f"graphic on {gname}", p["GRAPHIC"], g, 3.0)
        for i, c in enumerate(p["SERIES"]):
            add(f"series[{i}] on {gname}", c, g, 3.0)
        for name, c in p["SEGMENTS"].items():
            add(f"segment {name} on {gname}", c, g, 3.0)

    # Text and icons on the accent fill.
    add("on-accent text", p["ON_ACCENT"], p["ACCENT"], 4.5)
    add("on-accent, hover fill", p["ON_ACCENT"], p["ACCENT_HI"], 4.5)
    add("on-accent, pressed fill", p["ON_ACCENT"], p["ACCENT_DIM"], 4.5)
    # A focus ring must also clear the fill it is drawn against. On filled
    # buttons that ring is ON_ACCENT, not FOCUS -- see the #Primary:focus rule.
    add("focus ring vs input fill", p["FOCUS"], p["SURFACE_2"], 3.0)
    add("focus ring vs accent fill", p["ON_ACCENT"], p["ACCENT"], 3.0)
    add("focus ring vs hover fill", p["ON_ACCENT"], p["ACCENT_HI"], 3.0)
    add("graphic vs its track", p["GRAPHIC"], p["SURFACE_2"], 3.0)
    add("large text on card", p["TEXT"], p["SURFACE"], 3.0)
    return rows


def audit_fonts():
    """Report what each type step actually resolves to on this machine.

    A missing family is the quietest failure in the whole design system: Qt
    substitutes silently, every measurement in the layout shifts, and nothing
    anywhere says so.
    """
    import sys
    from PyQt6.QtGui import QFontInfo
    from PyQt6.QtWidgets import QApplication
    owned = QApplication.instance() is None
    app = QApplication(sys.argv) if owned else QApplication.instance()
    rows = []
    for name in TYPE:
        want = TYPE[name]
        info = QFontInfo(qfont(name))
        wanted_family = (MONO_FAMILIES if name == "mono" else FAMILIES)[0]
        rows.append((name, wanted_family, info.family(), want[0],
                     info.pixelSize(), want[1], info.weight()))
    return rows


def _main():
    for (name, want_fam, got_fam, want_px, got_px,
         want_w, got_w) in audit_fonts():
        flag = "" if got_fam == want_fam else f"  <-- NOT {want_fam}"
        size = "" if got_px == want_px else f"  <-- size {want_px} -> {got_px}"
        print(f"{name:9} {got_fam:22} {got_px:2}px w{got_w}{flag}{size}")

    total = fails = 0
    for mode in PALETTES:
        rows = audit(mode)
        bad = [r for r in rows if not r[-1]]
        total += len(rows)
        fails += len(bad)
        print(f"\n=== {mode.upper()}  "
              f"{len(rows) - len(bad)}/{len(rows)} pass ===")
        for label, fg, bg, r, need, ok in rows:
            if not ok:
                print(f"  FAIL {label:34} {fg} on {bg}  "
                      f"{r:5.2f} < {need}")
        worst = min(rows, key=lambda r: r[3] / r[4])
        print(f"  tightest passing margin: {worst[0]} "
              f"{worst[3]:.2f} vs {worst[4]} required")
    print(f"\n{total - fails}/{total} pairs pass across both modes")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
