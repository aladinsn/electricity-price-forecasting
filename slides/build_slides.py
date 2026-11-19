"""Build the final presentation (CLAUDE.md section 11) from the figures in figures/.

Run from the repository root:

    uv run --with python-pptx python slides/build_slides.py

Writes slides/presentation.pptx. The numbers quoted on the slides come from 06_results.ipynb
(main test 2026-04-01 → 2026-09-30, control 2026-10-16 → 2026-11-08); update them here if the
results are recomputed. Speaker notes carry the talking points and a target time per slide.
"""

import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "slides" / "presentation.pptx"

FONT = "Arial"
NAVY = RGBColor(0x1B, 0x2A, 0x41)
SLATE = RGBColor(0x32, 0x4A, 0x5F)
MUTED = RGBColor(0x5C, 0x6B, 0x7A)
LIGHT = RGBColor(0xEE, 0xF2, 0xF6)
AMBER = RGBColor(0xF2, 0xC1, 0x4E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
# Same hues as the matplotlib figures: classical C0, TimesFM 3 C1, Timestral C2, baselines C7
C_CLASSICAL = RGBColor(0x1F, 0x77, 0xB4)
C_TIMESFM = RGBColor(0xFF, 0x7F, 0x0E)
C_TIMESTRAL = RGBColor(0x2C, 0xA0, 0x2C)
C_BASELINE = RGBColor(0x7F, 0x7F, 0x7F)

W, H = 13.333, 7.5
MARGIN = 0.6
FOOTER = "Day-ahead electricity price forecasting, DE-LU · Aladin & Léon · MSc HES-SO"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]
TOTAL = 20  # checked at the end


# --------------------------------------------------------------------------------------------- helpers
def _runs(paragraph, text: str, size: float, color: RGBColor, bold: bool = False, italic: bool = False):
    """Add text to a paragraph; segments between ** are bold."""
    for k, part in enumerate(re.split(r"\*\*", text)):
        if not part:
            continue
        run = paragraph.add_run()
        run.text = part
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.bold = bold or (k % 2 == 1)
        run.font.italic = italic


def text(slide, x, y, w, h, content, size=16, color=NAVY, bold=False, italic=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, bullets=False, space_after=6, name=None):
    """Text box. `content` is a string or a list of strings (one paragraph each);
    an item can be a tuple (text, level) for a sub-bullet."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = anchor
    items = content if isinstance(content, list) else [content]
    for i, item in enumerate(items):
        line, level = item if isinstance(item, tuple) else (item, 0)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        item_size = size - 2 * level
        if bullets:
            _bullet(p, level, item_size)
        _runs(p, line, item_size, color, bold, italic)
    return box


def _bullet(p, level, size):
    """Real paragraph bullet (no literal bullet character in the text)."""
    from pptx.oxml.ns import qn

    indent = 0.25 + 0.3 * level
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Emu(Inches(indent))))
    pPr.set("indent", str(-Emu(Inches(0.22))))
    for tag in ("a:buClr", "a:buSzPct", "a:buFont", "a:buChar", "a:buNone"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    clr = pPr.makeelement(qn("a:buClr"), {})
    srgb = clr.makeelement(qn("a:srgbClr"), {"val": "324A5F" if level == 0 else "5C6B7A"})
    clr.append(srgb)
    font = pPr.makeelement(qn("a:buFont"), {"typeface": FONT})
    char = pPr.makeelement(qn("a:buChar"), {"char": "•" if level == 0 else "–"})
    pPr.extend([clr, font, char])


def rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE, line=None, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    if name:
        s.name = name
    return s


def card(slide, x, y, w, h, title, body, accent=SLATE, size=14, title_size=17):
    """Light card with a coloured title and body text."""
    rect(slide, x, y, w, h, LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(slide, x + 0.2, y + 0.15, w - 0.4, 0.45, title, size=title_size, color=accent, bold=True)
    text(slide, x + 0.2, y + 0.65, w - 0.4, h - 0.8, body, size=size, color=NAVY, bullets=isinstance(body, list),
         space_after=4)


def image(slide, name, x, y, w=None, h=None, alt=""):
    """Picture keeping the aspect ratio of the PNG (give w or h)."""
    path = FIG / name
    iw, ih = Image.open(path).size
    if w is None:
        w = h * iw / ih
    if h is None:
        h = w * ih / iw
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    pic._element.nvPicPr.cNvPr.set("descr", alt or name)
    return pic


def table(slide, rows, x, y, col_widths, row_h=0.42, size=14, header_fill=NAVY, first_col_bold=True,
          fills=None, bold_cells=(), center=True):
    """Native table. rows[0] is the header. fills: {(r, c): RGBColor}."""
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y), Inches(sum(col_widths)),
                                   Inches(row_h * n_rows))
    tbl = shape.table
    for c, cw in enumerate(col_widths):
        tbl.columns[c].width = Inches(cw)
    for r in range(n_rows):
        tbl.rows[r].height = Inches(row_h)
        for c in range(n_cols):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if (center and c > 0) else PP_ALIGN.LEFT
            header = r == 0
            bold = header or (c == 0 and first_col_bold) or (r, c) in bold_cells
            _runs(p, rows[r][c], size, WHITE if header else NAVY, bold=bold)
            cell.fill.solid()
            if header:
                cell.fill.fore_color.rgb = header_fill
            elif fills and (r, c) in fills:
                cell.fill.fore_color.rgb = fills[(r, c)]
            else:
                cell.fill.fore_color.rgb = LIGHT if r % 2 else WHITE
    return shape


def stat(slide, x, y, w, value, label, color=NAVY, value_size=40, label_size=13, label_color=MUTED):
    value_h = value_size / 72 * 1.45
    text(slide, x, y, w, value_h, value, size=value_size, color=color, bold=True)
    text(slide, x, y + value_h + 0.05, w, 0.7, label, size=label_size, color=label_color)


_count = 0


def new_slide(title=None, section=None, dark=False, notes=""):
    """Blank slide with background, title, section tag and footer. Returns the slide."""
    global _count
    _count += 1
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = NAVY if dark else WHITE
    if title:
        if section:
            text(s, MARGIN, 0.3, 8, 0.32, section.upper(), size=12, color=MUTED if not dark else AMBER, bold=True,
                 name="Section")
        text(s, MARGIN, 0.58, W - 2 * MARGIN, 0.75, title, size=28, color=WHITE if dark else NAVY, bold=True,
             name="Title")
    if not dark:
        text(s, MARGIN, H - 0.45, 9, 0.3, FOOTER, size=10, color=MUTED, name="Footer")
        text(s, W - MARGIN - 1.2, H - 0.45, 1.2, 0.3, f"{_count} / {TOTAL}", size=10, color=MUTED,
             align=PP_ALIGN.RIGHT, name="Slide number")
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


CONTENT_Y = 1.5  # top of the content area, below the title


# --------------------------------------------------------------------------------------------- 1. title
s = new_slide(dark=True, notes=(
    "≈ 0:45. Introduce ourselves and the question in one sentence: we forecast tomorrow's 24 hourly electricity "
    "prices in Germany-Luxembourg, and we ask two things: how much do forecasts of load, wind and solar help, and "
    "do the new pre-trained foundation models, used without any training, beat the models of the course?"))
text(s, MARGIN, 0.9, 6, 0.4, "TIME SERIES COURSE PROJECT · MSc HES-SO LAUSANNE", size=13, color=AMBER, bold=True)
text(s, MARGIN, 1.4, 11.5, 1.75, "Forecasting tomorrow's electricity prices", size=40, color=WHITE, bold=True)
text(s, MARGIN, 3.35, 11.5, 1.2,
     "How much do covariates help, and do zero-shot foundation models beat the classical models of the course?",
     size=22, color=LIGHT)
text(s, MARGIN, 4.75, 11.5, 0.5, "German-Luxembourg (DE-LU) day-ahead market · April – November 2026", size=16,
     color=LIGHT)
text(s, MARGIN, 6.2, 8, 0.5, "Aladin & Léon · 24 November 2026", size=16, color=WHITE, bold=True)

# --------------------------------------------------------------------------------------------- 2. market
s = new_slide("One auction a day sets 24 prices for tomorrow", "Context", notes=(
    "≈ 1:30. Day-ahead market: every day before 12:00 Berlin time, producers and buyers submit bids for each hour "
    "(now each quarter-hour) of the next day; the European market coupling clears one price per period and per zone. "
    "The price is set by the last plant needed: the merit order. Wind and solar have almost zero marginal cost: when "
    "there is a lot of them, the cheap plants are enough and the price is low, even negative (inflexible plants pay to "
    "keep running). With little wind and high demand, gas plants set the price. DE-LU has a lot of wind and solar, "
    "so it is the zone where weather forecasts should matter most."))
# Timeline of the auction
text(s, MARGIN, CONTENT_Y, 5.6, 0.4, "Timeline", size=18, color=SLATE, bold=True)
rect(s, MARGIN + 0.1, 2.55, 5.3, 0.06, MUTED)
for xx, top, bottom in [(0.7, "D−1, morning", "Forecasts of load, wind, solar for D are published"),
                        (2.5, "D−1, 12:00", "Gate closure: bids for the 24 h of D"),
                        (4.3, "D−1, ≈ 13:00", "One price per hour of D is published")]:
    rect(s, xx + 0.55, 2.45, 0.26, 0.26, AMBER if "12:00" in top else SLATE, MSO_SHAPE.OVAL)
    text(s, xx, 2.85, 1.75, 0.4, top, size=14, color=NAVY, bold=True)
    text(s, xx, 3.25, 1.75, 1.4, bottom, size=13, color=MUTED)
text(s, MARGIN, 4.75, 5.6, 1.9, [
    "Our forecast is issued on **D−1 before 12:00**: the 24 hourly prices of day D",
    "Since Oct. 2025 the market trades quarter-hours: we average them to hours",
], size=15, bullets=True)
# Merit order sketch
X0, Y0, PW, PH = 6.9, 1.6, 5.8, 4.4  # plot area
text(s, X0, CONTENT_Y, PW, 0.4, "Merit order: the last plant needed sets the price", size=18, color=SLATE, bold=True)
stack = [("Wind + solar", 1.5, 0.08, C_TIMESTRAL), ("Nuclear, lignite", 1.0, 0.45, C_BASELINE),
         ("Coal", 0.9, 0.95, SLATE), ("Gas", 1.2, 1.7, C_TIMESFM), ("Oil", 0.5, 2.7, RGBColor(0xC0, 0x39, 0x2B))]
base_y = Y0 + PH
xx = X0 + 0.4
for label, width, height, col in stack:
    rect(s, xx, base_y - height, width - 0.05, height, col)
    text(s, xx - 0.05, base_y + 0.05, width + 0.1, 0.5, label, size=12, color=NAVY, align=PP_ALIGN.CENTER)
    xx += width
rect(s, X0 + 0.4, base_y, 5.2, 0.03, NAVY)
rect(s, X0 + 0.4, Y0 + 1.0, 0.03, PH - 1.0, NAVY)
text(s, X0 - 0.35, Y0 + 0.75, 1.6, 0.3, "price", size=12, color=MUTED)
text(s, X0 + 5.0, base_y + 0.45, 1.0, 0.3, "capacity", size=12, color=MUTED, align=PP_ALIGN.RIGHT)
for dx, label in [(1.6, "windy, sunny Sunday:\nlow or negative price"), (4.3, "calm winter evening:\ngas sets a high price")]:
    rect(s, X0 + 0.4 + dx, Y0 + 1.25, 0.04, PH - 1.25, AMBER)
    text(s, X0 + 0.4 + dx - 1.0 + (0.0 if dx < 3 else -0.4), Y0 + 0.55, 2.4, 0.7, label, size=12, color=NAVY,
         bold=True, align=PP_ALIGN.CENTER)

# --------------------------------------------------------------------------------------------- 3. question
s = new_slide("Research question: a two-way table", "Context", notes=(
    "≈ 1:00. The whole project is this table. A row tells us what the covariates bring inside one model family. A "
    "column compares families that see exactly the same information. The covariates are the same four for every "
    "model: forecasted load, wind, solar and a non-working-day flag. Foundation models are used zero-shot: no "
    "training, no fine-tuning, no hyperparameter search."))
text(s, MARGIN, CONTENT_Y, W - 2 * MARGIN, 0.9,
     "How much do covariates (load, wind and solar forecasts, calendar) improve day-ahead price forecasts, and do "
     "pre-trained foundation models used zero-shot beat the classical models seen in the course?",
     size=18, color=SLATE, italic=True)
table(s, [["", "Without covariates", "With covariates"],
          ["Classical (course)", "SARIMA", "SARIMAX"],
          ["Foundation model, zero-shot", "TimesFM 3, Timestral", "TimesFM 3 + cov., Timestral + cov."]],
      MARGIN, 2.75, [3.73, 3.7, 4.7], row_h=0.6, size=17)
card(s, MARGIN, 4.85, 5.85, 1.75, "A row = value of the covariates",
     "Same model family, with or without the forecasts of load, wind, solar and the non-working-day flag",
     accent=SLATE, size=15)
card(s, MARGIN + 6.25, 4.85, 5.85, 1.75, "A column = family against family",
     "Same information for both: prices only, or prices + the same four covariates", accent=SLATE, size=15)

# --------------------------------------------------------------------------------------------- 4. data
s = new_slide("Data: ENTSO-E prices and day-ahead forecasts, 2023 – 2026", "Data", notes=(
    "≈ 1:30. Everything comes from the ENTSO-E Transparency Platform through entsoe-py. Important: the covariates are "
    "always the day-ahead forecasts published by the grid operators, never the actual load or generation, also in the "
    "past part of the context, so the model sees the same kind of data in the past and in the future. Prices are "
    "checked against Energy-Charts. Periods: we start in 2023 to avoid the 2021–2022 energy crisis; everything we "
    "chose (SARIMA orders) was chosen on the pre-test period only. The control period comes after the release of both "
    "foundation models: we use it to check for contamination."))
table(s, [["Series (DE-LU, hourly)", "Nature", "Unit"],
          ["Day-ahead price", "auction result: the target", "EUR/MWh"],
          ["load_fc", "day-ahead load forecast", "GW"],
          ["wind_fc", "onshore + offshore forecast", "GW"],
          ["solar_fc", "day-ahead solar forecast", "GW"],
          ["hour, dow, non_working_day", "calendar (+ German holidays)", "–"]],
      MARGIN, CONTENT_Y + 0.05, [2.8, 2.9, 1.1], row_h=0.42, size=13)
text(s, 7.9, CONTENT_Y, 4.8, 2.7, [
    "**Forecasts only** for load, wind, solar: never the actual values",
    "15-min → hourly mean, in UTC, then Berlin time with **24 h per day** (DST: interpolate / average)",
    "Gaps: single missing hours (autumn DST nights), interpolated",
    "Spikes and negative prices are kept as they are",
], size=14, bullets=True)
# Period timeline (not to scale)
TY = 5.3
text(s, MARGIN, TY - 0.75, 6, 0.4, "Periods (not to scale)", size=18, color=SLATE, bold=True)
blocks = [  # x, width, label, colour, dates
    (MARGIN, 5.6, "Pre-test: exploration, SARIMA orders", SLATE, "1 Jan 2023 – 31 Mar 2026"),
    (MARGIN + 5.7, 3.9, "Main test: all models", C_CLASSICAL, "1 Apr – 30 Sep 2026, 183 days"),
    (MARGIN + 10.1, 2.0, "Control", AMBER, "16 Oct – 8 Nov, 24 days"),
]
for x, w, label, col, sub in blocks:
    rect(s, x, TY, w, 0.45, col)
    text(s, x + 0.05, TY + 0.05, w - 0.1, 0.38, label, size=13, color=NAVY if col == AMBER else WHITE, bold=True)
    text(s, x, TY + 0.52, w, 0.32, sub, size=12, color=MUTED)
# Release dates: TimesFM 3 at the end of August (5/6 of the main test), Timestral between test and control
x_tfm, x_tst = MARGIN + 5.7 + 3.9 * 5 / 6, MARGIN + 9.85
for x in (x_tfm, x_tst):
    rect(s, x - 0.015, TY - 0.22, 0.03, 0.22, NAVY)
text(s, x_tfm - 3.05, TY - 0.5, 3.0, 0.3, "TimesFM 3 released 31 Aug", size=11, color=NAVY, align=PP_ALIGN.RIGHT)
text(s, x_tst + 0.05, TY - 0.5, 2.6, 0.3, "Timestral released 15 Oct", size=11, color=NAVY)

# --------------------------------------------------------------------------------------------- 5. exploration
s = new_slide("Prices have a daily shape, cheaper weekends and heavy tails", "Data", notes=(
    "≈ 1:15. Lecture 2 tools. Daily shape: morning peak, midday dip caused by solar, evening peak at 19–20 h. Weekends "
    "and holidays are cheaper and have no morning peak: hence a daily season and the non-working-day flag. The "
    "distribution has heavy tails: about 5 % of hours are negative, and extremes go from −500 to 936 EUR/MWh. We do "
    "not filter them: they are real prices, and they will be the hardest cases."))
image(s, "price_profiles_boxplots.png", MARGIN, CONTENT_Y, w=W - 2 * MARGIN,
      alt="Boxplots of the price by hour of day and by day of week, 2023 to 2026")
stat(s, MARGIN, 5.4, 3.6, "≈ 5 %", "of hours have a negative price", color=C_TIMESTRAL)
stat(s, MARGIN + 4.1, 5.4, 3.6, "−500 … 936", "EUR/MWh: lowest and highest hourly price, 2023 – 2026",
     color=RGBColor(0xC0, 0x39, 0x2B))
stat(s, MARGIN + 8.2, 5.4, 3.9, "24 h, 168 h", "seasons: ACF peaks at lags 24 and 168", color=SLATE)

# --------------------------------------------------------------------------------------------- 6. covariates
s = new_slide("Wind and solar push prices down, load pushes them up", "Data", notes=(
    "≈ 1:00. Lecture 3: correlation between the price and each covariate forecast. Wind and solar lower the price, "
    "load raises it; the very high prices only happen when wind and solar are low. Correlations are moderate and "
    "they do not prove causality: solar and load both follow the time of day. This is why we let the models use all "
    "four covariates together."))
image(s, "price_vs_covariates.png", MARGIN, CONTENT_Y, w=W - 2 * MARGIN,
      alt="Scatter plots of the price against the load, wind and solar forecasts")
for k, (value, label, col) in enumerate([("r = +0.29", "price vs load forecast", SLATE),
                                         ("r = −0.37", "price vs wind forecast", SLATE),
                                         ("r = −0.43", "price vs solar forecast", SLATE)]):
    stat(s, MARGIN + k * 3.2, 5.4, 3.0, value, label, color=col, value_size=32)
text(s, 10.2, 5.45, 2.5, 1.2, "Moderate: no covariate explains the price alone. Correlation ≠ causality.",
     size=13, color=MUTED, italic=True)

# --------------------------------------------------------------------------------------------- 7. task
s = new_slide("The task: 24 prices of day D, with what is known on D−1", "Protocol", notes=(
    "≈ 1:15. This is the rule that protects the evaluation from leakage. At the forecast origin we know prices up to "
    "the end of D−1, the forecasts of load, wind and solar for D, and the calendar. We never use actual load or "
    "generation, not even in the past. One limitation: wind and solar forecasts can legally be published until 18:00, "
    "so possibly after gate closure. Using them is standard practice in the literature (epftoolbox), we mention it."))
table(s, [["Information at the origin (D−1, before 12:00)", "Used?"],
          ["Prices up to the end of D−1", "yes"],
          ["Day-ahead forecasts of load, wind, solar for D (and their past values)", "yes"],
          ["Calendar of D: hour, day of week, German public holidays", "yes"],
          ["Prices of day D", "no: target"],
          ["Actual load, wind, solar (past or future)", "never"]],
      MARGIN, CONTENT_Y + 0.05, [6.6, 1.5], row_h=0.5, size=15,
      fills={(1, 1): RGBColor(0xD8, 0xEF, 0xD8), (2, 1): RGBColor(0xD8, 0xEF, 0xD8),
             (3, 1): RGBColor(0xD8, 0xEF, 0xD8), (4, 1): RGBColor(0xF6, 0xD5, 0xD1),
             (5, 1): RGBColor(0xF6, 0xD5, 0xD1)})
card(s, 9.2, CONTENT_Y + 0.05, 3.5, 3.0, "Forecasts in the past too",
     "The covariates are forecasts over the context window too: a model sees the same kind of data before and after "
     "the origin", size=14)
text(s, MARGIN, 5.05, 11.5, 1.5, [
    "**Rolling origin**: one forecast per day D, moving forward one day at a time, on 183 + 24 days",
    "Limitation: wind/solar forecasts may be published until 18:00 on D−1, after gate closure (common practice to use "
    "them)",
], size=15, bullets=True)

# --------------------------------------------------------------------------------------------- 8. models
s = new_slide("Ten models, from naive to foundation models", "Protocol", notes=(
    "≈ 1:45. Baselines: copy yesterday, copy last week, and the reference naive of epftoolbox (last week for Monday "
    "and weekend, yesterday otherwise): this is the denominator of the rMAE. Holt-Winters additive, since prices can "
    "be negative. SARIMA(1,0,1)(1,1,1)24: chosen on the pre-test only with seasonal differencing, ACF/PACF, AIC/BIC and "
    "Ljung-Box, then frozen. Re-estimated every Monday on the last 8 weeks, and on the other days the state is updated "
    "with the new prices without re-estimating (append with refit=False). SARIMAX: same orders plus the four "
    "covariates as a linear regression. Foundation models: TimesFM 3 from Google (331M parameters) and Timestral-209M "
    "from Mistral, both zero-shot, 2,048 hours of context, same four covariates as known-future inputs; they give "
    "quantiles. We turned off Timestral's default clipping at 0, since negative prices are real."))
cw, gap = 3.85, 0.25
card(s, MARGIN, CONTENT_Y, cw, 3.6, "Baselines", [
    "Naive D−1, naive D−7",
    "**Reference naive** (epftoolbox): D−7 on Mon, Sat, Sun, else D−1: denominator of the rMAE",
    "**Holt-Winters** additive, season 168 h, refitted daily on 8 weeks",
], accent=C_BASELINE)
card(s, MARGIN + cw + gap, CONTENT_Y, cw, 3.6, "Classical (lecture 4)", [
    "**SARIMA(1,0,1)(1,1,1)₂₄**, orders chosen on the pre-test only, then frozen",
    "Refit every Monday on 8 weeks; other days: new data, same parameters",
    "**SARIMAX**: same orders + load, wind, solar, non-working day",
], accent=C_CLASSICAL)
card(s, MARGIN + 2 * (cw + gap), CONTENT_Y, cw, 3.6, "Foundation, zero-shot", [
    "**TimesFM 3** (Google, 331M) and **Timestral-209M** (Mistral)",
    "No training: 2,048 h of prices as context",
    "**+ cov.**: the same 4 covariates as SARIMAX, over context + day D",
    "Median + quantiles 10 … 90 %",
], accent=C_TIMESFM)
rect(s, MARGIN, 5.4, W - 2 * MARGIN, 1.05, WHITE, MSO_SHAPE.ROUNDED_RECTANGLE, line=SLATE)
text(s, MARGIN + 0.25, 5.5, W - 2 * MARGIN - 0.5, 0.9,
     "Same information in a column of the table: SARIMAX and the \"+ cov.\" foundation models all get **load_fc, "
     "wind_fc, solar_fc, non_working_day**. Default parameters, no hyperparameter search, no fine-tuning.",
     size=15, anchor=MSO_ANCHOR.MIDDLE)

# --------------------------------------------------------------------------------------------- 9. evaluation
s = new_slide("Evaluation: same days, same horizon, out of sample", "Protocol", notes=(
    "≈ 1:15. One common loop for all ten models: for each day D it gives the model the history up to D−1 and the "
    "covariates of D, and stores the 24 forecasts. All models are compared on the same 4,392 hours. MAE is the main "
    "metric; rMAE divides by the reference naive, below 1 means better than the naive. The pinball loss scores the "
    "quantiles of the foundation models. The Diebold-Mariano test of epftoolbox works on the daily vector of 24 errors, "
    "so it does not suffer from the correlation between the hours of a day."))
# Rolling loop diagram
DY = CONTENT_Y + 0.2
for k in range(4):
    yy = DY + k * 0.55
    rect(s, MARGIN + k * 0.45, yy, 3.6, 0.38, SLATE)
    text(s, MARGIN + k * 0.45 + 0.05, yy + 0.02, 3.5, 0.34, "history: prices + covariate forecasts", size=11,
         color=WHITE)
    rect(s, MARGIN + k * 0.45 + 3.65, yy, 0.75, 0.38, AMBER)
    text(s, MARGIN + k * 0.45 + 3.65, yy + 0.02, 0.75, 0.34, "D" if k == 0 else f"D+{k}", size=12, color=NAVY,
         bold=True, align=PP_ALIGN.CENTER)
text(s, MARGIN, DY + 2.3, 5.8, 1.0,
     "For every day D: fit or update the model on the history up to D−1, forecast the 24 hours of D, move one day",
     size=14, color=MUTED)
table(s, [["Metric", "What it measures"],
          ["MAE (main)", "mean absolute error, EUR/MWh"],
          ["rMAE", "MAE / MAE of the reference naive (< 1: better)"],
          ["RMSE", "weights large errors more (spikes)"],
          ["Pinball loss", "quantiles 10 … 90 % (foundation models)"],
          ["Diebold-Mariano", "is A significantly better than B? daily vectors of 24 errors, α = 5 %"]],
      6.9, CONTENT_Y + 0.05, [1.9, 3.9], row_h=0.55, size=14, center=False)
text(s, MARGIN, 5.05, 12, 0.9,
     "4,392 forecasts per model on the main test (183 days × 24 h), 576 on the control period. Forecasts cached once "
     "per model; nothing tuned on the test.", size=14, color=NAVY)

# --------------------------------------------------------------------------------------------- 10. results overview
s = new_slide("The three models with covariates come first", "Results", notes=(
    "≈ 1:30. rMAE of all ten models on both periods. On the main test the ranking is: Timestral + cov. 0.37, "
    "TimesFM 3 + cov. 0.41, SARIMAX 0.51, then the foundation models without covariates around 0.65–0.69, then "
    "Holt-Winters and SARIMA around 0.8, then the naives. Every model with covariates beats every model without. "
    "On the control period, all rMAE drop because the reference naive is much worse in autumn (D−7 works badly when "
    "the level moves from week to week); the order of the models stays the same."))
image(s, "results_rmae_all_models.png", (W - 11.0) / 2, CONTENT_Y - 0.05, w=11.0,
      alt="Horizontal bar chart of the rMAE of all models on the main test and on the control period")
stat(s, MARGIN, 5.45, 3.8, "13.0 EUR/MWh", "MAE of Timestral + cov., the best model (rMAE 0.37)", color=C_TIMESTRAL,
     value_size=26)
stat(s, MARGIN + 4.1, 5.45, 3.8, "17.9 EUR/MWh", "MAE of SARIMAX, the best course model (rMAE 0.51)",
     color=C_CLASSICAL, value_size=26)
stat(s, MARGIN + 8.2, 5.45, 3.9, "35.0 EUR/MWh", "MAE of the reference naive (rMAE 1)", color=C_BASELINE,
     value_size=26)

# --------------------------------------------------------------------------------------------- 11. two-way table
s = new_slide("Covariates cut the error by about 40 % in every family", "Results", notes=(
    "≈ 1:45. The answer to the research question. Rows: covariates reduce the MAE by 39 % for the classical family, "
    "40 % for TimesFM 3 and 43 % for Timestral. The relative gain hardly depends on the family. Columns: with the same "
    "information, the foundation models are 17 to 22 % better than SARIMA with prices only, and 19 to 27 % better than "
    "SARIMAX with covariates. Diagonal: SARIMAX (17.9) beats the foundation models without covariates (22.8 and 24.3): "
    "on this task, giving the covariates to a course model brings more than switching to a foundation model that "
    "only sees prices. The best result combines both."))
G = RGBColor(0xD8, 0xEF, 0xD8)
table(s, [["MAE, main test [EUR/MWh]", "Without cov.", "With cov.", "Covariate gain"],
          ["Classical (SARIMA → SARIMAX)", "29.4", "17.9", "−39 %"],
          ["TimesFM 3", "24.3", "14.4", "−40 %"],
          ["Timestral-209M", "22.8", "**13.0**", "−43 %"],
          ["Foundation vs classical", "−17 %, −22 %", "−19 %, −27 %", ""]],
      MARGIN, CONTENT_Y + 0.05, [4.53, 2.5, 2.5, 2.6], row_h=0.55, size=16,
      fills={(1, 3): G, (2, 3): G, (3, 3): G})
text(s, MARGIN, 4.45, 9.8, 0.4, "Control period (24 days, autumn): gains −50 % (classical), −32 % (TimesFM 3), "
     "−34 % (Timestral)", size=13, color=MUTED)
card(s, MARGIN, 5.0, 5.85, 1.65, "Diagonal: covariates matter more than the model",
     "SARIMAX 17.9 beats TimesFM 3 (24.3) and Timestral (22.8) without covariates", accent=C_CLASSICAL, size=14,
     title_size=15)
card(s, MARGIN + 6.25, 5.0, 5.85, 1.65, "The best result combines both",
     "Foundation model + covariates: Timestral + cov. 13.0, TimesFM 3 + cov. 14.4", accent=C_TIMESTRAL, size=14,
     title_size=15)

# --------------------------------------------------------------------------------------------- 12. DM tests
s = new_slide("The differences are significant, except Timestral vs TimesFM 3", "Results", notes=(
    "≈ 1:30. Multivariate Diebold-Mariano test, one-sided: green means the column model is significantly better than "
    "the row model. Planned tests: covariates within each family, p below 1e-28; best foundation model against SARIMAX, "
    "p = 1e-18; all ten planned tests survive a Holm correction. Robustness with squared errors (norm 2): everything "
    "holds except Timestral vs TimesFM 3, p = 0.15 and 0.09. Timestral is better on typical hours, but not on the large "
    "errors. We present the two foundation models as close."))
image(s, "dm_pvalues_main_test.png", MARGIN, CONTENT_Y - 0.05, h=5.25,
      alt="Matrices of Diebold-Mariano p-values between all models, norm 1 and norm 2")
text(s, 10.85, CONTENT_Y + 0.1, 1.95, 5.2, [
    "**Covariates**, each family: p < 1e-28",
    "**Best foundation vs SARIMAX**: p ≈ 1e-18",
    "Holds with Holm, norm 2, control",
    "**Timestral vs TimesFM 3**: not robust (norm 2: p = 0.15, 0.09)",
], size=14, bullets=True, space_after=10)

# --------------------------------------------------------------------------------------------- 13. spike
s = new_slide("Spikes: covariates raise the evening peak, no model reaches it", "Analysis", notes=(
    "≈ 1:15. Monday 14 September, 697 EUR/MWh at 19:00, mean wind forecast only 3.2 GW. Without covariates the models "
    "forecast an ordinary evening. With covariates, the three families raise the evening peak to 290–360 EUR/MWh and "
    "halve their error on the day, but nobody reaches 697. The 90 % quantile of Timestral + cov. goes up to about "
    "515. Low wind is necessary for a spike, but its size depends on gas prices and available capacity, which no model "
    "sees."))
image(s, "forecast_example_spike.png", MARGIN, CONTENT_Y, w=W - 2 * MARGIN,
      alt="Forecasts of all families on 14 September 2026, the day with the highest price of the test")
text(s, MARGIN, 5.25, 12, 1.5, [
    "Mon 14 Sep: **697 EUR/MWh** at 19:00, wind forecast 3.2 GW; with covariates the peak is raised to **290 – 360**",
    "On all spike hours (> 222 EUR/MWh): forecasts too low by 35 – 89 EUR/MWh; 10–90 % bands cover 44 – 71 %, not 80 %",
], size=15, bullets=True)

# --------------------------------------------------------------------------------------------- 14. negative
s = new_slide("Negative prices: the covariates get the sign right", "Analysis", notes=(
    "≈ 1:15. Easter Sunday, 18 hours below zero, the windiest day of the test. The models without covariates forecast "
    "an ordinary Sunday. SARIMAX is the best model of that day: its linear regression on wind and solar extrapolates "
    "down to −78. The foundation models with covariates go below zero but stay closer to the range of their context. "
    "Over the whole test, the share of negative hours where the forecast is also negative goes from 16–49 % without "
    "covariates to 54–67 % with."))
image(s, "forecast_example_negative_prices.png", MARGIN, CONTENT_Y, w=W - 2 * MARGIN,
      alt="Forecasts of all families on Easter Sunday, 5 April 2026, with 18 negative hours")
text(s, MARGIN, 5.25, 12, 1.5, [
    "Easter Sunday 5 Apr: 18 h below 0, wind forecast 39.5 GW; **SARIMAX best of the day** (linear extrapolation)",
    "Negative hours forecast as negative: **16 – 49 %** without covariates → **54 – 67 %** with",
], size=15, bullets=True)

# --------------------------------------------------------------------------------------------- 15. by hour
s = new_slide("Errors peak at the evening ramp and the midday solar dip", "Analysis", notes=(
    "≈ 1:00. MAE by hour. Every model is best at night, where prices change slowly. The worst hour is 20:00 for every "
    "model: solar is gone and gas sets the price. Between 6:00 and 18:00 SARIMA is barely better than the naive, "
    "SARIMAX halves its error: that is where wind and solar forecasts matter. By wind: models without covariates "
    "forecast an average day, too low when the wind is weak and too high when it is strong."))
image(s, "analysis_mae_by_hour.png", MARGIN, CONTENT_Y, w=W - 2 * MARGIN,
      alt="MAE by hour of day for the main models")
text(s, MARGIN, 5.55, 12, 1.2, [
    "Worst hour for every model: **20:00** (Timestral + cov. 25.1, SARIMAX 32.6 EUR/MWh); best: night (6 – 12)",
    "Without covariates: an \"average\" day, too low on calm days (bias −11 to −20), too high on windy days (+15 to +21)",
], size=15, bullets=True)

# --------------------------------------------------------------------------------------------- 16. covariate gain
s = new_slide("The covariates help most where wind and solar set the price", "Analysis", notes=(
    "≈ 1:15. The covariate gain of each family by group. It is largest on windy days (46–51 %), in the morning and at "
    "midday, and on negative hours. It is smallest on low-wind days and on spikes, where gas sets the price. Two "
    "differences between families: at night the classical family gains only 8 % (one linear regression for all "
    "hours), the foundation models 18–23 %; on negative hours the classical family gains the most, thanks to linear "
    "extrapolation. Overall, the gain is similar: the covariates do not favour one family."))
image(s, "analysis_covariate_gain.png", MARGIN, CONTENT_Y, w=W - 2 * MARGIN,
      alt="Covariate gain per family by hour block, wind quartile and price regime")
text(s, MARGIN, 5.55, 12, 1.2, [
    "Largest on **windy days** (46 – 51 %) and negative hours; smallest on **low-wind days and spikes** (27 – 39 %)",
    "Night: classical +8 % only vs 18 – 23 % for foundation models; negative hours: classical gains most (47 %)",
], size=15, bullets=True)

# --------------------------------------------------------------------------------------------- 17. failure case
s = new_slide("Failure case: the models know the sign, not the depth", "Analysis", notes=(
    "≈ 1:15. 1 May, Labour Day, −499 EUR/MWh at 13:00, almost the market floor of −500. High solar, a holiday and wind. "
    "All the models with covariates forecast a negative dip, but none goes deeper than about −100. The depth depends "
    "on the flexibility of the system: exports, storage, inflexible plants, which no covariate describes. The hardest "
    "days are the same for every model: deep negative days around late April – early May and low-wind spike days. "
    "The 10 % worst days are 24–28 % of the error for every model, so the ranking is not driven by a few days."))
image(s, "analysis_worst_day.png", MARGIN, CONTENT_Y, w=W - 2 * MARGIN,
      alt="Forecasts of all families on 1 May 2026, the worst day of the best model")
text(s, MARGIN, 5.25, 12, 1.5, [
    "1 May (holiday): **−499 EUR/MWh** at 13:00; models with covariates go down to −41 … −102 only",
    "Hardest days are the **same for all models**: deep negative days (Apr – May) and low-wind spike days",
], size=15, bullets=True)

# --------------------------------------------------------------------------------------------- 18. contamination
s = new_slide("No sign that the foundation models had seen the test prices", "Analysis", notes=(
    "≈ 1:45. Both foundation models were released during or right after the main test, so the test prices might be in "
    "their training data. TimesFM 3's model card lists its data: nothing from 2026 and no price series. Timestral "
    "describes only 'a mixture of real and synthetic data', without a cutoff, so the documentation cannot exclude it. "
    "Empirical check: compare to SARIMAX, which cannot be contaminated, on the main test and on the control period, "
    "after both releases. If a foundation model had memorised the test, its ratio would get worse on the control "
    "period. It does not: TimesFM 3 + cov. 0.81 → 0.80, Timestral + cov. 0.73 → 0.79, inside the month-to-month "
    "spread and the bootstrap intervals. Caveats: 24 days only, and a different season."))
image(s, "contamination_relative_mae.png", (W - 11.3) / 2, CONTENT_Y - 0.1, w=11.3,
      alt="MAE relative to SARIMAX by month of the main test and on the control period")
text(s, MARGIN, 5.55, 6.0, 1.3, [
    "**Model cards**: TimesFM 3: no 2026 data, no price series; Timestral: real data and cutoff **not stated**",
    "Limits: 24 control days, autumn vs spring–summer",
], size=14, bullets=True)
text(s, 6.85, 5.55, 5.9, 1.3, [
    "**MAE / SARIMAX**, main → control: TimesFM 3 + cov. 0.81 → 0.80, Timestral + cov. 0.73 → 0.79",
    "Inside the monthly spread and the 95 % bootstrap intervals",
], size=14, bullets=True)

# --------------------------------------------------------------------------------------------- 19. conclusions
s = new_slide("Conclusions", dark=True, notes=(
    "≈ 1:30. Three messages. One: covariate forecasts are the most valuable ingredient, about −40 % MAE in every "
    "family, significant. Two: zero-shot foundation models beat the course models with the same information, by 17 "
    "to 27 %, and Timestral + cov. is the best overall, but TimesFM 3 and Timestral are close. Three: no model "
    "forecasts the extremes: the missing information is fuel prices and system flexibility."))
for k, (value, head, body, col) in enumerate([
    ("−40 %", "Covariates help every family",
     "SARIMA → SARIMAX −39\u00a0%, TimesFM 3 −40\u00a0%, Timestral −43\u00a0% MAE; significant (DM)", AMBER),
    ("−17 to −27 %", "Foundation models beat the course models",
     "with the same information, zero-shot; best: Timestral + cov., 13.0 EUR/MWh (rMAE 0.37)", AMBER),
    ("Spikes, deep negatives", "No model forecasts the extremes",
     "right direction, wrong size: the missing information is fuel prices and flexibility", AMBER),
]):
    x = MARGIN + k * 4.1
    rect(s, x, 1.75, 3.85, 3.9, SLATE, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x + 0.25, 1.95, 3.4, 0.9, value, size=32 if k < 2 else 24, color=col, bold=True, anchor=MSO_ANCHOR.BOTTOM)
    text(s, x + 0.25, 3.0, 3.4, 0.9, head, size=18, color=WHITE, bold=True)
    text(s, x + 0.25, 3.95, 3.4, 1.6, body, size=15, color=LIGHT)
text(s, MARGIN, 6.05, 12, 0.5, "Covariates matter more than the model: SARIMAX (17.9) beats both foundation models "
     "without covariates (22.8, 24.3).", size=15, color=LIGHT, italic=True)

# --------------------------------------------------------------------------------------------- 20. limitations
s = new_slide("Limitations and outlook", "Conclusions", notes=(
    "≈ 1:15. Limitations: wind and solar forecasts may come after gate closure; gas and CO2 prices are missing although "
    "they often set the price, which is why spikes are missed; we forecast hourly averages while the market trades "
    "quarter-hours; the DST convention; and contamination cannot be fully excluded for Timestral. Outlook: the "
    "residuals show volatility clustering, calm and turbulent periods, which is what GARCH models describe. "
    "Thank the audience and open questions."))
card(s, MARGIN, CONTENT_Y, 7.4, 3.75, "Limitations", [
    "Wind/solar forecasts can be published until 18:00 on D−1, after gate closure",
    "**Gas and CO₂ prices are missing**, although they often set the price",
    "Hourly averages, while the market now trades quarter-hours",
    "DST convention: 24 h per day (interpolate / average)",
    "Contamination of Timestral not excluded by its documentation; 24 control days",
], accent=SLATE, size=15)
card(s, 8.3, CONTENT_Y, 4.4, 3.75, "Outlook", [
    "Volatility changes over time (calm vs turbulent periods): **GARCH** models for the variance",
    "Add fuel and CO₂ prices as covariates",
    "Longer control period, more zones",
], accent=C_TIMESTRAL, size=15)
text(s, MARGIN, 5.65, W - 2 * MARGIN, 0.7, "Thank you. Questions?", size=28, color=NAVY, bold=True)

assert _count == TOTAL, _count
prs.save(OUT)
print(f"wrote {OUT.relative_to(ROOT)} ({_count} slides)")
