from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(r"E:\project\D-FINE-master")
SRC = Path(r"E:\桌面\Framework.png")
OUT_DIR = ROOT / "output" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "framework_palette_preview.png"


W, H = 2400, 1350
BG = "#F6F7F9"
PANEL = "#FFFFFF"
INK = "#1F2937"
MUTED = "#6B7280"
LINE = "#374151"
LIGHT_LINE = "#CBD5E1"
BASE_FILL = "#F3F4F6"
BACKBONE_FILL = "#E7EEF9"
CGF_FILL = "#DDF3EE"
ENCODER_FILL = "#F7E8C6"
C3K2_FILL = "#E8F0F7"
DECODER_FILL = "#F4F6F8"
OUTPUT_FILL = "#F8FAFC"
RED = "#D1495B"
BLUE = "#2F5DA8"
TEAL = "#2A9D8F"
AMBER = "#C8841A"


def font(name="arial", size=32, bold=False):
    candidates = []
    if name == "times":
        candidates = [
            r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf",
            r"C:\Windows\Fonts\cambria.ttc",
        ]
    else:
        candidates = [
            r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
        ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


F_TITLE = font("times", 36, True)
F_LABEL = font("arial", 28, True)
F_TEXT = font("arial", 27)
F_SMALL = font("arial", 22)
F_TINY = font("arial", 18)


def text_center(draw, box, text, fnt, fill=INK, spacing=4):
    x1, y1, x2, y2 = box
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        b = draw.textbbox((0, 0), line, font=fnt)
        widths.append(b[2] - b[0])
        heights.append(b[3] - b[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, tw, th in zip(lines, widths, heights):
        draw.text((x1 + (x2 - x1 - tw) / 2, y), line, font=fnt, fill=fill)
        y += th + spacing


def rect(draw, box, fill=PANEL, outline=LINE, width=2, radius=10):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, p1, p2, fill=LINE, width=3):
    draw.line([p1, p2], fill=fill, width=width)
    x1, y1 = p1
    x2, y2 = p2
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 14
    a1 = ang + math.pi * 0.82
    a2 = ang - math.pi * 0.82
    pts = [
        (x2, y2),
        (x2 + size * math.cos(a1), y2 + size * math.sin(a1)),
        (x2 + size * math.cos(a2), y2 + size * math.sin(a2)),
    ]
    draw.polygon(pts, fill=fill)


def add_shadow(base, box, radius=12, offset=(6, 8), blur=12, alpha=45):
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    x1, y1, x2, y2 = box
    ox, oy = offset
    sd.rounded_rectangle((x1 + ox, y1 + oy, x2 + ox, y2 + oy), radius=radius, fill=(15, 23, 42, alpha))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    base.alpha_composite(shadow)


def crop_resize(src, crop, size):
    im = src.crop(crop).resize(size, Image.Resampling.LANCZOS)
    return im


def paste_photo(canvas, photo, xy, size, label=None, bbox=False):
    x, y = xy
    w, h = size
    add_shadow(canvas, (x, y, x + w, y + h), radius=4, offset=(7, 7), blur=10, alpha=35)
    canvas.paste(photo.resize(size, Image.Resampling.LANCZOS), (x, y))
    d = ImageDraw.Draw(canvas)
    d.rectangle((x, y, x + w, y + h), outline="#FFFFFF", width=5)
    if bbox:
        d.rectangle((x + 28, y + 58, x + w - 34, y + h - 60), outline=RED, width=4)
        d.text((x + 32, y + 70), "2:0.88", font=F_SMALL, fill=BLUE)
    if label:
        d.text((x, y - 34), label, font=F_SMALL, fill=INK)


def draw_stack(canvas, photo, xy, size, label=None, bbox=False):
    x, y = xy
    w, h = size
    d = ImageDraw.Draw(canvas)
    for dx, dy, tint in [(34, -30, "#D1D5DB"), (17, -15, "#E5E7EB")]:
        d.rounded_rectangle((x + dx, y + dy, x + dx + w, y + dy + h), radius=2, fill=tint, outline="#FFFFFF", width=4)
    paste_photo(canvas, photo, (x, y), size, label=label, bbox=bbox)


def main():
    src = Image.open(SRC).convert("RGB")
    # Crops from the original framework figure. They are only used as visual placeholders.
    top_in = crop_resize(src, (110, 100, 620, 640), (260, 245))
    bot_in = crop_resize(src, (120, 900, 620, 1410), (260, 245))
    aux_in = crop_resize(src, (120, 1580, 620, 2080), (260, 245))
    top_out = crop_resize(src, (3290, 95, 3810, 640), (270, 245))
    bot_out = crop_resize(src, (3290, 960, 3810, 1510), (270, 245))

    canvas = Image.new("RGBA", (W, H), BG)
    d = ImageDraw.Draw(canvas)

    # Panels.
    rect(d, (35, 35, W - 35, 425), fill=PANEL, outline="#E5E7EB", width=2, radius=6)
    rect(d, (35, 470, W - 35, H - 35), fill=PANEL, outline="#E5E7EB", width=2, radius=6)
    d.line((35, 452, W - 35, 452), fill="#9CA3AF", width=2)
    d.text((1120, 60), "a) baseline", font=F_TITLE, fill=INK)
    d.text((1100, 500), "b) LM²-DETR", font=F_TITLE, fill=INK)

    # Baseline.
    draw_stack(canvas, top_in, (95, 135), (240, 220), "224×224×64")
    rect(d, (510, 150, 800, 320), fill=BASE_FILL, outline=LINE, width=2, radius=4)
    text_center(d, (510, 150, 800, 320), "HGNetv2", F_TEXT)
    rect(d, (965, 150, 1395, 320), fill=BASE_FILL, outline=LINE, width=2, radius=4)
    text_center(d, (965, 150, 1395, 320), "Transformer Encoder", F_TEXT)
    rect(d, (1530, 150, 1840, 320), fill=BASE_FILL, outline=LINE, width=2, radius=4)
    text_center(d, (1530, 150, 1840, 320), "Decoder\n& Detection Head", F_TEXT)
    draw_stack(canvas, top_out, (2050, 135), (250, 220), bbox=False)
    for p1, p2 in [((370, 235), (500, 235)), ((805, 235), (955, 235)), ((1405, 235), (1520, 235)), ((1850, 235), (2035, 235))]:
        arrow(d, p1, p2, fill=LINE, width=3)
    d.text((833, 118), "32×32×2048", font=F_SMALL, fill=INK)

    # Proposed left.
    draw_stack(canvas, bot_in, (95, 655), (240, 220), "224×224×64")
    draw_stack(canvas, aux_in, (95, 1015), (240, 220), "224×224×64")
    rect(d, (455, 630, 735, 840), fill=BACKBONE_FILL, outline=BLUE, width=2, radius=6)
    text_center(d, (455, 630, 735, 840), "L-HGNetv2", F_LABEL, fill=INK)
    rect(d, (455, 995, 735, 1205), fill=BACKBONE_FILL, outline=BLUE, width=2, radius=6)
    text_center(d, (455, 995, 735, 1205), "L-HGNetv2", F_LABEL, fill=INK)
    arrow(d, (370, 760), (445, 760), fill=LINE, width=3)
    arrow(d, (370, 1125), (445, 1125), fill=LINE, width=3)
    d.text((780, 665), "32×32×2048", font=F_SMALL, fill=INK)
    d.text((780, 1030), "32×32×2048", font=F_SMALL, fill=INK)

    # Curved-ish arrows to CGF.
    rect(d, (800, 780, 1065, 1015), fill=CGF_FILL, outline=TEAL, width=3, radius=6)
    text_center(d, (800, 780, 1065, 1015), "Cross-view\nGated Fusion", F_LABEL)
    arrow(d, (735, 735), (795, 835), fill=TEAL, width=4)
    arrow(d, (735, 1100), (795, 960), fill=TEAL, width=4)
    d.text((755, 760), "x", font=F_LABEL, fill=INK)
    d.text((755, 1055), "y", font=F_LABEL, fill=INK)
    arrow(d, (1065, 898), (1135, 898), fill=LINE, width=3)

    # Encoder container.
    rect(d, (1145, 555, 1935, 1250), fill="#FBFCFE", outline="#334155", width=3, radius=6)
    d.text((1315, 580), "Multi-scale Sparse Encoder", font=F_TITLE, fill=INK)
    rect(d, (1180, 640, 1900, 840), fill="#FFFFFF", outline=LIGHT_LINE, width=2, radius=4)
    rect(d, (1180, 875, 1900, 1215), fill="#FFFFFF", outline=LIGHT_LINE, width=2, radius=4)

    # TSA route.
    for i, (x, lab, c) in enumerate([(1210, "S3", "c 2048"), (1295, "S4", "c 1024"), (1380, "S5", "c 2048")]):
        d.polygon([(x, 690), (x + 45, 670), (x + 65, 735), (x + 20, 760)], fill="#E5E7EB", outline=LINE)
        d.text((x + 12, 705), lab, font=F_SMALL, fill=INK)
        d.text((x - 2, 646), c, font=F_TINY, fill=MUTED)
        if i < 2:
            arrow(d, (x + 68, 715), (x + 82, 715), fill=LINE, width=2)
    rect(d, (1515, 660, 1585, 800), fill=ENCODER_FILL, outline=AMBER, width=2, radius=4)
    text_center(d, (1515, 660, 1585, 800), "TSA", F_SMALL, fill=INK)
    rect(d, (1618, 660, 1692, 800), fill="#FFFFFF", outline=LINE, width=2, radius=4)
    text_center(d, (1618, 660, 1692, 800), "Add&Norm", F_TINY)
    rect(d, (1725, 660, 1795, 800), fill="#FFFFFF", outline=LINE, width=2, radius=4)
    text_center(d, (1725, 660, 1795, 800), "FFN", F_SMALL)
    rect(d, (1825, 660, 1895, 800), fill="#FFFFFF", outline=LINE, width=2, radius=4)
    text_center(d, (1825, 660, 1895, 800), "F5", F_SMALL)
    for p1, p2 in [((1448, 715), (1505, 715)), ((1588, 730), (1610, 730)), ((1695, 730), (1718, 730)), ((1798, 730), (1818, 730))]:
        arrow(d, p1, p2, fill=LINE, width=2)

    # C3K2-FPN.
    c3_boxes = [(1220, 940, 1290, 1120), (1440, 940, 1510, 1120), (1660, 940, 1730, 1120)]
    for b in c3_boxes:
        rect(d, b, fill=C3K2_FILL, outline=BLUE, width=2, radius=4)
        text_center(d, b, "C3K2", F_SMALL)
    for b, label in [((1340, 920, 1420, 990), "1×1\nCBS"), ((1340, 1045, 1420, 1115), "3×3\nCBS"), ((1560, 920, 1640, 990), "1×1\nCBS"), ((1560, 1045, 1640, 1115), "3×3\nCBS")]:
        rect(d, b, fill="#FFFFFF", outline=LIGHT_LINE, width=2, radius=4)
        text_center(d, b, label, F_TINY)
    rect(d, (1785, 990, 1855, 1090), fill=OUTPUT_FILL, outline=LINE, width=2, radius=4)
    text_center(d, (1785, 990, 1855, 1090), "Concat", F_TINY)
    d.text((1452, 1178), "C3K2-FPN", font=F_LABEL, fill=INK)
    for p1, p2 in [((1292, 1030), (1332, 960)), ((1422, 960), (1438, 960)), ((1512, 1030), (1552, 960)), ((1642, 960), (1658, 960)), ((1732, 1030), (1778, 1030))]:
        arrow(d, p1, p2, fill=LINE, width=2)

    # Encoder output and decoder.
    arrow(d, (1935, 1030), (1995, 1030), fill=LINE, width=3)
    rect(d, (2005, 925, 2280, 1215), fill=DECODER_FILL, outline=LINE, width=2, radius=4)
    text_center(d, (2005, 925, 2280, 1215), "Decoder\n& Detection Head", F_LABEL)
    draw_stack(canvas, bot_out, (2055, 615), (250, 220), bbox=False)
    arrow(d, (2140, 915), (2140, 850), fill=LINE, width=3)

    # Small legend strip.
    legend_y = 1280
    items = [
        (BACKBONE_FILL, BLUE, "Lightweight backbone"),
        (CGF_FILL, TEAL, "Cross-view fusion"),
        (ENCODER_FILL, AMBER, "Sparse attention"),
        (C3K2_FILL, BLUE, "C3K2-FPN"),
    ]
    x = 65
    for fill, edge, label in items:
        d.rounded_rectangle((x, legend_y, x + 32, legend_y + 20), radius=3, fill=fill, outline=edge, width=2)
        d.text((x + 42, legend_y - 3), label, font=F_TINY, fill=MUTED)
        x += 300

    canvas.convert("RGB").save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
