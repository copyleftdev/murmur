#!/usr/bin/env python3
"""Generate Murmur's identity from the mark the application already draws.

The waveform is not redrawn here. It is lifted out of packaging/murmur.svg,
which murmur-hud generates from crates/murmur-hud/src/icon.rs -- so the logo on
the website, the icon in the panel and the meter in the overlay are all the same
geometry by construction rather than by discipline.

The wordmark is set in Inter Display Bold and converted to outlines, so the logo
renders identically on a machine that has never heard of Inter.
"""

import pathlib
import re
import subprocess

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "brand"

# The palette. Teal is the product's one accent: it is the listening state in the
# overlay, the mark in the icon, and the only colour on the page that means
# "this is Murmur".
TEAL = "#5cdbb5"
INK = "#eef0f3"        # wordmark on dark
INK_DARK = "#12131a"   # wordmark on light
SLAB = "#12131a"

WORD = "murmur"
# Inter's default advances are set for text, not for a six-letter wordmark;
# opening them slightly stops "murmur" reading as one dense block.
TRACKING = 28  # font units at 2048 upem


def mark_bars() -> tuple[list[tuple[float, float, float, float, float]], tuple[float, float, float, float]]:
    """The waveform bars from the generated icon, and their bounding box."""
    svg = (ROOT / "packaging" / "murmur.svg").read_text()
    rects = re.findall(
        r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" rx="([\d.]+)" fill="([^"]+)"',
        svg,
    )
    bars = [
        (float(x), float(y), float(w), float(h), float(r))
        for x, y, w, h, r, fill in rects
        if fill.lower() == TEAL
    ]
    if not bars:
        raise SystemExit("no bars found in packaging/murmur.svg -- has the icon changed?")

    left = min(b[0] for b in bars)
    right = max(b[0] + b[2] for b in bars)
    top = min(b[1] for b in bars)
    bottom = max(b[1] + b[3] for b in bars)
    return bars, (left, top, right - left, bottom - top)


def wordmark(font_path: pathlib.Path) -> tuple[str, float, float, float]:
    """Outlined `murmur`, its advance width, cap height and descender."""
    font = TTFont(font_path)
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    upem = font["head"].unitsPerEm
    cap = font["OS/2"].sCapHeight if hasattr(font["OS/2"], "sCapHeight") else upem * 0.72

    paths, x = [], 0.0
    for character in WORD:
        name = cmap[ord(character)]
        pen = SVGPathPen(glyphs)
        glyphs[name].draw(pen)
        commands = pen.getCommands()
        if commands:
            paths.append(f'<path transform="translate({x:.1f} 0)" d="{commands}"/>')
        x += font["hmtx"][name][0] + TRACKING

    width = x - TRACKING
    # x-height, not cap height: `murmur` has no capitals and no ascenders, so the
    # x-height is what the eye reads as the height of the word.
    #
    # Everything here stays in font units. Converting to em early is what made
    # the first attempt scale the glyphs twice and produce a 900,000-unit canvas.
    return "\n      ".join(paths), width, float(font["OS/2"].sxHeight), upem


def lockup(ink: str, mark_colour: str, mono: bool = False) -> str:
    """Mark and wordmark on one baseline."""
    bars, (bx, by, bw, bh) = mark_bars()
    glyph_paths, word_width, x_height, _upem = wordmark(FONT)

    height = 100.0                      # design units for the whole lockup
    mark_h = height                     # the mark sets the height
    scale = mark_h / bh
    mark_w = bw * scale
    gap = mark_h * 0.34

    # The word is set so its x-height matches the mark's middle bar region,
    # which is what makes the two read as one object rather than two.
    word_h = mark_h * 0.58
    word_scale = word_h / x_height          # font units -> design units
    word_w = word_width * word_scale
    baseline = height * 0.5 + word_h * 0.5

    total_w = mark_w + gap + word_w
    fill = "currentColor" if mono else mark_colour
    text_fill = "currentColor" if mono else ink

    bar_shapes = "\n      ".join(
        f'<rect x="{(x - bx) * scale:.2f}" y="{(y - by) * scale:.2f}" '
        f'width="{w * scale:.2f}" height="{h * scale:.2f}" rx="{r * scale:.2f}"/>'
        for x, y, w, h, r in bars
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w:.1f} {height:.1f}" \
width="{total_w:.1f}" height="{height:.1f}" role="img" aria-label="Murmur">
  <title>Murmur</title>
  <g fill="{fill}">
      {bar_shapes}
  </g>
  <g fill="{text_fill}" transform="translate({mark_w + gap:.2f} {baseline:.2f}) \
scale({word_scale:.6f} {-word_scale:.6f})">
      {glyph_paths}
  </g>
</svg>
"""


def bare_mark(colour: str) -> str:
    bars, (bx, by, bw, bh) = mark_bars()
    height = 100.0
    scale = height / bh
    shapes = "\n    ".join(
        f'<rect x="{(x - bx) * scale:.2f}" y="{(y - by) * scale:.2f}" '
        f'width="{w * scale:.2f}" height="{h * scale:.2f}" rx="{r * scale:.2f}"/>'
        for x, y, w, h, r in bars
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {bw * scale:.1f} {height:.1f}" \
width="{bw * scale:.1f}" height="{height:.1f}" role="img" aria-label="Murmur">
  <title>Murmur</title>
  <g fill="{colour}">
    {shapes}
  </g>
</svg>
"""


FONT = pathlib.Path("/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf")
if not FONT.exists():
    FONT = pathlib.Path(
        subprocess.check_output(["fc-match", "-f", "%{file}", "Inter:weight=bold"]).decode()
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written = {
        "murmur-logo.svg": lockup(INK, TEAL),
        "murmur-logo-light.svg": lockup(INK_DARK, TEAL),
        "murmur-logo-mono.svg": lockup(INK, TEAL, mono=True),
        "murmur-mark.svg": bare_mark(TEAL),
        "murmur-mark-mono.svg": bare_mark("currentColor"),
        "murmur-icon.svg": (ROOT / "packaging" / "murmur.svg").read_text(),
    }
    for name, content in written.items():
        (OUT / name).write_text(content)
        print(f"  {name:<26} {len(content):>6} bytes")


if __name__ == "__main__":
    main()
