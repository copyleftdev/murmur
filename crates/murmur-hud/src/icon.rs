//! Murmur's mark, drawn rather than shipped.
//!
//! The icon is the same waveform the overlay draws, computed from the same kind
//! of profile, at whatever size is asked for. That means there is no binary
//! asset to keep in step with the design, no image decoder in the dependency
//! tree, and a tray icon that is crisp at every panel height rather than a
//! scaled 22-pixel PNG.

use std::fmt::Write as _;

/// The accent colour, matching the overlay's listening state.
const MARK: [u8; 3] = [92, 219, 181];
/// The tile behind the mark, for places that expect a solid app icon.
const TILE: [u8; 3] = [18, 19, 23];

const BARS: usize = 7;
/// Samples per axis when deciding how much of a pixel a shape covers.
const SUPERSAMPLE: u32 = 3;

/// Bar geometry as fractions of the canvas, shared by both renderings so the
/// vector icon and the pixel icon cannot drift apart.
///
/// Sized so the whole mark sits inside the slab: at the first attempt the bars
/// were fractionally wider than the tile and hung over its rounded edge.
const BAR_WIDTH: f32 = 0.072;
const BAR_GAP: f32 = 0.045;
const BAR_MIN_HEIGHT: f32 = 0.16;
const BAR_SWELL: f32 = 0.58;
const SLAB_INSET: f32 = 0.04;
const SLAB_RADIUS: f32 = 0.22;

/// Where bar `index` starts, and how tall it is, on a canvas `extent` across.
fn bar(index: usize, extent: f32) -> (f32, f32) {
    let width = extent * BAR_WIDTH;
    let gap = extent * BAR_GAP;
    let span = BARS as f32 * width + (BARS - 1) as f32 * gap;
    let left = (extent - span) / 2.0;

    let position = (index as f32 + 0.5) / BARS as f32;
    // Clamped before the fractional power: `sin` can return a hair below zero
    // at the ends, and a negative base makes it `NaN`.
    let profile = (std::f32::consts::PI * position).sin().max(0.0).powf(0.6);
    (
        left + index as f32 * (width + gap),
        extent * (BAR_MIN_HEIGHT + BAR_SWELL * profile),
    )
}

/// RGBA8 pixels for an icon `size` across.
///
/// With `tile`, the mark sits on a rounded slab, which is what a launcher or
/// window list expects. Without, only the bars are drawn, which is what a panel
/// wants — it composites onto whatever colour the theme happens to be.
#[must_use]
pub fn rgba(size: u32, tile: bool) -> Vec<u8> {
    let mut pixels = vec![0u8; (size * size * 4) as usize];
    let extent = size as f32;

    for y in 0..size {
        for x in 0..size {
            let (mark, slab) = coverage(x, y, extent);
            let offset = ((y * size + x) * 4) as usize;

            let (rgb, alpha) = if tile {
                // The mark is opaque where it covers, the slab beneath it fills
                // the rest, and the rounded corners stay clear.
                let alpha = mark.max(slab);
                let rgb = blend(TILE, MARK, if alpha > 0.0 { mark / alpha } else { 0.0 });
                (rgb, alpha)
            } else {
                (MARK, mark)
            };

            pixels[offset] = rgb[0];
            pixels[offset + 1] = rgb[1];
            pixels[offset + 2] = rgb[2];
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                pixels[offset + 3] = (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    pixels
}

/// ARGB pixels, which is what the `StatusNotifierItem` specification asks for.
#[must_use]
pub fn argb(size: u32, tile: bool) -> Vec<u8> {
    let mut pixels = rgba(size, tile);
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.rotate_right(1);
    }
    pixels
}

/// How much of this pixel the bars and the slab each cover, antialiased.
fn coverage(x: u32, y: u32, extent: f32) -> (f32, f32) {
    let samples = SUPERSAMPLE * SUPERSAMPLE;
    let mut mark = 0u32;
    let mut slab = 0u32;

    for sy in 0..SUPERSAMPLE {
        for sx in 0..SUPERSAMPLE {
            let px = x as f32 + (sx as f32 + 0.5) / SUPERSAMPLE as f32;
            let py = y as f32 + (sy as f32 + 0.5) / SUPERSAMPLE as f32;
            if in_bars(px, py, extent) {
                mark += 1;
            }
            if in_slab(px, py, extent) {
                slab += 1;
            }
        }
    }
    (mark as f32 / samples as f32, slab as f32 / samples as f32)
}

/// The waveform: bars tallest in the middle, with rounded ends.
fn in_bars(px: f32, py: f32, extent: f32) -> bool {
    let width = extent * BAR_WIDTH;

    for i in 0..BARS {
        let (x0, height) = bar(i, extent);
        if px < x0 || px > x0 + width {
            continue;
        }
        let y0 = (extent - height) / 2.0;
        if py < y0 || py > y0 + height {
            continue;
        }

        // Round the ends by clipping the corners of the bar to a capsule.
        let radius = width / 2.0;
        let cx = x0 + radius;
        let centred_y = py.clamp(y0 + radius, y0 + height - radius);
        if (px - cx).powi(2) + (py - centred_y).powi(2) <= radius.powi(2) + 1e-3 {
            return true;
        }
    }
    false
}

/// A rounded slab covering most of the canvas.
fn in_slab(px: f32, py: f32, extent: f32) -> bool {
    let inset = extent * SLAB_INSET;
    let radius = extent * SLAB_RADIUS;
    let (min, max) = (inset, extent - inset);
    if px < min || px > max || py < min || py > max {
        return false;
    }
    let cx = px.clamp(min + radius, max - radius);
    let cy = py.clamp(min + radius, max - radius);
    (px - cx).powi(2) + (py - cy).powi(2) <= radius.powi(2) + 1e-3
}

fn blend(under: [u8; 3], over: [u8; 3], amount: f32) -> [u8; 3] {
    let amount = amount.clamp(0.0, 1.0);
    let mix = |a: u8, b: u8| {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            (f32::from(a) * (1.0 - amount) + f32::from(b) * amount).round() as u8
        }
    };
    [
        mix(under[0], over[0]),
        mix(under[1], over[1]),
        mix(under[2], over[2]),
    ]
}

/// The same mark as scalable vector art, for the icon theme.
///
/// Icon themes want something that is crisp at any size, and an SVG needs no
/// encoder — so the installed icon is generated from the same geometry as the
/// pixels above rather than being a separate asset that can drift from them.
#[must_use]
pub fn svg() -> String {
    let extent = 256.0f32;
    let width = extent * BAR_WIDTH;
    let inset = extent * SLAB_INSET;

    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"256\" height=\"256\" \
         viewBox=\"0 0 256 256\">\n  <rect x=\"{inset}\" y=\"{inset}\" \
         width=\"{size}\" height=\"{size}\" rx=\"{radius}\" fill=\"#{tile:02x}{tile2:02x}{tile3:02x}\"/>\n",
        size = extent - 2.0 * inset,
        radius = extent * SLAB_RADIUS,
        tile = TILE[0],
        tile2 = TILE[1],
        tile3 = TILE[2],
    );

    for i in 0..BARS {
        let (x, height) = bar(i, extent);
        let _ = writeln!(
            out,
            "  <rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{width:.2}\" height=\"{height:.2}\" \
             rx=\"{radius:.2}\" fill=\"#{r:02x}{g:02x}{b:02x}\"/>",
            y = (extent - height) / 2.0,
            radius = width / 2.0,
            r = MARK[0],
            g = MARK[1],
            b = MARK[2],
        );
    }
    out.push_str("</svg>\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alpha_at(pixels: &[u8], size: u32, x: u32, y: u32) -> u8 {
        pixels[((y * size + x) * 4 + 3) as usize]
    }

    #[test]
    fn the_buffer_is_exactly_the_size_asked_for() {
        for size in [16u32, 22, 48, 64, 256] {
            assert_eq!(rgba(size, true).len(), (size * size * 4) as usize);
            assert_eq!(argb(size, false).len(), (size * size * 4) as usize);
        }
    }

    #[test]
    fn the_mark_is_drawn_through_the_middle() {
        let size = 64;
        let pixels = rgba(size, false);
        assert_eq!(
            alpha_at(&pixels, size, size / 2, size / 2),
            255,
            "the centre is empty"
        );
    }

    #[test]
    fn a_panel_icon_leaves_its_corners_clear() {
        let size = 64;
        let pixels = rgba(size, false);
        for (x, y) in [(0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1)] {
            assert_eq!(
                alpha_at(&pixels, size, x, y),
                0,
                "corner {x},{y} was painted"
            );
        }
    }

    #[test]
    fn a_tiled_icon_is_rounded_rather_than_square() {
        let size = 64;
        let pixels = rgba(size, true);
        assert_eq!(
            alpha_at(&pixels, size, 0, 0),
            0,
            "the tile has square corners"
        );
        assert_eq!(alpha_at(&pixels, size, size / 2, size / 2), 255);
    }

    #[test]
    fn the_waveform_is_taller_in_the_middle_than_at_the_ends() {
        let size = 64;
        let pixels = rgba(size, false);
        let painted_rows = |column: u32| {
            (0..size)
                .filter(|y| alpha_at(&pixels, size, column, *y) > 128)
                .count()
        };
        let centre = (0..size).map(painted_rows).max().unwrap_or(0);
        assert!(centre > 0, "nothing was drawn");

        let first_bar = (0..size / 4).map(painted_rows).max().unwrap_or(0);
        assert!(first_bar > 0, "the outer bars are missing");
        assert!(centre > first_bar, "the mark is flat, not a waveform");
    }

    #[test]
    fn argb_is_the_same_pixels_with_the_channels_rotated() {
        let rgba = rgba(16, true);
        let argb = argb(16, true);
        for (rgba, argb) in rgba.chunks_exact(4).zip(argb.chunks_exact(4)) {
            assert_eq!(argb, [rgba[3], rgba[0], rgba[1], rgba[2]]);
        }
    }

    #[test]
    fn the_mark_stays_inside_the_slab() {
        let extent = 256.0f32;
        let inset = extent * SLAB_INSET;
        for i in 0..BARS {
            let (x, height) = bar(i, extent);
            assert!(
                x >= inset,
                "bar {i} starts at {x}, outside the tile at {inset}"
            );
            assert!(
                x + extent * BAR_WIDTH <= extent - inset,
                "bar {i} overhangs the right edge of the tile"
            );
            let top = (extent - height) / 2.0;
            assert!(
                top >= inset && top + height <= extent - inset,
                "bar {i} overhangs vertically"
            );
        }
    }

    /// The packaged icon is a file, and files go stale.
    #[test]
    fn the_installed_icon_matches_what_the_code_draws() {
        let packaged = include_str!("../../../packaging/murmur.svg");
        assert_eq!(
            packaged,
            svg(),
            "packaging/murmur.svg is out of date; regenerate it with `murmur-hud --install`"
        );
    }

    #[test]
    fn the_vector_mark_has_a_bar_for_every_pixel_bar() {
        let svg = svg();
        assert!(svg.starts_with("<svg"), "{svg:.40}");
        assert!(svg.trim_end().ends_with("</svg>"));
        // One slab plus one rectangle per bar.
        assert_eq!(svg.matches("<rect").count(), BARS + 1);
        assert!(!svg.contains("NaN"), "the vector mark contains NaN");
    }

    #[test]
    fn tiny_sizes_still_draw_something() {
        let size = 16;
        let pixels = rgba(size, false);
        assert!(
            pixels.chunks_exact(4).any(|p| p[3] > 0),
            "the mark vanished at {size}px"
        );
    }
}
