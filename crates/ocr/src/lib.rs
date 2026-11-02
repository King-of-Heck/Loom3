//! pdfcraft-ocr — Scan & OCR ▸ Recognize text (L4).
//!
//! Recognition runs the ocrs engine (MIT/Apache-2.0) with its pre-trained models (CC-BY-SA-4.0,
//! fetched by `cargo xtask models`; see ATTRIBUTION.toml). The caller renders a page to pixels;
//! [`Ocr::recognize`] finds the words in them, and [`text_layer`] turns words placed in user space
//! into page content: invisible text (rendering mode 3) over each word, so the page becomes a
//! searchable image (Acrobat's "Searchable Image (Exact)": the image is left untouched).
//!
//! The models read the Latin alphabet (English and other languages written without accents).

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::path::{Path, PathBuf};

use rten_imageproc::{PointF, RotatedRect};

pub use pdfcraft_fonts::helvetica_width;

/// The model files, as named in ATTRIBUTION.toml.
pub const DETECTION_MODEL: &str = "text-detection.onnx";
pub const RECOGNITION_MODEL: &str = "text-recognition.onnx";

/// The recognition model's alphabet, as it was trained (ocrs-models' `DEFAULT_ALPHABET`): its
/// output classes follow this order. The ocrs crate's own copy has an `E` where the training had
/// `€`, which made a scanned "€250" read as "E250".
pub const ALPHABET: &str = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~€ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// The languages the models read (ISO 639-1); all use the Latin alphabet without accents.
pub const LANGUAGES: &[(&str, &str)] = &[("en", "English")];

#[derive(Debug, thiserror::Error)]
pub enum OcrError {
    /// Shown to end users (release packages ship the models; a source build fetches them with
    /// `cargo xtask models`). The UI catalogs key this exact text.
    #[error("the text recognition models are not installed (reinstall Loom, or set PDFCRAFT_MODELS to the folder that holds them)")]
    NoModels,
    #[error("loading {0}: {1}")]
    Load(String, String),
    #[error("text recognition failed: {0}")]
    Recognize(String),
    #[error("the image is empty")]
    EmptyImage,
}

/// Where the two model files are.
#[derive(Clone, Debug, PartialEq)]
pub struct Models {
    pub detection: PathBuf,
    pub recognition: PathBuf,
}

impl Models {
    /// A complete model pair in `dir`, preferring current ONNX files over legacy RTEN files.
    /// An incomplete update never mixes formats or hides an intact legacy pair.
    pub fn in_dir(dir: &Path) -> Option<Models> {
        [(DETECTION_MODEL, RECOGNITION_MODEL), ("text-detection.rten", "text-recognition.rten")].into_iter().find_map(|(detection, recognition)| {
            let m = Models { detection: dir.join(detection), recognition: dir.join(recognition) };
            (m.detection.is_file() && m.recognition.is_file()).then_some(m)
        })
    }

    /// Look for the models: `$PDFCRAFT_MODELS`, then where the release packages install them
    /// beside the executable ([`Models::dirs_beside_exe`]), then the source tree's `assets/models/`.
    pub fn find() -> Option<Models> {
        Self::search_dirs().iter().find_map(|d| Self::in_dir(d))
    }

    /// The directories [`Models::find`] looks in, in order.
    pub fn search_dirs() -> Vec<PathBuf> {
        let mut dirs = Vec::new();
        if let Some(d) = std::env::var_os("PDFCRAFT_MODELS") {
            dirs.push(PathBuf::from(d));
        }
        if let Some(exe) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
            dirs.extend(Self::dirs_beside_exe(&exe));
        }
        dirs.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models"));
        dirs
    }

    /// Where the release packages put the models, relative to the directory holding the
    /// executable (packaging/*):
    /// - `models/` beside it: the Windows MSI and portable zip;
    /// - `../Resources/models`: the macOS app bundle (`Contents/MacOS` → `Contents/Resources`);
    /// - `../share/pdfcraft/models`: the FHS-style trees (`bin/` → `share/`) of the Linux
    ///   deb/rpm/tar.gz/AppImage/Flatpak and the FreeBSD tarball, wherever they are installed.
    pub fn dirs_beside_exe(exe_dir: &Path) -> Vec<PathBuf> {
        vec![exe_dir.join("models"), exe_dir.join("../Resources/models"), exe_dir.join("../share/pdfcraft/models")]
    }
}

/// A recognised word: its text and its box in image pixels `[left, top, right, bottom]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub rect: [f32; 4],
}

/// A recognised line of words, in reading order.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Line {
    pub words: Vec<Word>,
}

impl Line {
    pub fn text(&self) -> String {
        self.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ")
    }
}

/// A loaded recogniser. Loading takes a moment; keep one and reuse it.
pub struct Ocr {
    engine: ocrs::OcrEngine,
}

impl Ocr {
    pub fn load(models: &Models) -> Result<Ocr, OcrError> {
        let load = |p: &Path| rten::Model::load_file(p).map_err(|e| OcrError::Load(p.display().to_string(), e.to_string()));
        let params = ocrs::OcrEngineParams {
            detection_model: Some(load(&models.detection)?),
            recognition_model: Some(load(&models.recognition)?),
            alphabet: Some(ALPHABET.into()),
            ..Default::default()
        };
        let engine = ocrs::OcrEngine::new(params).map_err(|e| OcrError::Load("ocr engine".into(), e.to_string()))?;
        Ok(Ocr { engine })
    }

    /// Load the models found by [`Models::find`].
    pub fn find() -> Result<Ocr, OcrError> {
        Self::load(&Models::find().ok_or(OcrError::NoModels)?)
    }

    /// Recognise the text in an RGBA (or RGB, or grey) image, `width` × `height` pixels.
    pub fn recognize(&self, pixels: &[u8], width: u32, height: u32) -> Result<Vec<Line>, OcrError> {
        if width == 0 || height == 0 || pixels.is_empty() {
            return Err(OcrError::EmptyImage);
        }
        let grey = despeckle(grey_page(pixels, width, height), width as usize);
        let err = |e: &dyn std::fmt::Display| OcrError::Recognize(e.to_string());
        let input = self.prepare(&grey, width, height)?;
        let words = self.find_words(&grey, &input, width, height)?;
        // Lines and their reading order are worked out once, for the whole page.
        let lines = self.engine.find_text_lines(&input, &words);
        let read = self.engine.recognize_text(&input, &lines).map_err(|e| err(&e))?;
        use ocrs::TextItem;
        Ok(read
            .into_iter()
            .flatten()
            .map(|line| Line {
                words: line
                    .words()
                    .map(|w| {
                        let r = w.bounding_rect();
                        Word { text: w.to_string(), rect: [r.left() as f32, r.top() as f32, r.right() as f32, r.bottom() as f32] }
                    })
                    .filter(|w| !w.text.trim().is_empty())
                    .collect(),
            })
            .filter(|l| !l.words.is_empty())
            .collect())
    }

    fn prepare(&self, grey: &[u8], width: u32, height: u32) -> Result<ocrs::OcrInput, OcrError> {
        let err = |e: &dyn std::fmt::Display| OcrError::Recognize(e.to_string());
        let source = ocrs::ImageSource::from_bytes(grey, (width, height)).map_err(|e| err(&e))?;
        self.engine.prepare_input(source).map_err(|e| err(&e))
    }

    /// The boxes of the page's words. The engine shrinks what it is given to one small size to find
    /// them, which loses ordinary text on a whole page, so the page is searched in overlapping
    /// bands. The whole page is searched too, for text too tall to fit in a band's overlap.
    fn find_words(&self, grey: &[u8], page: &ocrs::OcrInput, width: u32, height: u32) -> Result<Vec<RotatedRect>, OcrError> {
        let err = |e: &dyn std::fmt::Display| OcrError::Recognize(e.to_string());
        let whole = self.engine.detect_words(page).map_err(|e| err(&e))?;
        let plan = bands(height, width);
        if plan.is_empty() {
            return Ok(whole);
        }
        let tall_from = overlap(height) as f32 * 0.6;
        let mut words: Vec<RotatedRect> = whole.into_iter().filter(|w| w.height() >= tall_from).collect();
        let tall = words.len();
        // The engine pads its boxes by a fixed amount at the size it searches. That is less of a
        // band's word than of a whole page's, and a box too tight clips symbols such as €.
        let margin = height as f32 / 1000.0;
        let stride = width as usize;
        for band in plan {
            let Some(data) = grey.get(band.top as usize * stride..band.bottom as usize * stride) else { continue };
            let input = self.prepare(data, width, band.bottom - band.top)?;
            for word in self.engine.detect_words(&input).map_err(|e| err(&e))? {
                let c = word.center();
                let (x, y) = (c.x, c.y + band.top as f32);
                if word.height() >= tall_from || !band.keeps(y) || words.iter().take(tall).any(|t| contains(t, x, y)) {
                    continue;
                }
                // A word on the edge between two bands can be found by both.
                if band.near_edge(y) && words.iter().any(|t| contains(t, x, y)) {
                    continue;
                }
                words.push(RotatedRect::new(PointF::from_yx(y, x), word.up_axis(), word.width() + margin, word.height()));
            }
        }
        Ok(words)
    }
}

/// Whether the point is inside the box, taken as upright (words on a page are close to it).
fn contains(word: &RotatedRect, x: f32, y: f32) -> bool {
    let c = word.center();
    (x - c.x).abs() <= word.width() / 2.0 && (y - c.y).abs() <= word.height() / 2.0
}

/// A page shorter than this is read whole: it isn't shrunk enough to lose text.
const MIN_BANDED_HEIGHT: u32 = 1200;

/// How many pixels neighbouring bands share. A line shorter than half of this is whole in the band
/// that keeps it.
fn overlap(height: u32) -> u32 {
    height / 12
}

/// A horizontal slice of the page: the rows read, and the rows whose lines it keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Band {
    top: u32,
    bottom: u32,
    own_top: u32,
    own_bottom: u32,
}

/// The bands of a page, about four on a portrait sheet and fewer on a landscape one. The rows each
/// band keeps cover the page exactly once.
fn bands(height: u32, width: u32) -> Vec<Band> {
    if height < MIN_BANDED_HEIGHT || width == 0 {
        return Vec::new();
    }
    let count = ((height as f32 / width as f32 * 3.0).round() as u32).clamp(2, 8);
    let half = overlap(height) / 2;
    (0..count)
        .map(|i| {
            let own_top = height / count * i;
            let own_bottom = if i + 1 == count { height } else { height / count * (i + 1) };
            Band { top: own_top.saturating_sub(half), bottom: own_bottom.saturating_add(half).min(height), own_top, own_bottom }
        })
        .collect()
}

/// How far past its own rows a band still keeps a word, so that one found slightly differently by
/// two bands is not missed by both.
const EDGE_TOLERANCE: f32 = 8.0;

impl Band {
    fn keeps(&self, y: f32) -> bool {
        y >= self.own_top as f32 - EDGE_TOLERANCE && y <= self.own_bottom as f32 + EDGE_TOLERANCE
    }

    fn near_edge(&self, y: f32) -> bool {
        (y - self.own_top as f32).abs() <= EDGE_TOLERANCE * 2.0 || (y - self.own_bottom as f32).abs() <= EDGE_TOLERANCE * 2.0
    }
}

/// `pixels` as one grey byte a pixel: RGBA and RGB by the usual luminance weights, anything else
/// as given.
fn grey_page(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    let luma = |r: u8, g: u8, b: u8| ((u32::from(r) * 299 + u32::from(g) * 587 + u32::from(b) * 114) / 1000) as u8;
    let area = (width as usize).saturating_mul(height as usize);
    if pixels.len() == area.saturating_mul(4) {
        pixels.as_chunks::<4>().0.iter().map(|&[r, g, b, _]| luma(r, g, b)).collect()
    } else if pixels.len() == area.saturating_mul(3) {
        pixels.as_chunks::<3>().0.iter().map(|&[r, g, b]| luma(r, g, b)).collect()
    } else {
        pixels.to_vec()
    }
}

/// A pixel this much darker or lighter than all eight of its neighbours is a speck.
const SPECK_CONTRAST: i16 = 48;
/// The share of a page that must be specks (1 in this many pixels) for it to count as speckled.
const SPECKLED: usize = 1000;

/// Remove scanner specks: each pixel becomes the middle value of its 3x3 neighbourhood. A page
/// without specks is left alone, because the same filter wears away the thin strokes of small or
/// low-resolution text. This changes only what is read, not the page's image.
fn despeckle(grey: Vec<u8>, width: usize) -> Vec<u8> {
    if width < 3 {
        return grey;
    }
    let mut out = grey.clone();
    let mut specks = 0usize;
    let rows: Vec<&[u8]> = grey.chunks_exact(width).collect();
    for (row, near) in out.chunks_exact_mut(width).skip(1).zip(rows.windows(3)) {
        let &[a, b, c] = near else { continue };
        for (px, ((x, y), z)) in row.iter_mut().skip(1).zip(a.windows(3).zip(b.windows(3)).zip(c.windows(3))) {
            let mut v = [0u8; 9];
            for (slot, value) in v.iter_mut().zip(x.iter().chain(y).chain(z)) {
                *slot = *value;
            }
            let centre = i16::from(v[4]);
            let around = v.iter().enumerate().filter(|(i, _)| *i != 4).map(|(_, n)| i16::from(*n));
            let (low, high) = around.fold((i16::MAX, i16::MIN), |(l, h), n| (l.min(n), h.max(n)));
            if centre + SPECK_CONTRAST < low || centre > high + SPECK_CONTRAST {
                specks += 1;
            }
            let (_, middle, _) = v.select_nth_unstable(4);
            *px = *middle;
        }
    }
    if specks.saturating_mul(SPECKLED) > grey.len() { out } else { grey }
}

/// A word placed on a page, in PDF user space: the bottom-left corner of its box, and the
/// vectors along its bottom edge (`across`, left → right as read) and its left edge (`up`).
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedWord {
    pub text: String,
    pub origin: [f64; 2],
    pub across: [f64; 2],
    pub up: [f64; 2],
}

impl PlacedWord {
    /// Place a pixel-space word with `to_user`, which maps an image pixel (x right, y down) to
    /// user space.
    pub fn place(word: &Word, to_user: impl Fn(f32, f32) -> [f64; 2]) -> PlacedWord {
        let [l, t, r, b] = word.rect;
        let o = to_user(l, b);
        let br = to_user(r, b);
        let tl = to_user(l, t);
        PlacedWord { text: word.text.clone(), origin: o, across: [br[0] - o[0], br[1] - o[1]], up: [tl[0] - o[0], tl[1] - o[1]] }
    }
}

/// Helvetica's ascender and descender (em fractions): a word box spans about this much.
const BOX_EM: f64 = 0.93;
const DESCENT: f64 = 0.21;
/// Helvetica's cap height and x-height (em fractions).
const CAP_HEIGHT: f64 = 0.718;
const X_HEIGHT: f64 = 0.523;

/// How far `text`'s ink reaches above and below the baseline, in em fractions: a word box is
/// tight around the ink, so "DT" spans the cap height only, "sun" the x-height, and "gap" reaches
/// down to the descender. Text without letters or digits keeps the full ascender-descender span.
fn ink_extent(text: &str) -> (f64, f64) {
    if !text.chars().any(char::is_alphanumeric) {
        return (BOX_EM - DESCENT, -DESCENT);
    }
    let tall = text
        .chars()
        .any(|c| c.is_uppercase() || c.is_ascii_digit() || "bdfhijklt!?/\\()[]{}|'\"#$%&@".contains(c) || (c.is_alphabetic() && !c.is_ascii()));
    let low = text.chars().any(|c| "gjpqy()[]{}|_".contains(c));
    (if tall { CAP_HEIGHT } else { X_HEIGHT }, if low { -DESCENT } else { 0.0 })
}

fn num(v: f64) -> String {
    let s = format!("{:.4}", if v.abs() < 1e-9 { 0.0 } else { v });
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".into() } else { s.into() }
}

/// Page content that writes `words` as invisible text in `/PCHelv` (standard Helvetica,
/// WinAnsiEncoding), each word stretched to its box so selection and search highlight the
/// right place. Marked content `/OCR` so it can be told apart from the page's own text.
pub fn text_layer(words: &[PlacedWord]) -> Vec<u8> {
    // `3 Tr` and the font are graphics state, which `ET` doesn't reset: without `q … Q` they
    // would stay set for every later content stream on the page, so text added after OCR (a
    // header or footer) drew invisibly (#3029).
    let mut out = b"/OCR BMC\nq\nBT\n3 Tr\n/PCHelv 1 Tf\n".to_vec();
    for w in words {
        let width = helvetica_width(&w.text, 1.0);
        let height = w.up[0].hypot(w.up[1]);
        if width <= 0.0 || height <= 0.0 {
            continue;
        }
        // The em square: the word's own ink (cap height or x-height, descenders or not) spans the
        // box, so the text is as tall as the scanned word; its width is stretched to fit the word.
        let (top, bottom) = ink_extent(&w.text);
        let em = [w.up[0] / (top - bottom), w.up[1] / (top - bottom)];
        let ax = [w.across[0] / width, w.across[1] / width];
        let o = [w.origin[0] - em[0] * bottom, w.origin[1] - em[1] * bottom];
        let m = [ax[0], ax[1], em[0], em[1], o[0], o[1]];
        let nums: Vec<String> = m.iter().map(|v| num(*v)).collect();
        out.extend_from_slice(nums.join(" ").as_bytes());
        out.extend_from_slice(b" Tm ");
        out.extend(pdfcraft_fonts::literal(&pdfcraft_fonts::win_ansi(&w.text)));
        out.extend_from_slice(b" Tj\n");
    }
    out.extend_from_slice(b"ET\nQ\nEMC\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn places_words_through_the_mapping() {
        // 2 pixels per point, page 612 × 792, y flipped.
        let to_user = |x: f32, y: f32| [x as f64 / 2.0, 792.0 - y as f64 / 2.0];
        let w = PlacedWord::place(&Word { text: "Hello".into(), rect: [100.0, 200.0, 300.0, 240.0] }, to_user);
        assert_eq!(w.origin, [50.0, 672.0]);
        assert_eq!(w.across, [100.0, 0.0]);
        assert_eq!(w.up, [0.0, 20.0]);
    }

    /// The text layer is Helvetica in WinAnsiEncoding: every character the model can recognise
    /// must be one it can hold, or recognised text would be saved as `?`. A model with a wider
    /// alphabet (Chinese, for example) needs a text layer in another font first.
    #[test]
    fn the_model_alphabet_fits_the_text_layer_font() {
        assert_eq!(pdfcraft_fonts::first_undrawable(ALPHABET), None);
    }

    #[test]
    fn text_layer_is_invisible_text_fitted_to_each_box() {
        let w = PlacedWord { text: "Hi".into(), origin: [10.0, 20.0], across: [30.0, 0.0], up: [0.0, 9.3] };
        let s = String::from_utf8(text_layer(&[w])).unwrap();
        assert!(s.contains("3 Tr") && s.contains("/PCHelv 1 Tf"), "{s}");
        let width = helvetica_width("Hi", 1.0);
        // "Hi" has no descender: its box spans the cap height, from the baseline up.
        assert!(s.contains(&format!("{} 0 0 {} 10 20 Tm (Hi) Tj", num(30.0 / width), num(9.3 / CAP_HEIGHT))), "{s}");
        assert!(s.starts_with("/OCR BMC") && s.ends_with("EMC\n"));
    }

    /// #3029: the layer's invisible rendering mode and font end with the layer. Every operator
    /// that sets them sits inside a `q … Q` pair that the layer closes itself, so content drawn
    /// after it on the page starts from the graphics state it had before.
    #[test]
    fn the_layer_restores_the_graphics_state_it_changes() {
        let w = PlacedWord { text: "Hi".into(), origin: [10.0, 20.0], across: [30.0, 0.0], up: [0.0, 9.3] };
        let s = String::from_utf8(text_layer(&[w])).unwrap();
        let mut depth = 0_i32;
        for token in s.split_ascii_whitespace() {
            match token {
                "q" => depth += 1,
                "Q" => depth -= 1,
                "Tr" | "Tf" | "Tm" => assert!(depth > 0, "`{token}` outside q … Q in {s:?}"),
                _ => {}
            }
            assert!(depth >= 0, "unbalanced Q in {s:?}");
        }
        assert_eq!(depth, 0, "the layer leaves a q open: {s:?}");
    }

    /// A word box is tight around the ink, so the text's size and baseline come from the letters:
    /// capitals, lowercase without ascenders, descenders.
    #[test]
    fn the_text_is_as_tall_as_the_scanned_word() {
        assert_eq!(ink_extent("DT"), (CAP_HEIGHT, 0.0));
        assert_eq!(ink_extent("402337"), (CAP_HEIGHT, 0.0));
        assert_eq!(ink_extent("sun"), (X_HEIGHT, 0.0));
        assert_eq!(ink_extent("gap"), (X_HEIGHT, -DESCENT));
        assert_eq!(ink_extent("Gallus.Ze@camtel.cm"), (CAP_HEIGHT, 0.0));
        assert_eq!(ink_extent("Corporation"), (CAP_HEIGHT, -DESCENT));
        assert_eq!(ink_extent("--"), (BOX_EM - DESCENT, -DESCENT));
        let w = PlacedWord { text: "gap".into(), origin: [0.0, 100.0], across: [20.0, 0.0], up: [0.0, 7.3] };
        let s = String::from_utf8(text_layer(&[w])).unwrap();
        let em = 7.3 / (X_HEIGHT + DESCENT);
        assert!(s.contains(&format!(" 0 0 {} 0 {} Tm (gap) Tj", num(em), num(100.0 + em * DESCENT))), "{s}");
    }

    #[test]
    fn rotated_pages_keep_the_reading_direction() {
        // A page turned 90°: image x runs up the page.
        let to_user = |x: f32, y: f32| [y as f64, x as f64];
        let w = PlacedWord::place(&Word { text: "Up".into(), rect: [0.0, 0.0, 50.0, 10.0] }, to_user);
        assert_eq!(w.across, [0.0, 50.0]);
        assert_eq!(w.up, [-10.0, 0.0]);
    }

    /// The current ONNX pair wins when an installation also retains both legacy models.
    #[test]
    fn model_discovery_prefers_the_complete_onnx_pair() {
        let dir = tempfile::tempdir().unwrap();
        for file in ["text-detection.onnx", "text-recognition.onnx"] {
            std::fs::write(dir.path().join(file), b"model").unwrap();
        }
        let expected = Some(Models { detection: dir.path().join("text-detection.onnx"), recognition: dir.path().join("text-recognition.onnx") });
        assert_eq!(Models::in_dir(dir.path()), expected, "a complete ONNX pair is sufficient");
        for file in ["text-detection.rten", "text-recognition.rten"] {
            std::fs::write(dir.path().join(file), b"model").unwrap();
        }
        assert_eq!(Models::in_dir(dir.path()), expected, "the complete ONNX pair takes precedence over legacy models");
    }

    /// An interrupted ONNX update still permits an intact legacy pair in the same directory.
    #[test]
    fn model_discovery_falls_back_to_legacy_when_the_onnx_pair_is_incomplete() {
        for current in ["text-detection.onnx", "text-recognition.onnx"] {
            let dir = tempfile::tempdir().unwrap();
            for file in [current, "text-detection.rten", "text-recognition.rten"] {
                std::fs::write(dir.path().join(file), b"model").unwrap();
            }
            assert_eq!(
                Models::in_dir(dir.path()),
                Some(Models { detection: dir.path().join("text-detection.rten"), recognition: dir.path().join("text-recognition.rten") }),
                "{current} alone must not hide a complete legacy pair"
            );
        }
    }

    /// Detection and recognition must come from one complete format pair, never a mixed update.
    #[test]
    fn model_discovery_refuses_mixed_format_pairs() {
        for pair in [["text-detection.onnx", "text-recognition.rten"], ["text-detection.rten", "text-recognition.onnx"]] {
            let dir = tempfile::tempdir().unwrap();
            for file in pair {
                std::fs::write(dir.path().join(file), b"model").unwrap();
            }
            assert_eq!(Models::in_dir(dir.path()), None, "{pair:?} is not a complete model pair");
        }
    }

    /// Existing source builds and PDFCRAFT_MODELS overrides retain their complete RTEN pair.
    #[test]
    fn model_discovery_keeps_legacy_only_installations() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(Models::in_dir(dir.path()), None);
        std::fs::write(dir.path().join("text-detection.rten"), b"model").unwrap();
        assert_eq!(Models::in_dir(dir.path()), None, "one legacy model is incomplete");
        std::fs::write(dir.path().join("text-recognition.rten"), b"model").unwrap();
        assert_eq!(
            Models::in_dir(dir.path()),
            Some(Models { detection: dir.path().join("text-detection.rten"), recognition: dir.path().join("text-recognition.rten") })
        );
    }

    /// #103: every release package's layout puts the models where the installed app looks.
    /// Mirrors packaging/{windows/package.ps1,macos/package.sh,linux/package.sh,freebsd/package.sh}.
    #[test]
    fn finds_the_models_in_every_release_package_layout() {
        let base = std::env::temp_dir().join(format!("pdfcraft-ocr-layouts-{}", std::process::id()));
        let layouts = [
            ("windows", "Loom", "Loom/models"),
            ("macos", "Loom.app/Contents/MacOS", "Loom.app/Contents/Resources/models"),
            ("linux-deb-rpm", "usr/bin", "usr/share/pdfcraft/models"),
            ("linux-tar", "pdfcraft-1.0.0-linux-x86_64/bin", "pdfcraft-1.0.0-linux-x86_64/share/pdfcraft/models"),
            ("appimage", "Loom.AppDir/usr/bin", "Loom.AppDir/usr/share/pdfcraft/models"),
            ("flatpak", "app/bin", "app/share/pdfcraft/models"),
            ("freebsd", "usr/local/bin", "usr/local/share/pdfcraft/models"),
        ];
        for (name, exe_dir, models_dir) in layouts {
            let root = base.join(name);
            let (exe_dir, models_dir) = (root.join(exe_dir), root.join(models_dir));
            std::fs::create_dir_all(&exe_dir).unwrap();
            std::fs::create_dir_all(&models_dir).unwrap();
            let found = || Models::dirs_beside_exe(&exe_dir).iter().find_map(|d| Models::in_dir(d));
            assert_eq!(found(), None, "{name}: no models yet");
            for file in [DETECTION_MODEL, RECOGNITION_MODEL] {
                std::fs::write(models_dir.join(file), b"model").unwrap();
            }
            let models = found().unwrap_or_else(|| panic!("{name}: models in {} not found", models_dir.display()));
            assert_eq!(models.detection.canonicalize().unwrap(), models_dir.join(DETECTION_MODEL).canonicalize().unwrap(), "{name}");
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The message tells end users what to do; release builds have no cargo.
    #[test]
    fn the_alphabet_is_the_trained_one() {
        let chars: Vec<char> = ALPHABET.chars().collect();
        assert_eq!(chars.len(), 96);
        // The euro sign follows the ASCII punctuation (class 44; 0 is the CTC blank), not a second E.
        assert_eq!(chars.iter().position(|&c| c == '€'), Some(43));
        assert_eq!(chars.iter().filter(|&&c| c == 'E').count(), 1);
    }

    #[test]
    fn every_row_of_a_page_belongs_to_exactly_one_band() {
        for (h, w) in [(3300, 2550), (2550, 3300), (1200, 900), (9000, 300), (u32::MAX / 2, 1)] {
            let plan = bands(h, w);
            assert!(plan.len() >= 2 && plan.len() <= 8, "{h}x{w}: {}", plan.len());
            assert_eq!(plan.first().map(|b| b.own_top), Some(0));
            assert_eq!(plan.last().map(|b| b.own_bottom), Some(h));
            for pair in plan.windows(2) {
                assert_eq!(pair[0].own_bottom, pair[1].own_top, "{h}x{w}");
            }
            for b in &plan {
                assert!(b.top <= b.own_top && b.own_bottom <= b.bottom && b.bottom <= h, "{b:?}");
            }
        }
    }

    #[test]
    fn a_word_on_the_edge_between_bands_is_kept_by_both_and_flagged() {
        let plan = bands(3300, 2550);
        let edge = plan[0].own_bottom as f32;
        assert!(plan[0].keeps(edge - 1.0) && plan[0].keeps(edge + 1.0));
        assert!(plan[1].keeps(edge - 1.0) && plan[1].keeps(edge + 1.0));
        assert!(plan[1].near_edge(edge) && !plan[1].near_edge(edge + 200.0));
        assert!(!plan[0].keeps(edge + 100.0), "well inside the next band is that band's");
    }

    #[test]
    fn contains_is_the_upright_box() {
        let word = RotatedRect::from_rect(rten_imageproc::RectF::from_tlbr(10.0, 20.0, 30.0, 60.0));
        assert!(contains(&word, 40.0, 20.0));
        assert!(!contains(&word, 61.0, 20.0) && !contains(&word, 40.0, 31.0));
    }

    #[test]
    fn small_pages_and_empty_widths_are_read_whole() {
        assert!(bands(800, 600).is_empty());
        assert!(bands(3300, 0).is_empty());
    }

    #[test]
    fn despeckle_removes_a_lone_dot_and_keeps_a_stroke() {
        let mut page = vec![255u8; 25];
        page[12] = 0;
        assert!(despeckle(page, 5).iter().all(|&v| v == 255), "a speck goes");
        let mut bar = vec![255u8; 25];
        for x in 0..5 {
            bar[5 + x] = 0;
            bar[10 + x] = 0;
            bar[15 + x] = 0;
        }
        assert_eq!(despeckle(bar, 5)[12], 0, "a stroke stays");
        // A thin line on an otherwise clean page: a median filter would erase it.
        let mut line = vec![255u8; 2500];
        for row in line.chunks_exact_mut(50) {
            row[25] = 0;
        }
        assert_eq!(despeckle(line.clone(), 50), line, "a page without specks is left alone");
        assert_eq!(despeckle(vec![1, 2, 3], 1), [1, 2, 3], "too narrow to have neighbours");
        assert_eq!(despeckle(vec![1, 2, 3, 4, 5], 3).len(), 5, "a short last row keeps its length");
    }

    #[test]
    fn grey_page_reads_each_pixel_format() {
        assert_eq!(grey_page(&[255, 255, 255, 255, 0, 0, 0, 255], 2, 1), [255, 0]);
        assert_eq!(grey_page(&[255, 255, 255, 0, 0, 0], 2, 1), [255, 0]);
        assert_eq!(grey_page(&[7, 9], 2, 1), [7, 9]);
    }

    #[test]
    fn missing_models_message_is_for_end_users() {
        let message = OcrError::NoModels.to_string();
        assert!(!message.contains("cargo") && message.contains("PDFCRAFT_MODELS"), "{message}");
    }

    /// End to end on rendered text, when the models are installed (`cargo xtask models`).
    #[test]
    fn reads_rendered_text() {
        let Some(models) = Models::find() else {
            eprintln!("skipped: OCR models not installed");
            return;
        };
        let ocr = Ocr::load(&models).unwrap();
        let (w, h, px) = test_image::hello();
        let lines = ocr.recognize(&px, w, h).unwrap();
        let text: Vec<String> = lines.iter().map(Line::text).collect();
        assert!(text.iter().any(|t| t.contains("HELL") && t.contains("WOR")), "{text:?}");
        let word = lines.iter().flat_map(|l| &l.words).find(|w| w.text.contains("HELL")).unwrap();
        assert!(word.rect[0] >= 10.0 && word.rect[0] < 60.0, "{word:?}");
    }

    mod test_image {
        /// "HELLO WORLD" drawn with blocky 5×7 letters, 4 pixels per dot, black on white.
        pub fn hello() -> (u32, u32, Vec<u8>) {
            const GLYPHS: &[(char, [&str; 7])] = &[
                ('H', ["X...X", "X...X", "X...X", "XXXXX", "X...X", "X...X", "X...X"]),
                ('E', ["XXXXX", "X....", "X....", "XXXX.", "X....", "X....", "XXXXX"]),
                ('L', ["X....", "X....", "X....", "X....", "X....", "X....", "XXXXX"]),
                ('O', [".XXX.", "X...X", "X...X", "X...X", "X...X", "X...X", ".XXX."]),
                ('W', ["X...X", "X...X", "X...X", "X.X.X", "X.X.X", "XX.XX", "X...X"]),
                ('R', ["XXXX.", "X...X", "X...X", "XXXX.", "X.X..", "X..X.", "X...X"]),
                ('D', ["XXXX.", "X...X", "X...X", "X...X", "X...X", "X...X", "XXXX."]),
            ];
            let text = "HELLO WORLD";
            let (dot, x0, y0) = (4u32, 40u32, 40u32);
            let (w, h) = (x0 * 2 + text.len() as u32 * 6 * dot, y0 * 2 + 7 * dot);
            let mut px = vec![255u8; (w * h * 4) as usize];
            for (i, c) in text.chars().enumerate() {
                let Some((_, rows)) = GLYPHS.iter().find(|g| g.0 == c) else { continue };
                for (ry, row) in rows.iter().enumerate() {
                    for (rx, b) in row.bytes().enumerate() {
                        if b != b'X' {
                            continue;
                        }
                        for dy in 0..dot {
                            for dx in 0..dot {
                                let x = x0 + (i as u32 * 6 + rx as u32) * dot + dx;
                                let y = y0 + ry as u32 * dot + dy;
                                let o = ((y * w + x) * 4) as usize;
                                px[o..o + 3].copy_from_slice(&[0, 0, 0]);
                            }
                        }
                    }
                }
            }
            (w, h, px)
        }
    }
}
