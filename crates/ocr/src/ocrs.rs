//! The pure-Rust engine: the ocrs crate with its pre-trained models, wrapped as a
//! [`Recognizer`] (`OcrsRecognizer`). Reading is offline and local; the models do not report a
//! confidence, so its words have `confidence: None`.

use std::path::{Path, PathBuf};

use rten_imageproc::{PointF, RotatedRect};

use super::{Line, MAX_PIXELS, MAX_SIDE, OcrError, OcrImage, RecognizeOptions, Recognizer, Word};

/// The validated byte length of a `width` × `height` RGBA raster, with the same caps
/// [`OcrImage::new`] applies: zero sides are empty, sides over [`MAX_SIDE`] or totals over
/// [`MAX_PIXELS`] are too large. `Ocr::recognize` takes raw dimensions (they come from
/// callers, ultimately from documents, so they are untrusted), and the naive
/// `width * height * 4` overflows `u32` for absurd values — the caps are checked first, in
/// `usize`, so nothing overflows and nothing huge is ever touched.
fn checked_len(width: u32, height: u32) -> Result<usize, OcrError> {
    if width == 0 || height == 0 {
        return Err(OcrError::EmptyImage);
    }
    if width > MAX_SIDE || height > MAX_SIDE {
        return Err(OcrError::ImageTooLarge);
    }
    let pixels = (width as usize).checked_mul(height as usize).ok_or(OcrError::ImageTooLarge)?;
    if pixels > MAX_PIXELS as usize {
        return Err(OcrError::ImageTooLarge);
    }
    pixels.checked_mul(4).ok_or(OcrError::ImageTooLarge)
}

/// The languages the models read (ISO 639-1); all use the Latin alphabet without accents.
pub const LANGUAGES: &[(&str, &str)] = &[("en", "English")];

/// The model files, as named in ATTRIBUTION.toml.
pub const DETECTION_MODEL: &str = "text-detection.onnx";
pub const RECOGNITION_MODEL: &str = "text-recognition.onnx";

pub const ALPHABET: &str = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~€ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Where the two model files are.
#[derive(Clone, Debug, PartialEq)]
pub struct Models {
    pub detection: PathBuf,
    pub recognition: PathBuf,
}

impl Models {
    pub fn dirs_beside_exe(exe_dir: &Path) -> Vec<PathBuf> {
        vec![exe_dir.join("models"), exe_dir.join("../Resources/models"), exe_dir.join("../share/pdfcraft/models")]
    }

    /// The models in `dir`, preferring current ONNX files over legacy RTEN files. An incomplete
    /// update never mixes formats or hides an intact legacy pair.
    pub fn in_dir(dir: &Path) -> Option<Models> {
        [(DETECTION_MODEL, RECOGNITION_MODEL), ("text-detection.rten", "text-recognition.rten")].into_iter().find_map(|(detection, recognition)| {
            let m = Models { detection: dir.join(detection), recognition: dir.join(recognition) };
            (m.detection.is_file() && m.recognition.is_file()).then_some(m)
        })
    }

    /// Look for the models: `$PDFCRAFT_MODELS`, then `models/` beside the executable (and
    /// `Resources/models` in a macOS bundle), then the source tree's `assets/models/`.
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
        if pixels.is_empty() {
            return Err(OcrError::EmptyImage);
        }
        checked_len(width, height)?;
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
                        Word::new(w.to_string(), [r.left() as f32, r.top() as f32, r.right() as f32, r.bottom() as f32], "ocrs")
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

/// The ocrs engine as a [`Recognizer`] (the default engine).
pub struct OcrsRecognizer {
    engine: Ocr,
}

impl OcrsRecognizer {
    /// Load the models from `models`.
    pub fn load(models: &Models) -> Result<OcrsRecognizer, OcrError> {
        Ok(OcrsRecognizer { engine: Ocr::load(models)? })
    }

    /// Load the models found by [`Models::find`].
    pub fn find() -> Result<OcrsRecognizer, OcrError> {
        Self::load(&Models::find().ok_or(OcrError::NoModels)?)
    }
}

impl Recognizer for OcrsRecognizer {
    fn id(&self) -> &'static str {
        "ocrs"
    }

    fn available(&self) -> bool {
        Models::find().is_some()
    }

    fn languages(&self) -> Vec<(String, String)> {
        LANGUAGES.iter().map(|(c, n)| ((*c).to_string(), (*n).to_string())).collect()
    }

    fn recognize(&self, image: &OcrImage, _options: &RecognizeOptions) -> Result<Vec<Line>, OcrError> {
        self.engine.recognize(image.rgba(), image.width(), image.height())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Raw dimensions are untrusted (they come from callers, ultimately from documents): zero
    /// sides, sides or pixel totals over the caps, and products that would overflow the naive
    /// `width * height * 4` are errors, never panics, before anything is touched.
    #[test]
    fn raw_dimensions_are_capped_and_checked_not_panicked_at() {
        assert!(matches!(checked_len(0, 10), Err(OcrError::EmptyImage)));
        assert!(matches!(checked_len(10, 0), Err(OcrError::EmptyImage)));
        assert!(matches!(checked_len(MAX_SIDE + 1, 1), Err(OcrError::ImageTooLarge)));
        // 65536² would wrap u32 in the old `width * height * 4`; the side cap refuses it first.
        assert!(matches!(checked_len(u32::MAX, u32::MAX), Err(OcrError::ImageTooLarge)));
        // Legal sides whose product is over the 64 MP area cap.
        assert!(matches!(checked_len(MAX_SIDE, 7000), Err(OcrError::ImageTooLarge)));
        assert_eq!(checked_len(2, 2).unwrap(), 16);
        // Exactly the area cap fits: 8000 × 8000 = 64 MP.
        assert_eq!(checked_len(8000, 8000).unwrap(), 8000usize * 8000 * 4);
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
}
