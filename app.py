import os
import time
import cv2
import joblib
import numpy as np
from scipy.stats import skew
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import warnings

warnings.filterwarnings('ignore')


# ── Startup validation ────────────────────────────────────────────────────────

def _load_artifact(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"STARTUP FAILED: {name} not found at:\n  {path}\n"
            f"Download from Kaggle: /kaggle/working/models/{os.path.basename(path)}\n"
            f"{'='*60}"
        )
    return joblib.load(path)


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
model  = _load_artifact(os.path.join(BASE_DIR, "models", "mlp_model.joblib"), "MLP model")
scaler = _load_artifact(os.path.join(BASE_DIR, "models", "scaler.joblib"),    "Scaler")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB   = 5

FEATURE_COLS = [
    "Kurtosis", "Skewness", "Standard Deviation", "Range", "Median", "Geometric Mean",
    "Hjorth Mobility", "Hjorth Complexity",
    "Entropy", "Even-Odd Ratio", "Chi-Square Statistic", "Energy",
    "Pixel Diff Mean", "Pixel Diff Std", "Pixel Diff Max", "Pixel Diff Median",
    "Pairwise Hist Diff Mean", "Pairwise Hist Diff Std", "Pairwise Hist Diff Max",
    "SPAM_horizontal_mean",    "SPAM_horizontal_std",    "SPAM_horizontal_median",
    "SPAM_horizontal_skew",    "SPAM_horizontal_entropy",
    "SPAM_vertical_mean",      "SPAM_vertical_std",      "SPAM_vertical_median",
    "SPAM_vertical_skew",      "SPAM_vertical_entropy",
    "SPAM_diagonal_pd_mean",   "SPAM_diagonal_pd_std",   "SPAM_diagonal_pd_median",
    "SPAM_diagonal_pd_skew",   "SPAM_diagonal_pd_entropy",
    "SPAM_diagonal_nd_mean",   "SPAM_diagonal_nd_std",   "SPAM_diagonal_nd_median",
    "SPAM_diagonal_nd_skew",   "SPAM_diagonal_nd_entropy",
]  # 39 features — Adjacent Bin Difference and Pairwise Hist Diff Sum dropped


# ── Feature extraction — mirrors training code exactly ────────────────────────

def _extract_statistical_features(hist: np.ndarray):
    hist = hist.astype(np.float64)
    hist_normalized = hist / (hist.sum() + 1e-8)
    intensity_bins  = np.arange(256, dtype=np.float64)

    mean     = float(np.sum(intensity_bins * hist_normalized))
    variance = float(np.sum(((intensity_bins - mean) ** 2) * hist_normalized))
    std_dev  = float(np.sqrt(variance)) if variance > 0 else 0.0

    if std_dev > 0:
        skewness = float(np.sum((((intensity_bins - mean) / std_dev) ** 3) * hist_normalized))
        kurt     = float(np.sum((((intensity_bins - mean) / std_dev) ** 4) * hist_normalized)) - 3
    else:
        skewness = 0.0
        kurt     = 0.0

    cumsum     = np.cumsum(hist_normalized)
    median_idx = np.where(cumsum >= 0.5)[0]
    median     = float(intensity_bins[median_idx[0]] if len(median_idx) > 0 else 0)

    hist_nz   = hist[hist > 0]
    geo_mean  = float(np.exp(np.mean(np.log(hist_nz + 1e-8)))) if len(hist_nz) > 0 else 0.0
    pix_range = float(np.max(hist) - np.min(hist))

    return kurt, skewness, std_dev, pix_range, median, geo_mean


def _extract_hjorth_features(signal: np.ndarray):
    signal     = signal.astype(np.float64)
    activity   = float(np.var(signal))
    d1         = np.diff(signal)
    mobility   = float(np.sqrt(np.var(d1) / (activity + 1e-8)))
    d2         = np.diff(d1)
    complexity = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-8)) / (mobility + 1e-8))
    return mobility, complexity


def _extract_entropy_chi_square(hist: np.ndarray):
    hist       = hist.astype(np.float64)
    hist_norm  = hist / (hist.sum() + 1e-8)
    entropy    = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-8)))
    uniform    = np.ones(256) / 256.0
    chi_square = float(np.sum((hist_norm - uniform) ** 2 / (uniform + 1e-8)))
    energy     = float(np.sum(hist_norm ** 2))
    return entropy, chi_square, energy


def _extract_even_odd_ratio(hist: np.ndarray) -> float:
    return float(hist[::2].sum() / (hist[1::2].sum() + 1e-8))


def _extract_pixel_differences(img: np.ndarray):
    img_f     = img.astype(np.float64)
    diff_h    = np.abs(np.diff(img_f, axis=1))
    diff_v    = np.abs(np.diff(img_f, axis=0))
    diff_dp   = np.abs(img_f[:-1, :-1] - img_f[1:, 1:])
    diff_dn   = np.abs(img_f[:-1, 1:]  - img_f[1:, :-1])
    all_diffs = np.concatenate([diff_h.ravel(), diff_v.ravel(), diff_dp.ravel(), diff_dn.ravel()])
    return (
        float(np.mean(all_diffs)),
        float(np.std(all_diffs)),
        float(np.max(all_diffs)),
        float(np.median(all_diffs)),
    )


def _extract_pairwise_histogram_diff(hist: np.ndarray):
    pw = np.abs(hist[:-1] - hist[1:])
    return float(np.mean(pw)), float(np.std(pw)), float(np.max(pw))


def _extract_spam_features(img: np.ndarray):
    img_f = img.astype(np.float64)

    def _spam_dir(mat: np.ndarray):
        flat        = mat.ravel()
        mean        = float(np.mean(flat))
        std         = float(np.std(flat))
        median      = float(np.median(flat))
        skewness    = float(skew(flat))
        hist_s, _   = np.histogram(flat, bins=256, range=(0, 256))
        hist_s      = hist_s.astype(np.float64)
        hist_s_norm = hist_s / (hist_s.sum() + 1e-8)
        entropy     = float(-np.sum(hist_s_norm * np.log2(hist_s_norm + 1e-8)))
        return mean, std, median, skewness, entropy

    spam_h  = np.abs(np.diff(img_f, axis=1))
    spam_v  = np.abs(np.diff(img_f, axis=0))
    spam_dp = np.abs(img_f[:-1, :-1] - img_f[1:, 1:])
    spam_dn = np.abs(img_f[:-1, 1:]  - img_f[1:, :-1])

    return (
        *_spam_dir(spam_h),
        *_spam_dir(spam_v),
        *_spam_dir(spam_dp),
        *_spam_dir(spam_dn),
    )


def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Input : 256×256 uint8 grayscale numpy array
    Output: (1, 39) float32 — matches training feature order exactly
    """
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

    kurt, skewness, std_dev, pix_range, median, geo_mean = _extract_statistical_features(hist)
    mobility, complexity                                  = _extract_hjorth_features(hist)
    entropy, chi_sq, energy                               = _extract_entropy_chi_square(hist)
    even_odd                                              = _extract_even_odd_ratio(hist)
    pd_mean, pd_std, pd_max, pd_median                   = _extract_pixel_differences(img)
    pw_mean, pw_std, pw_max                               = _extract_pairwise_histogram_diff(hist)
    spam                                                  = _extract_spam_features(img)

    values = [
        kurt, skewness, std_dev, pix_range, median, geo_mean,
        mobility, complexity,
        entropy, even_odd, chi_sq, energy,
        pd_mean, pd_std, pd_max, pd_median,
        pw_mean, pw_std, pw_max,
        *spam,
    ]

    assert len(values) == 39, f"Feature count mismatch: got {len(values)}, expected 39"
    return np.array([values], dtype=np.float32)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/lsb")
def lsb():
    return render_template("lsb.html")

@app.route("/dct")
def dct():
    return render_template("dct.html")

@app.route("/pvd")
def pvd():
    return render_template("pvd.html")

@app.route("/test")
def test():
    return render_template("test.html")


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.perf_counter()

    try:
        # 1. File presence
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image provided"}), 400

        # 2. Extension check
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": "Unsupported format. Use jpg, jpeg, png or bmp"}), 400

        # 3. Size check
        file.seek(0, 2)
        size_mb = file.tell() / (1024 * 1024)
        file.seek(0)
        if size_mb > MAX_FILE_SIZE_MB:
            return jsonify({"error": f"File too large. Max {MAX_FILE_SIZE_MB}MB"}), 400

        # 4. Decode in memory — no disk writes
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 422

        # 5. Resize to 256×256 (matches training image size)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        # 6. Re-encode through JPEG pipeline to match training distribution
        #    Training images were JPEGs loaded via cv2.imread — this simulates that
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        img    = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

        # 7. Extract → scale → infer
        x_raw    = extract_features(img)
        x_scaled = scaler.transform(x_raw)
        prob = float(model.predict_proba(x_scaled)[0][0])

        label = "Steganographic" if prob >= 0.5 else "Clean"
        risk  = "High" if prob >= 0.75 else "Medium" if prob >= 0.5 else "Low"

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return jsonify({
            "is_suspicious": prob >= 0.5,
            "label":         label,
            "confidence":    round(prob * 100, 1),
            "probability": {
                "clean": round((1 - prob) * 100, 1),
                "stego": round(prob * 100, 1),
            },
            "risk_level":   risk,
            "inference_ms": elapsed_ms,
            "image_info": {
                "width": 256, "height": 256, "mode": "Grayscale"
            }
        })

    except AssertionError as e:
        return jsonify({"error": str(e)}), 500
    except ValueError as e:
        return jsonify({"error": f"Feature extraction failed: {str(e)}"}), 422
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)