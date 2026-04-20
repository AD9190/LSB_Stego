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

app = Flask(__name__)
CORS(app)

# ── Load models once at startup ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
model  = joblib.load(os.path.join(BASE_DIR, "models", "mlp_model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.joblib"))

ALLOWED_EXTENSIONS = {".jpg", ".jpeg"}
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
]

# ── Optimised feature helpers ─────────────────────────────────────────────────

def _fast_entropy(flat_float: np.ndarray) -> float:
    """Entropy via np.bincount — ~5x faster than np.histogram."""
    flat_int  = np.clip(flat_float, 0, 255).astype(np.uint8)
    hist      = np.bincount(flat_int.ravel(), minlength=256).astype(np.float64)
    hist_norm = hist / (hist.sum() + 1e-8)
    return float(-np.sum(hist_norm * np.log2(hist_norm + 1e-8)))


def _spam_stats(diff_matrix: np.ndarray):
    """SPAM stats for one direction — uses ravel (no copy) and fast entropy."""
    flat = diff_matrix.ravel()
    return (
        float(np.mean(flat)),
        float(np.std(flat)),
        float(np.median(flat)),
        float(skew(flat)),
        _fast_entropy(flat),
    )


def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Input : 256×256 uint8 grayscale numpy array
    Output: (1, 39) float32 — redundant features excluded, same order as training
    Key optimisation: spatial diff matrices computed ONCE, shared across all feature groups
    """
    # ── Convert once ──────────────────────────────────────────────────────────
    img_f = img.astype(np.float64)

    # ── Histogram (cv2 is faster than np.histogram for 8-bit images) ──────────
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().astype(np.float64)
    hist_norm = hist / (hist.sum() + 1e-8)
    intensity  = np.arange(256, dtype=np.float64)

    # ── Statistical moments ───────────────────────────────────────────────────
    mean     = float(np.dot(intensity, hist_norm))
    variance = float(np.dot((intensity - mean) ** 2, hist_norm))
    std_dev  = float(np.sqrt(variance)) if variance > 0 else 0.0

    skewness = float(np.dot(((intensity - mean) / (std_dev + 1e-8)) ** 3, hist_norm)) if std_dev > 0 else 0.0
    kurt     = float(np.dot(((intensity - mean) / (std_dev + 1e-8)) ** 4, hist_norm)) - 3 if std_dev > 0 else 0.0

    cumsum    = np.cumsum(hist_norm)
    mid_idx   = np.searchsorted(cumsum, 0.5)               # faster than np.where
    median    = float(intensity[min(mid_idx, 255)])

    hist_nz   = hist[hist > 0]
    geo_mean  = float(np.exp(np.mean(np.log(hist_nz + 1e-8)))) if len(hist_nz) > 0 else 0.0
    pix_range = float(np.max(hist) - np.min(hist))

    # ── Hjorth parameters (on histogram signal) ───────────────────────────────
    activity   = float(np.var(hist))
    d1         = np.diff(hist)
    mobility   = float(np.sqrt(np.var(d1) / (activity + 1e-8)))
    d2         = np.diff(d1)
    complexity = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-8)) / (mobility + 1e-8))

    # ── Entropy / chi-square / energy ─────────────────────────────────────────
    entropy    = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-8)))
    uniform    = np.ones(256) / 256.0
    chi_sq     = float(np.sum((hist_norm - uniform) ** 2 / (uniform + 1e-8)))
    energy     = float(np.sum(hist_norm ** 2))

    # ── Even-odd ratio ────────────────────────────────────────────────────────
    even_odd = float(hist[::2].sum() / (hist[1::2].sum() + 1e-8))

    # ── Spatial diff matrices — computed ONCE, reused below ───────────────────
    diff_h  = np.abs(np.diff(img_f, axis=1))    # horizontal
    diff_v  = np.abs(np.diff(img_f, axis=0))    # vertical
    diff_dp = np.abs(img_f[:-1, :-1] - img_f[1:, 1:])   # diagonal +
    diff_dn = np.abs(img_f[:-1, 1:]  - img_f[1:, :-1])  # diagonal −

    # ── Pixel diff stats (all directions combined) ────────────────────────────
    all_diffs = np.concatenate([
        diff_h.ravel(), diff_v.ravel(), diff_dp.ravel(), diff_dn.ravel()
    ])
    pd_mean   = float(np.mean(all_diffs))
    pd_std    = float(np.std(all_diffs))
    pd_max    = float(np.max(all_diffs))
    pd_median = float(np.median(all_diffs))

    # ── Pairwise histogram diff (Adjacent Bin Difference + Sum dropped) ───────
    pw_diffs  = np.abs(hist[:-1] - hist[1:])
    pw_mean   = float(np.mean(pw_diffs))
    pw_std    = float(np.std(pw_diffs))
    pw_max    = float(np.max(pw_diffs))
    # pw_sum  dropped (redundant with pw_mean)

    # ── SPAM directional features — reuse diff matrices ───────────────────────
    h_mean,  h_std,  h_med,  h_skw,  h_ent  = _spam_stats(diff_h)
    v_mean,  v_std,  v_med,  v_skw,  v_ent  = _spam_stats(diff_v)
    dp_mean, dp_std, dp_med, dp_skw, dp_ent = _spam_stats(diff_dp)
    dn_mean, dn_std, dn_med, dn_skw, dn_ent = _spam_stats(diff_dn)

    # ── Assemble in FEATURE_COLS order ────────────────────────────────────────
    values = [
        kurt, skewness, std_dev, pix_range, median, geo_mean,
        mobility, complexity,
        entropy, even_odd, chi_sq, energy,
        pd_mean, pd_std, pd_max, pd_median,
        pw_mean, pw_std, pw_max,
        h_mean,  h_std,  h_med,  h_skw,  h_ent,
        v_mean,  v_std,  v_med,  v_skw,  v_ent,
        dp_mean, dp_std, dp_med, dp_skw, dp_ent,
        dn_mean, dn_std, dn_med, dn_skw, dn_ent,
    ]

    assert len(values) == len(FEATURE_COLS), \
        f"Feature count mismatch: got {len(values)}, expected {len(FEATURE_COLS)}"

    return np.array([values], dtype=np.float32)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
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
            return jsonify({"error": "Unsupported format. Use jpg or jpeg"}), 400

        # 3. Size check — seek to end, then reset
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

        # 5. Resize to 256×256
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        # 6. Extract → scale → infer
        x_raw    = extract_features(img)
        x_scaled = scaler.transform(x_raw)
        prob     = float(model.predict_proba(x_scaled)[0][1])

        label = "Steganographic" if prob >= 0.5 else "Clean"
        risk  = "High" if prob >= 0.75 else "Medium" if prob >= 0.5 else "Low"

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return jsonify({
            "label":      label,
            "confidence": round(prob * 100, 1),
            "probability": {
                "clean": round((1 - prob) * 100, 1),
                "stego": round(prob * 100, 1),
            },
            "risk_level":    risk,
            "inference_ms":  elapsed_ms,
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
    # Single worker dev server
    # For production: gunicorn -w 4 -b 0.0.0.0:5000 app:app
    app.run(debug=True, host="0.0.0.0", port=5000)