"""
Generic Regime-Aware Mixture-of-Experts (MoE) Multi-Horizon Forecaster

Goal:
- Read a CSV that already contains:
    - a date column
    - one target column (numeric)
    - zero or more feature columns (numeric)
- build_dataset() constructs a supervised multi-horizon dataset using:
    - rolling stats on selected columns
    - optional wavelet energy + STFT features on selected columns (windowed)
- fit a regime model (HMM if available else KMeans) using chosen regime feature columns
- train per-regime experts (median + quantiles) for each horizon step
- forecast next H business days as a regime-mixture

This is intentionally generic: no domain-specific terms, no hardcoded column names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt
from scipy.signal import stft

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor

# Optional HMM
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False


# =========================
# CONFIG
# =========================
@dataclass
class MoEConfig:
    # I/O
    csv_path: str = "data.csv"
    date_col: str = "date"
    target_col: str = "target"

    # If empty, the library will use all numeric columns except date_col + target_col
    feature_cols: Optional[List[str]] = None

    # Time framing
    lookback: int = 60
    horizon: int = 30

    # Rolling feature engineering
    rolling_windows: Sequence[int] = (5, 20)

    # Spectral features
    use_wavelet: bool = True
    wavelet: str = "db4"
    wavelet_level: int = 3

    use_stft: bool = True

    # Clipping (helps stability on huge scales)
    target_clip: float = 5e10

    # Regimes
    n_regimes: int = 3
    regime_names: Sequence[str] = ("CALM", "NORMAL", "STRESS")

    # Columns used to infer regimes (must exist in ds after build_dataset)
    # Defaults are generic: abs(target_t), abs(ret-like feature if present), and rolling std of target_t
    regime_feature_mode: str = "auto"  # "auto" or "explicit"
    regime_feature_cols_explicit: Optional[List[str]] = None

    # Experts
    quantiles: Sequence[float] = (0.10, 0.50, 0.90)
    cv_splits: int = 5

    # Output filenames
    features_out_csv: str = "features_moe_multi_horizon.csv"
    forecast_out_csv: str = "forecast_moe.csv"


# =========================
# LOAD
# =========================
def load_csv_generic(path: str, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"CSV missing date column '{date_col}'")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return df


# =========================
# FEATURE BUILDING HELPERS
# =========================
def rolling_features(series: pd.Series, win: int) -> pd.DataFrame:
    s = series
    return pd.DataFrame(
        {
            f"mean_{win}": s.rolling(win).mean(),
            f"std_{win}": s.rolling(win).std(),
            f"skew_{win}": s.rolling(win).skew(),
            f"kurt_{win}": s.rolling(win).kurt(),
            f"min_{win}": s.rolling(win).min(),
            f"max_{win}": s.rolling(win).max(),
        }
    )

def wavelet_energy(x: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    # For short windows, pywt may complain on high level; handle robustly.
    try:
        coeffs = pywt.wavedec(x, wavelet, level=level)
    except Exception:
        # Fall back to a single-level decomposition or zeros
        try:
            coeffs = pywt.wavedec(x, wavelet, level=1)
        except Exception:
            return np.zeros(level + 1, dtype=float)

    e = np.array([float(np.sum(c**2)) for c in coeffs], dtype=float)
    e[~np.isfinite(e)] = 0.0
    return e

def stft_feats(x: np.ndarray) -> np.ndarray:
    if len(x) < 8:
        return np.zeros(4, dtype=float)
    _, _, Z = stft(x, nperseg=min(64, len(x)))
    P = (np.abs(Z) ** 2).mean(axis=1)
    if not np.all(np.isfinite(P)) or P.sum() <= 0:
        return np.zeros(4, dtype=float)

    p = P / P.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))

    n = len(P)
    b1 = max(1, n // 3)
    b2 = max(2, 2 * n // 3)
    low = P[:b1].sum()
    mid = P[b1:b2].sum()
    high = P[b2:].sum()

    out = np.array([entropy, low, mid, high], dtype=float)
    out[~np.isfinite(out)] = 0.0
    return out


# =========================
# BUILD SUPERVISED DATASET (Generic)
# =========================
def build_dataset(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    feature_cols: Optional[List[str]],
    lookback: int,
    horizon: int,
    rolling_windows: Sequence[int],
    target_clip: float,
    use_wavelet: bool,
    wavelet: str,
    wavelet_level: int,
    use_stft: bool,
) -> pd.DataFrame:
    """
    Produces a dataset with:
      - date
      - target_t (current target)
      - optional raw feature_t columns
      - rolling stats for each selected series (features + target)
      - wavelet + stft features for each selected series (features + target)
      - targets: y_t_plus_1 ... y_t_plus_H
    """

    if date_col not in df.columns:
        raise ValueError(f"Missing date_col='{date_col}' in df")
    if target_col not in df.columns:
        raise ValueError(f"Missing target_col='{target_col}' in df")

    d = df.copy()

    # Coerce numeric
    d[target_col] = pd.to_numeric(d[target_col], errors="coerce")
    d = d.dropna(subset=[date_col, target_col]).reset_index(drop=True)

    # Determine feature columns (default: all numeric except date+target)
    if feature_cols is None:
        numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
        # df may treat some numeric-like columns as object; user can pass explicitly in that case.
        feature_cols = [c for c in numeric_cols if c not in {target_col}]
    else:
        missing = [c for c in feature_cols if c not in d.columns]
        if missing:
            raise ValueError(f"feature_cols missing in CSV: {missing}")

    # Ensure features numeric
    for c in feature_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)

    # Clip target (robustness)
    d["_target_clipped"] = d[target_col].clip(-target_clip, target_clip)

    # Base frame
    base = pd.DataFrame({date_col: d[date_col].values})
    base["target_t"] = d["_target_clipped"].astype(float).values

    # Include raw feature values at time t (optional but useful)
    for c in feature_cols:
        base[f"feat_{c}_t"] = d[c].astype(float).values

    # Rolling features for target + each feature
    for win in rolling_windows:
        # target rolling
        rf = rolling_features(pd.Series(base["target_t"]), win).add_prefix(f"tgt_")
        rf = rf.add_prefix(f"r{win}_")
        base = pd.concat([base, rf], axis=1)

        # feature rolling
        for c in feature_cols:
            s = pd.Series(d[c].astype(float).values)
            f_rf = rolling_features(s, win).add_prefix(f"f_{c}_")
            f_rf = f_rf.add_prefix(f"r{win}_")
            base = pd.concat([base, f_rf], axis=1)

    # Build supervised rows
    rows = []
    n = len(base)
    min_i = lookback
    max_i = n - horizon
    if max_i <= min_i:
        raise ValueError(
            f"Not enough data to build supervised rows. "
            f"Need len(df) > lookback+horizon, got n={n}, lookback={lookback}, horizon={horizon}"
        )

    # Pre-extract series arrays for windowing efficiency
    series_map: Dict[str, np.ndarray] = {"target_t": base["target_t"].to_numpy(dtype=float)}
    for c in feature_cols:
        series_map[f"feat_{c}_t"] = base[f"feat_{c}_t"].to_numpy(dtype=float)

    for i in range(min_i, max_i):
        row: Dict[str, float] = {}
        row["date"] = base[date_col].iloc[i]
        row["target_t"] = float(base["target_t"].iloc[i]) if np.isfinite(base["target_t"].iloc[i]) else 0.0

        # Add raw feat_t
        for c in feature_cols:
            v = base[f"feat_{c}_t"].iloc[i]
            row[f"feat_{c}_t"] = float(v) if np.isfinite(v) else 0.0

        # Add rolling columns (safe scalar)
        for c in base.columns:
            if c in (date_col, "target_t") or c.startswith("feat_"):
                continue
            v = base[c].iloc[i]
            if isinstance(v, (pd.Series, np.ndarray, list)):
                v = np.asarray(v).ravel()[0]
            row[c] = float(v) if pd.notna(v) and np.isfinite(v) else 0.0

        # Add spectral features per windowed series (target + each feature)
        for s_name, arr in series_map.items():
            window = arr[i - lookback : i]
            if window.size != lookback:
                continue
            window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0).astype(float)

            if use_wavelet:
                w = wavelet_energy(window, wavelet=wavelet, level=wavelet_level)
                for j, val in enumerate(w):
                    row[f"w_{s_name}_e{j}"] = float(val)

            if use_stft:
                s = stft_feats(window)
                for j, val in enumerate(s):
                    row[f"s_{s_name}_{j}"] = float(val)

        # Multi-horizon targets
        for h in range(1, horizon + 1):
            y = base["target_t"].iloc[i + h]
            row[f"y_t_plus_{h}"] = float(y) if np.isfinite(y) else 0.0

        rows.append(row)

    ds = (
        pd.DataFrame(rows)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )
    return ds


# =========================
# REGIME MODEL
# =========================
def fit_regime_model(
    ds: pd.DataFrame,
    *,
    n_regimes: int,
    regime_feature_mode: str,
    regime_feature_cols_explicit: Optional[List[str]],
) -> Tuple[str, object, StandardScaler, np.ndarray, List[str]]:
    """
    Returns:
      kind: "hmm" or "kmeans"
      model
      scaler
      post: posterior (or hard 1-hot for kmeans)
      used_cols: list of columns used in regime model
    """
    if regime_feature_mode not in ("auto", "explicit"):
        raise ValueError("regime_feature_mode must be 'auto' or 'explicit'")

    if regime_feature_mode == "explicit":
        if not regime_feature_cols_explicit:
            raise ValueError("Explicit mode requires regime_feature_cols_explicit")
        missing = [c for c in regime_feature_cols_explicit if c not in ds.columns]
        if missing:
            raise ValueError(f"Regime feature columns missing in ds: {missing}")
        used_cols = list(regime_feature_cols_explicit)
        z = ds[used_cols].copy()
        z = z.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    else:
        # AUTO mode: robust defaults that always exist
        used_cols = []

        # 1) abs(target_t)
        used_cols.append("_abs_target")
        z = pd.DataFrame({"_abs_target": np.abs(pd.to_numeric(ds["target_t"], errors="coerce")).fillna(0.0)})

        # 2) rolling std of abs(target_t)
        used_cols.append("_target_std20")
        z["_target_std20"] = (
            pd.Series(z["_abs_target"].to_numpy(dtype=float)).rolling(20).std().fillna(0.0).to_numpy(dtype=float)
        )

        # 3) if there is any "return-like" feature, use abs of the first match
        ret_like = None
        for c in ds.columns:
            lc = c.lower()
            if "ret" in lc and ("_t" in lc or lc.endswith("_t")):
                ret_like = c
                break
        if ret_like is not None:
            used_cols.append("_abs_retlike")
            z["_abs_retlike"] = np.abs(pd.to_numeric(ds[ret_like], errors="coerce")).fillna(0.0)

    scaler = StandardScaler()
    Z = scaler.fit_transform(z)

    if HMM_AVAILABLE:
        hmm = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=300, random_state=42)
        hmm.fit(Z)
        post = hmm.predict_proba(Z)
        return ("hmm", hmm, scaler, post, used_cols)
    else:
        km = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        lab = km.fit_predict(Z)
        post = np.zeros((len(Z), n_regimes))
        post[np.arange(len(Z)), lab] = 1.0
        return ("kmeans", km, scaler, post, used_cols)


def order_regimes_by_stress(ds: pd.DataFrame, post: np.ndarray, n_regimes: int):
    abs_tgt = np.abs(ds["target_t"].to_numpy(dtype=float))
    hard = post.argmax(axis=1)

    stats = []
    for k in range(n_regimes):
        m = abs_tgt[hard == k].mean() if np.any(hard == k) else np.inf
        stats.append((k, m))

    stats_sorted = sorted(stats, key=lambda x: x[1])  # low -> high stress
    mapping = {old: new for new, (old, _) in enumerate(stats_sorted)}

    post2 = np.zeros_like(post)
    for old_k, new_k in mapping.items():
        post2[:, new_k] = post[:, old_k]
    return mapping, post2


def regime_posterior_last(
    ds: pd.DataFrame,
    *,
    kind: str,
    model: object,
    scaler: StandardScaler,
    mapping: dict,
    n_regimes: int,
    regime_feature_mode: str,
    regime_feature_cols_explicit: Optional[List[str]],
):
    # Construct the same z-row as fit_regime_model
    if regime_feature_mode == "explicit":
        cols = regime_feature_cols_explicit or []
        row = ds.iloc[[-1]][cols].copy()
        row = row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        Z = scaler.transform(row)
    else:
        last = ds.iloc[[-1]]
        abs_target = np.abs(pd.to_numeric(last["target_t"], errors="coerce")).fillna(0.0).to_numpy(dtype=float)
        target_std20 = pd.Series(np.abs(ds["target_t"].to_numpy(dtype=float))).rolling(20).std().iloc[-1]
        if not np.isfinite(target_std20):
            target_std20 = 0.0

        z = pd.DataFrame({"_abs_target": abs_target, "_target_std20": [float(target_std20)]})

        ret_like = None
        for c in ds.columns:
            lc = c.lower()
            if "ret" in lc and ("_t" in lc or lc.endswith("_t")):
                ret_like = c
                break
        if ret_like is not None:
            abs_retlike = np.abs(pd.to_numeric(last[ret_like], errors="coerce")).fillna(0.0).to_numpy(dtype=float)
            z["_abs_retlike"] = abs_retlike

        Z = scaler.transform(z)

    if kind == "hmm":
        p_old = model.predict_proba(Z)[0]
    else:
        lab = model.predict(Z)[0]
        p_old = np.zeros(n_regimes, dtype=float)
        p_old[lab] = 1.0

    # remap to ordered regimes
    p_new = np.zeros_like(p_old)
    for old_k, new_k in mapping.items():
        p_new[new_k] = p_old[old_k]
    return p_new


# =========================
# TRAIN EXPERTS (per regime)
# =========================
def train_experts(
    ds: pd.DataFrame,
    post: np.ndarray,
    *,
    horizon: int,
    n_regimes: int,
    regime_names: Sequence[str],
    quantiles: Sequence[float],
    cv_splits: int,
):
    y_cols = [f"y_t_plus_{h}" for h in range(1, horizon + 1)]
    X = ds.drop(columns=["date"] + y_cols)
    Y = ds[y_cols].to_numpy(dtype=float)

    labels = post.argmax(axis=1)

    experts = {}
    cv = TimeSeriesSplit(n_splits=cv_splits)

    for k in range(n_regimes):
        idx = np.where(labels == k)[0]
        if len(idx) < 50:
            print(f"[WARN] Regime {k} has only {len(idx)} samples. Consider fewer regimes or more data.")

        Xk = X.iloc[idx].reset_index(drop=True)
        Yk = Y[idx, :]

        name = regime_names[k] if k < len(regime_names) else f"REGIME_{k}"
        print(f"\nTraining expert regime {k} ({name}) | samples={len(idx)}")

        med_models = []
        q_models = {q: [] for q in quantiles}

        for h in range(horizon):
            med = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_depth=5,
                max_iter=600,
                random_state=42,
            )

            # Optional CV MAE diagnostics
            if len(Xk) >= 200:
                maes = []
                for tr, te in cv.split(Xk):
                    med.fit(Xk.iloc[tr], Yk[tr, h])
                    pred = med.predict(Xk.iloc[te])
                    maes.append(mean_absolute_error(Yk[te, h], pred))
                if (h + 1) in [1, 5, 10, 20, 30]:
                    print(f"  t+{h+1:02d} median MAE: {np.mean(maes):,.4f}")

            med.fit(Xk, Yk[:, h])
            med_models.append(med)

            for q in quantiles:
                qb = GradientBoostingRegressor(
                    loss="quantile",
                    alpha=q,
                    learning_rate=0.05,
                    n_estimators=600,
                    max_depth=3,
                    random_state=42,
                )
                qb.fit(Xk, Yk[:, h])
                q_models[q].append(qb)

        experts[k] = {"median": med_models, "quantiles": q_models}

    return experts


# =========================
# FORECAST (mixture across regimes)
# =========================
def forecast_horizon(
    ds: pd.DataFrame,
    experts: dict,
    p_regime: np.ndarray,
    *,
    horizon: int,
    n_regimes: int,
    quantiles: Sequence[float],
) -> pd.DataFrame:
    y_cols = [f"y_t_plus_{h}" for h in range(1, horizon + 1)]
    X_last = ds.drop(columns=["date"] + y_cols).iloc[[-1]]

    preds_med = []
    preds_q = {q: [] for q in quantiles}

    for h in range(horizon):
        m = 0.0
        for k in range(n_regimes):
            m += p_regime[k] * float(experts[k]["median"][h].predict(X_last)[0])
        preds_med.append(m)

        for q in quantiles:
            v = 0.0
            for k in range(n_regimes):
                v += p_regime[k] * float(experts[k]["quantiles"][q][h].predict(X_last)[0])
            preds_q[q].append(v)

    start = pd.to_datetime(ds["date"].iloc[-1]) + pd.Timedelta(days=1)
    dates = pd.bdate_range(start, periods=horizon)

    out = pd.DataFrame({"date": dates})
    # Always output p50 as the "median mixture" (even if 0.5 not in quantiles)
    out["pred_p50"] = preds_med

    # If user provided 0.10/0.90 etc, add those
    for q in sorted(set(quantiles)):
        if q == 0.50:
            continue
        out[f"pred_p{int(round(q*100)):02d}"] = preds_q[q]

    # Standard ordering if p10/p50/p90 exist
    cols = ["date"]
    if "pred_p10" in out.columns:
        cols += ["pred_p10"]
    cols += ["pred_p50"]
    if "pred_p90" in out.columns:
        cols += ["pred_p90"]
    # add any remaining quantiles
    for c in out.columns:
        if c.startswith("pred_p") and c not in cols:
            cols.append(c)

    return out[cols]


# =========================
# ORCHESTRATOR
# =========================
class GenericMoEForecaster:
    def __init__(self, config: MoEConfig):
        self.cfg = config

        self.ds_: Optional[pd.DataFrame] = None
        self.regime_kind_: Optional[str] = None
        self.regime_model_: Optional[object] = None
        self.regime_scaler_: Optional[StandardScaler] = None
        self.regime_mapping_: Optional[dict] = None
        self.regime_post_ord_: Optional[np.ndarray] = None
        self.regime_used_cols_: Optional[List[str]] = None
        self.experts_: Optional[dict] = None

    def fit(self) -> "GenericMoEForecaster":
        cfg = self.cfg

        raw = load_csv_generic(cfg.csv_path, cfg.date_col)

        ds = build_dataset(
            raw,
            date_col=cfg.date_col,
            target_col=cfg.target_col,
            feature_cols=cfg.feature_cols,
            lookback=cfg.lookback,
            horizon=cfg.horizon,
            rolling_windows=cfg.rolling_windows,
            target_clip=cfg.target_clip,
            use_wavelet=cfg.use_wavelet,
            wavelet=cfg.wavelet,
            wavelet_level=cfg.wavelet_level,
            use_stft=cfg.use_stft,
        )
        self.ds_ = ds

        print(f"Supervised dataset: rows={len(ds):,} cols={ds.shape[1]:,}")
        ds.to_csv(cfg.features_out_csv, index=False)
        print(f"Saved: {cfg.features_out_csv}")

        kind, reg_model, scaler, post, used_cols = fit_regime_model(
            ds,
            n_regimes=cfg.n_regimes,
            regime_feature_mode=cfg.regime_feature_mode,
            regime_feature_cols_explicit=cfg.regime_feature_cols_explicit,
        )
        mapping, post_ord = order_regimes_by_stress(ds, post, cfg.n_regimes)

        self.regime_kind_ = kind
        self.regime_model_ = reg_model
        self.regime_scaler_ = scaler
        self.regime_mapping_ = mapping
        self.regime_post_ord_ = post_ord
        self.regime_used_cols_ = used_cols

        print(f"Regime model: {kind.upper()} | Remap (old->ordered): {mapping}")
        print(f"Regime features used: {used_cols}")

        self.experts_ = train_experts(
            ds,
            post_ord,
            horizon=cfg.horizon,
            n_regimes=cfg.n_regimes,
            regime_names=cfg.regime_names,
            quantiles=cfg.quantiles,
            cv_splits=cfg.cv_splits,
        )
        return self

    def predict(self) -> pd.DataFrame:
        if self.ds_ is None or self.experts_ is None:
            raise RuntimeError("Call fit() before predict().")

        cfg = self.cfg

        p_last = regime_posterior_last(
            self.ds_,
            kind=self.regime_kind_,
            model=self.regime_model_,
            scaler=self.regime_scaler_,
            mapping=self.regime_mapping_,
            n_regimes=cfg.n_regimes,
            regime_feature_mode=cfg.regime_feature_mode,
            regime_feature_cols_explicit=cfg.regime_feature_cols_explicit,
        )

        print("\nCurrent regime probabilities (ordered low->high stress):")
        for i in range(cfg.n_regimes):
            name = cfg.regime_names[i] if i < len(cfg.regime_names) else f"REGIME_{i}"
            print(f"  {name:8s}: {p_last[i]:.3f}")

        fc = forecast_horizon(
            self.ds_,
            self.experts_,
            p_last,
            horizon=cfg.horizon,
            n_regimes=cfg.n_regimes,
            quantiles=cfg.quantiles,
        )

        fc.to_csv(cfg.forecast_out_csv, index=False)
        print(f"\nSaved: {cfg.forecast_out_csv}")
        return fc

# =========================
# Run Model
# =========================
def run_model(csv_path,date_col,target_col,feature_cols=None):
    cfg = MoEConfig(
        csv_path=csv_path,
        date_col=date_col,
        target_col=target_col,

        feature_cols=feature_cols,

        lookback=60,
        horizon=30,

        rolling_windows=(5, 20),
        use_wavelet=True,
        wavelet="db4",
        wavelet_level=3,
        use_stft=True,

        n_regimes=3,
        regime_names=("CALM", "NORMAL", "STRESS"),

        # Regime features:
        # - auto uses abs(target_t), rolling std of abs(target_t), and optionally a ret-like feature if present
        regime_feature_mode="auto",

        # Or explicit:
        # regime_feature_mode="explicit",
        # regime_feature_cols_explicit=["target_t", "r20_tgt_std_20", "feat_some_retlike_t"],

        quantiles=(0.10, 0.50, 0.90),
        cv_splits=5,
    )

    model = GenericMoEForecaster(cfg).fit()
    forecast = model.predict()
    print("\nForecast head:\n", forecast.head().to_string(index=False))

# =========================
# Run model (example)
# =========================
run_model(csv_path="data.csv",date_col="date",target_col="price",feature_cols=["units"])