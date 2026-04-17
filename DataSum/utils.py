from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import pandas as pd


logger = logging.getLogger(__name__)


def robust_read_csv(path: str, date_column: Optional[str] = None, encoding: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    read_attempts = [
        {"encoding": encoding or "utf-8", "comment": "#", "low_memory": False},
        {"encoding": "latin-1", "comment": "#", "low_memory": False},
        {"encoding": encoding or "utf-8", "comment": "#", "low_memory": False, "engine": "python", "sep": None},
        {"encoding": "latin-1", "comment": "#", "low_memory": False, "engine": "python", "sep": None},
    ]
    for opts in read_attempts:
        try:
            if date_column:
                df = pd.read_csv(p, parse_dates=[date_column], **opts)
            else:
                df = pd.read_csv(p, **opts)
            return df
        except Exception:
            logger.debug("read_csv failed for %s with opts %s", p, opts, exc_info=True)
            continue
    # final attempt, raise if fails
    try:
        df = pd.read_csv(p, low_memory=False, encoding=encoding or "utf-8", comment="#")
        return df
    except Exception as e:
        logger.error("Failed to read CSV %s: %s", p, e)
        raise


def detect_datetime_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    candidates = [c for c in df.columns if any(k in c.lower() for k in ("time", "date", "timestamp", "datetime", "ts", "local"))]
    for c in candidates:
        try:
            ser = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            if ser.notna().sum() >= max(1, int(len(ser) * 0.3)):
                return c
        except Exception:
            continue
    if len(df.columns) > 0:
        c = df.columns[0]
        try:
            ser = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            if ser.notna().any():
                return c
        except Exception:
            pass
    return None


def infer_time_info(df: pd.DataFrame, date_col: Optional[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {"date_column": date_col, "start": None, "end": None, "inferred_freq": None, "median_delta_seconds": None}
    if not date_col or date_col not in df.columns:
        return info
    ser = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True).dropna().sort_values()
    if ser.empty:
        return info
    info["start"] = ser.iloc[0]
    info["end"] = ser.iloc[-1]
    try:
        info["inferred_freq"] = pd.infer_freq(ser)
    except Exception:
        info["inferred_freq"] = None
    if info["inferred_freq"] is None:
        diffs = ser.diff().dt.total_seconds().dropna()
        if not diffs.empty:
            info["median_delta_seconds"] = float(diffs.median())
    return info


def compute_numeric_stats(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    rows: List[Dict[str, Any]] = []
    for c in cols:
        ser = pd.to_numeric(df[c], errors="coerce")
        cnt = len(ser)
        non_null = int(ser.count())
        missing = int(cnt - non_null)
        mean = ser.mean()
        median = ser.median()
        std = ser.std()
        _min = ser.min()
        _25 = ser.quantile(0.25)
        _75 = ser.quantile(0.75)
        _max = ser.max()
        _sum = ser.sum(skipna=True)
        skew = ser.skew()
        kurt = ser.kurt()
        rows.append({
            "variable": c,
            "type": "numeric",
            "count": cnt,
            "non_null": non_null,
            "missing_count": missing,
            "mean": mean,
            "median": median,
            "std": std,
            "min": _min,
            "25%": _25,
            "75%": _75,
            "max": _max,
            "sum": _sum,
            "skew": skew,
            "kurtosis": kurt,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("variable")


def compute_categorical_stats(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    if cols is None:
        cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    rows: List[Dict[str, Any]] = []
    for c in cols:
        ser = df[c].astype(object).dropna()
        unique = int(ser.nunique(dropna=True)) if not ser.empty else 0
        top = None
        freq = None
        if not ser.empty:
            vc = ser.value_counts()
            top = vc.index[0]
            freq = int(vc.iloc[0])
        rows.append({"variable": c, "type": "categorical", "unique_count": unique, "top": top, "freq": freq})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("variable")


def save_summary(df: pd.DataFrame, out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=True)


def make_plots(df: pd.DataFrame, date_col: Optional[str], col: str, outdir: str, style: str = "advanced") -> None:
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except Exception:
        sns = None
    outp = Path(outdir)
    outp.mkdir(parents=True, exist_ok=True)
    ser = df[col]
    # histogram (+kde if seaborn available)
    plt.figure(figsize=(8, 4))
    vals = ser.dropna().values
    if len(vals) > 0:
        if sns is not None and style == "advanced":
            sns.histplot(vals, kde=True)
        else:
            plt.hist(vals, bins=50)
    plt.title(f"Distribution: {col}")
    plt.tight_layout()
    plt.savefig(outp / f"{col}_hist.png", dpi=150)
    plt.close()
    # violin (if seaborn)
    if sns is not None and len(vals) > 0 and style == "advanced":
        plt.figure(figsize=(4, 6))
        sns.violinplot(y=vals)
        plt.title(f"Violin: {col}")
        plt.tight_layout()
        plt.savefig(outp / f"{col}_violin.png", dpi=150)
        plt.close()
    # time series
    if date_col and date_col in df.columns:
        ser_time = pd.to_datetime(df[date_col], errors="coerce")
        df_ts = pd.DataFrame({date_col: ser_time, col: ser}).dropna(subset=[date_col])
        if not df_ts.empty:
            df_ts = df_ts.set_index(date_col).sort_index()
            plt.figure(figsize=(10, 4))
            df_ts[col].plot()
            plt.title(f"Time series: {col}")
            plt.tight_layout()
            plt.savefig(outp / f"{col}_ts.png", dpi=150)
            plt.close()
