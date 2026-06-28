import argparse
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator


DEFAULT_OUTPUT = "氣象線條圖.png"


def load_and_prepare(csv_path: Path, date_col: str = "LocalTime") -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=[date_col], index_col=date_col)
    cols = {c.lower(): c for c in df.columns}
    required = ["rh", "temperature", "globlrad"]
    for r in required:
        if r not in cols:
            raise KeyError(f"Required column '{r}' not found. Available: {list(df.columns)}")

    use_cols = [cols[r] for r in required]
    names = ["RH", "Temperature", "GloblRad"]
    # optional Wh
    if "wh" in cols:
        use_cols.append(cols["wh"])  # original column name
        names.append("Wh")

    df = df[use_cols]
    df.columns = names
    df = df.sort_index()
    df = df.interpolate(method="time").ffill().bfill()
    return df


def plot_and_save_series(series: pd.Series, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(series.index, series.values, lw=1)
    ax.set_ylabel(series.name)
    ax.set_title(title)
    ax.grid(True)
    locator = AutoDateLocator()
    formatter = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_three_subplots(df: pd.DataFrame, out_path: Path):
    # keep compatibility: only use RH/Temperature/GloblRad
    df3 = df.loc[:, ["RH", "Temperature", "GloblRad"]]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 9))
    colors = ["tab:blue", "tab:orange", "tab:green"]

    axes[0].plot(df3.index, df3["RH"], color=colors[0], label="RH")
    axes[0].set_ylabel("RH"); axes[0].grid(True); axes[0].legend(loc="upper right")

    axes[1].plot(df3.index, df3["Temperature"], color=colors[1], label="Temperature")
    axes[1].set_ylabel("Temperature"); axes[1].grid(True); axes[1].legend(loc="upper right")

    axes[2].plot(df3.index, df3["GloblRad"], color=colors[2], label="GloblRad")
    axes[2].set_ylabel("GloblRad"); axes[2].set_xlabel("Time"); axes[2].grid(True); axes[2].legend(loc="upper right")

    locator = AutoDateLocator(); formatter = DateFormatter("%Y-%m-%d")
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    fig.suptitle("RH / Temperature / GloblRad")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="繪製並輸出 RH/Temperature/GloblRad（可選 Wh）為獨立圖或合併三子圖")
    p.add_argument("--input", "-i", required=True, help="輸入 CSV 路徑")
    p.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="輸出 PNG 檔名（會以此為 prefix 產生多個檔案）")
    p.add_argument("--date-col", default="LocalTime", help="日期欄位名稱，預設 LocalTime")
    p.add_argument("--single", action="store_true", help="輸出原先的合成三子圖（RH/Temperature/GloblRad）")
    args = p.parse_args(argv)

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"Error: input CSV not found: {csv_path}")
        sys.exit(2)

    try:
        df = load_and_prepare(csv_path, date_col=args.date_col)
    except Exception as e:
        print(f"Error while loading/preparing data: {e}")
        raise

    out_base = Path(args.output)
    out_dir = out_base.parent if out_base.parent.exists() else Path('.')
    stem = out_base.stem

    if args.single:
        target = out_dir / (stem + ".png")
        plot_three_subplots(df, target)
        print(f"Saved combined plot to: {target}")
        return

    saved = []
    for col in df.columns:
        fname = f"{stem}_{col}.png"
        out_file = out_dir / fname
        plot_and_save_series(df[col], f"{col}", out_file)
        saved.append(out_file)

    print("Saved files:")
    for pth in saved:
        print(" -", pth)


if __name__ == "__main__":
    main()
