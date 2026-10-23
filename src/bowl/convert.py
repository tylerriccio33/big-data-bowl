"""Convert the raw Kaggle CSVs in data/raw to parquet in data/parquet."""

from pathlib import Path

import polars as pl

RAW = Path("data/raw")
OUT = Path("data/parquet")


def convert(csv: Path) -> Path:
    dest = OUT / f"{csv.stem}.parquet"
    (
        pl.scan_csv(csv, null_values=["NA", ""], infer_schema_length=100_000).sink_parquet(
            dest, compression="zstd"
        )
    )
    return dest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for csv in sorted(RAW.glob("*.csv")):
        dest = convert(csv)
        mb = dest.stat().st_size / 1e6
        print(f"{csv.name:32} -> {dest.name} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
