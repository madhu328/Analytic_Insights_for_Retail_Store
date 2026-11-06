"""POS loader: Brigade item-level CSV -> invoice-level aggregate -> Purplle POS CSV.

* One row per invoice (sales only; returns are skipped).
* PII is never propagated: customer names are dropped and phone numbers are replaced by a
  salted SHA-256 pseudonym (``customer_hash``) so repeat customers can still be counted.
* Output CSV matches the ``/pos/ingest`` contract:
  ``store_id,transaction_id,timestamp,basket_value_inr``.

Usage:
  python -m pipeline.pos_loader data/pos/Brigade_Bangalore_10_April_26.csv \
      --out-dir data/pos --store-map ST1008=STORE_BLR_002
  # anonymise a raw export before committing it:
  python -m pipeline.pos_loader RAW.csv --anonymise-to data/pos/brigade_anonymised.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

PII_COLUMNS = ["customer_name", "customer_number", "salesperson_name"]
SALT = os.getenv("POS_HASH_SALT", "store-intelligence")


def pseudonym(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() in ("", "nan"):
        return None
    return hashlib.sha256(f"{SALT}:{str(value).strip()}".encode()).hexdigest()[:16]


def anonymise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "customer_number" in df:
        df["customer_hash"] = df["customer_number"].map(pseudonym)
    return df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])


def load_items(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"invoice_number": str, "customer_number": str, "order_id": str})
    if "customer_hash" not in df.columns:
        df = anonymise(df)
    if "invoice_type" in df:
        df = df[df["invoice_type"].str.lower() == "sales"]
    df["transaction_datetime"] = pd.to_datetime(df["order_date"] + " " + df["order_time"],
                                                format="%d-%m-%Y %H:%M:%S")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    def uniq(s):
        return sorted({str(x) for x in s.dropna()})

    g = df.groupby("invoice_number", sort=False)
    out = pd.DataFrame({
        "invoice_number": g.size().index,
        "order_id": g["order_id"].first().values,
        "pos_store_code": g["store_id"].first().values,
        "transaction_datetime": g["transaction_datetime"].min().values,
        "customer_hash": g["customer_hash"].first().values,
        "salesperson_id": g["salesperson_id"].first().values if "salesperson_id" in df else None,
        "product_names": g["product_name"].apply(uniq).values,
        "brand_names": g["brand_name"].apply(uniq).values,
        "categories": g["dep_name"].apply(uniq).values,
        "total_quantity": g["qty"].sum().values,
        "total_amount": g["total_amount"].sum().round(2).values,
    })
    return out.sort_values("transaction_datetime").reset_index(drop=True)


def to_pos_contract(agg: pd.DataFrame, store_map: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({
        "store_id": agg["pos_store_code"].map(lambda c: store_map.get(c, c)),
        "transaction_id": "TXN_" + agg["invoice_number"].astype(str),
        "timestamp": pd.to_datetime(agg["transaction_datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "basket_value_inr": agg["total_amount"].round(2),
    })


def default_store_map() -> dict[str, str]:
    layout = Path(__file__).resolve().parent.parent / "config" / "store_layout.json"
    m = {}
    if layout.exists():
        for s in json.loads(layout.read_text()).get("stores", []):
            for code in s.get("pos_store_codes", []):
                m[code] = s["store_id"]
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--out-dir", default="data/pos")
    ap.add_argument("--store-map", nargs="*", default=[], help="POS_CODE=STORE_ID")
    ap.add_argument("--anonymise-to", help="only write an anonymised copy of the raw CSV")
    args = ap.parse_args(argv)
    if args.anonymise_to:
        raw = pd.read_csv(args.csv, dtype=str)
        anonymise(raw).to_csv(args.anonymise_to, index=False)
        print(f"anonymised {len(raw)} rows -> {args.anonymise_to}")
        return
    store_map = default_store_map() | dict(x.split("=", 1) for x in args.store_map)
    agg = aggregate(load_items(args.csv))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    agg_json = agg.assign(transaction_datetime=agg["transaction_datetime"].astype(str))
    agg_json.to_json(out / "aggregated_transactions.json", orient="records", indent=2)
    agg_json.to_csv(out / "aggregated_transactions.csv", index=False)
    pos = to_pos_contract(agg, store_map)
    pos.to_csv(out / "purpple_pos_transactions.csv", index=False)
    print(f"{len(agg)} invoices, INR {agg['total_amount'].sum():,.2f} -> {out}/purpple_pos_transactions.csv "
          f"(generated {datetime.now():%Y-%m-%d %H:%M})")


if __name__ == "__main__":
    main()
