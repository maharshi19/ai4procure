"""
AI4Procure v2.0 — Supply Chain Procurement Intelligence Platform
================================================================
A production-grade procurement intelligence engine built for supply chain companies.
Processes three SAP data layers and surfaces real-time risk, demand signals,
inventory health, and buyer action guidance.

DATA SOURCES (all confirmed linked via Plant USD6 + PRQ=Order ID):
  • EBAN  — Purchase Requisitions (SAP table EBAN)
  • COOIS — Production Orders     (SAP transaction COOIS / CO03)
  • MATDOC— Material Documents    (SAP Material Movement / MB51)

CONFIRMED LINKAGES FROM YOUR DATA:
  1. PRQ IDs → Production Order IDs (6 confirmed: 20006722, 20006790, 20006796,
                                      20006810, 20006835, 20006857)
  2. Plant USD6 → Shared across EBAN (8 rows) and MATDOC (243 rows)
  3. Company US10 → Shared across MATDOC and Orders (MRP controller PS1)
  4. Date overlap → All three files cover Jan 2024 – Mar 2026

MODEL NOTE:
  Current dataset has 25 months of orders. The seasonality model (Ridge regression
  with Fourier features + rolling averages) achieves MAPE ~42% on holdout — honest
  but not yet production-grade for forecasting. The system is architected to swap in
  Prophet/SARIMA once 36+ months of data and Material-level COOIS export are available.
  Phase 1 (alerts, cross-match, inventory intelligence) IS production-ready.

Usage:
  python ai4procure_v2.py --eban PATH --orders PATH --matdoc PATH [--port 5000]
  python ai4procure_v2.py --export   # export JSON payload only
  python ai4procure_v2.py --demo     # run with embedded demo data

Requirements:
  pip install flask pandas openpyxl scikit-learn numpy gunicorn
"""

import argparse, json, os, sys, warnings
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
    SKLEARN = True
except ImportError:
    SKLEARN = False

try:
    from flask import Flask, jsonify, send_from_directory, request
    FLASK = True
except ImportError:
    FLASK = False

VERSION = "2.0.0"

# ── Movement type reference ───────────────────────────────────────────────────
MVT_LABELS = {
    "101": "GR from Purchase Order",  "102": "Reversal GR from PO",
    "109": "GR Blocked Stock",        "201": "GI to Cost Center",
    "202": "Reversal GI Cost Center", "261": "GI for Production Order",
    "262": "Reversal GI Production",  "311": "Transfer Storage Location",
    "321": "Release from QI Stock",   "343": "Block to Restricted",
    "344": "Restricted to Block",     "411": "Transfer to Unrestricted",
    "501": "GR w/o Purchase Order",   "541": "GI to Subcontractor",
    "542": "GR from Subcontractor",   "555": "GI Scrapping",
    "641": "GI Inter-plant Transfer", "681": "GI Stock Transfer (WM)",
    "685": "GR from Delivery (WM)",   "701": "Physical Inventory Posting",
    "702": "Physical Inventory Count",
}
GR_TYPES  = {"101","109","501","542","685"}
GI_TYPES  = {"201","261","541","555","681"}
XFER_TYPES= {"311","321","343","344","411","641"}

# ── Confirmed cross-file links from your actual data ─────────────────────────
CONFIRMED_LINKS = {
    "prq_to_order": ["20006722","20006790","20006796","20006810","20006835","20006857"],
    "shared_plant": "USD6",
    "shared_company": "US10",
    "shared_mrp": "PS1",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def excel_serial(s):
    try:
        v = float(s)
        if v > 40000:
            return datetime(1899, 12, 30) + timedelta(days=int(v))
    except:
        pass
    return None

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def normalize_po(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"\D", "", s)
    return digits if len(digits) >= 8 else None

def load_file(path):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_excel(path)

def json_safe(v):
    if pd.isna(v):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v

def field_profile(df, sample_size=5):
    prof = []
    total = len(df)
    for c in df.columns:
        s = df[c]
        nn = int(s.notna().sum())
        sample = [json_safe(v) for v in s.dropna().head(sample_size).tolist()]
        prof.append({
            "field": str(c),
            "non_null": nn,
            "null": int(total - nn),
            "coverage_pct": round((nn / max(total, 1)) * 100, 2),
            "dtype": str(s.dtype),
            "sample": sample,
        })
    return prof

def records_from_df(df, limit=200):
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.head(max(1, int(limit))).iterrows():
        out.append({str(k): json_safe(v) for k, v in row.to_dict().items()})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# EBAN LOADER
# ══════════════════════════════════════════════════════════════════════════════

class EBANLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        raw = load_file(self.path) if not self.path.endswith(".csv") else pd.read_csv(self.path, header=1, low_memory=False)
        if "Purchase Requisition" not in raw.columns:
            raw = pd.read_csv(self.path, header=1, low_memory=False)

        self.raw_df = raw.copy()
        df = raw[raw["Purchase Requisition"].apply(
            lambda x: str(x).strip().isdigit() if pd.notna(x) else False
        )].copy()

        df["req_date"]      = df["Requisition date"].apply(excel_serial)
        df["delivery_date"] = df["Delivery Date"].apply(excel_serial)
        df["creation_date"] = df["Creation Date"].apply(excel_serial) if "Creation Date" in df.columns else None
        df["qty"]           = safe_num(df["Quantity requested"]).fillna(0)
        df["price"]         = safe_num(df["Valuation Price"]).fillna(0)
        df["total_value"]   = df["qty"] * df["price"]
        df["po_num"]        = safe_num(df["Purchase order"])
        df["has_po"]        = df["po_num"] > 4_000_000_000
        if "Purchase Order Date" in df.columns:
            df["po_date"] = df["Purchase Order Date"].apply(excel_serial)
        else:
            df["po_date"] = pd.NaT
        df["po_key"] = df["Purchase order"].apply(normalize_po)
        df["is_deleted"]    = df["Deletion Indicator"].notna()
        df["plant"]         = df["Plant"].astype(str).str.strip()
        df["prq_id"]        = df["Purchase Requisition"].astype(str)
        df["is_prod_order_link"] = df["prq_id"].isin(CONFIRMED_LINKS["prq_to_order"])

        self.df = df
        return self

    def po_start_dates(self):
        """Normalized PO -> earliest PO date map for supplier turnaround computation."""
        d = self.df[self.df["po_key"].notna() & self.df["po_date"].notna()].copy()
        if d.empty:
            return {}
        m = d.groupby("po_key")["po_date"].min().to_dict()
        return {str(k): v for k, v in m.items() if isinstance(v, datetime)}

    def alerts(self):
        out = []
        for prq, grp in self.df.groupby("prq_id"):
            val = grp["total_value"].sum()
            pos = grp["Purchase order"].dropna().astype(str).unique().tolist()
            items = grp["Short Text"].dropna().tolist()
            deleted = int(grp["is_deleted"].sum())
            is_linked = bool(grp["is_prod_order_link"].any())
            date_r = grp["req_date"].dropna().min()
            risk = ("CRITICAL" if deleted > 0 and val > 0 else
                    "HIGH" if val > 5000 or (is_linked and deleted > 0) else
                    "MEDIUM" if val > 1000 else "LOW")
            out.append({
                "prq": prq, "items": [i for i in items if str(i) not in ("0","nan")][:3],
                "item_count": len(grp), "total_value": round(val, 2),
                "purchase_orders": [p for p in pos if p not in ("nan","0","X")],
                "deleted_count": deleted, "risk": risk,
                "req_date": date_r.strftime("%Y-%m-%d") if pd.notna(date_r) else "—",
                "is_prod_order_link": is_linked,
                "material_group": str(grp["Material Group"].dropna().iloc[0]) if grp["Material Group"].dropna().any() else "—",
                "status": str(grp["Processing status"].dropna().iloc[0]) if grp["Processing status"].dropna().any() else "—",
                "plant": str(grp["plant"].iloc[0]),
            })
        return sorted(out, key=lambda x: {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}[x["risk"]])

    def prq_table(self):
        prq = self.df[self.df["has_po"]].copy()
        rows = []
        for _, r in prq.iterrows():
            val = float(r["total_value"])
            rows.append({
                "prq": r["prq_id"], "item": str(r["Item of requisition"]),
                "description": str(r["Short Text"])[:52] if pd.notna(r["Short Text"]) else "—",
                "qty": int(r["qty"]), "price": round(float(r["price"]), 2),
                "total_value": round(val, 2),
                "req_date": r["req_date"].strftime("%Y-%m-%d") if pd.notna(r["req_date"]) else "—",
                "delivery_date": r["delivery_date"].strftime("%Y-%m-%d") if pd.notna(r.get("delivery_date")) else "—",
                "purchase_order": str(r["Purchase order"]) if pd.notna(r["Purchase order"]) else "—",
                "material_group": str(r["Material Group"]) if pd.notna(r["Material Group"]) else "—",
                "status": str(r["Processing status"]) if pd.notna(r["Processing status"]) else "—",
                "plant": str(r["plant"]),
                "is_deleted": bool(r["is_deleted"]),
                "is_prod_order_link": bool(r["is_prod_order_link"]),
                "risk": ("CRITICAL" if r["is_deleted"] else "HIGH" if val > 5000 else "MEDIUM" if val > 1000 else "LOW"),
            })
        return rows

    def summary(self):
        df = self.df
        proc = df[df["has_po"]]
        return {
            "total_rows": len(df), "unique_prqs": df["prq_id"].nunique(),
            "procurement_prqs": df[df["has_po"]]["prq_id"].nunique(),
            "total_value": round(proc["total_value"].sum(), 2),
            "deleted_count": int(df["is_deleted"].sum()),
            "confirmed_order_links": len(CONFIRMED_LINKS["prq_to_order"]),
            "plant": CONFIRMED_LINKS["shared_plant"],
            "high_risk_value": round(proc[proc["total_value"] > 3000]["total_value"].sum(), 2),
        }

    def source_profile(self):
        df = self.raw_df if hasattr(self, "raw_df") else self.df
        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "fields": field_profile(df),
        }

    def source_records(self, limit=200):
        df = self.raw_df if hasattr(self, "raw_df") else self.df
        return records_from_df(df, limit=limit)


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCTION ORDERS LOADER
# ══════════════════════════════════════════════════════════════════════════════

class OrdersLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        df = pd.read_csv(self.path, low_memory=False) if self.path.endswith(".csv") else pd.read_excel(self.path)
        self.raw_df = df.copy()
        df["start_date"]    = pd.to_datetime(df["Basic Start Date"], errors="coerce")
        df["finish_date"]   = pd.to_datetime(df["Basic finish date"], errors="coerce")
        df["actual_start"]  = pd.to_datetime(df["Actual start date"], errors="coerce")
        df["actual_release"]= pd.to_datetime(df.get("Actual release date"), errors="coerce")
        df["lead_days"]     = (df["finish_date"] - df["start_date"]).dt.days
        df["is_started"]    = df["actual_start"].notna()
        df["month"]         = df["start_date"].dt.to_period("M")
        df["quarter"]       = df["start_date"].dt.to_period("Q")
        df["week"]          = df["start_date"].dt.to_period("W")
        df["order_str"]     = df["Order"].astype(str)
        df["is_linked"]     = df["order_str"].isin(CONFIRMED_LINKS["prq_to_order"])
        self.df = df
        return self

    def monthly(self):
        m = self.df.groupby("month").size().reset_index(name="orders")
        return [[str(r["month"]), int(r["orders"])] for _, r in m.iterrows()]

    def quarterly(self):
        q = self.df.groupby("quarter").size().reset_index(name="orders")
        return [[str(r["quarter"]), int(r["orders"])] for _, r in q.iterrows()]

    def weekly(self):
        w = self.df.groupby("week").size().reset_index(name="orders")
        return [[str(r["week"].start_time.date()), int(r["orders"])] for _, r in w.iterrows()]

    def build_model(self):
        """
        Seasonality model: Ridge regression with Fourier features + rolling averages.
        Honest MAPE reported. Model is ready to swap in Prophet once data volume >= 36 months.
        """
        monthly = self.df.groupby("month").size().reset_index(name="orders")
        monthly["month_str"]  = monthly["month"].astype(str)
        monthly = monthly[monthly["month_str"] < datetime.now().strftime("%Y-%m")].copy()
        monthly["t"]          = range(len(monthly))
        monthly["month_num"]  = monthly["month"].apply(lambda p: p.month)
        monthly["sin_annual"] = np.sin(2 * np.pi * monthly["month_num"] / 12)
        monthly["cos_annual"] = np.cos(2 * np.pi * monthly["month_num"] / 12)
        monthly["sin_semi"]   = np.sin(4 * np.pi * monthly["month_num"] / 12)
        monthly["cos_semi"]   = np.cos(4 * np.pi * monthly["month_num"] / 12)
        monthly["rolling_3"]  = monthly["orders"].rolling(3, min_periods=1).mean().shift(1).fillna(monthly["orders"].mean())
        monthly["rolling_6"]  = monthly["orders"].rolling(6, min_periods=1).mean().shift(1).fillna(monthly["orders"].mean())

        features = ["t", "sin_annual", "cos_annual", "sin_semi", "cos_semi", "rolling_3", "rolling_6"]
        n = len(monthly)
        split = max(int(n * 0.8), n - 5)
        train, test = monthly.iloc[:split], monthly.iloc[split:]

        model = Ridge(alpha=2.0)
        model.fit(train[features].values, train["orders"].values)

        # Metrics on holdout
        if len(test) > 0:
            y_pred = np.maximum(model.predict(test[features].values), 0)
            y_true = test["orders"].values
            mape = float(mean_absolute_percentage_error(y_true, np.maximum(y_pred, 1)) * 100)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            r2   = float(r2_score(y_true, y_pred))
            mae  = float(np.mean(np.abs(y_true - y_pred)))
            holdout = [{"month": row["month_str"], "actual": int(row["orders"]), "predicted": int(max(0, p))}
                       for (_, row), p in zip(test.iterrows(), y_pred)]
        else:
            mape, rmse, r2, mae, holdout = 0.0, 0.0, 0.0, 0.0, []

        # Retrain on all data
        model.fit(monthly[features].values, monthly["orders"].values)
        last = monthly.iloc[-1]
        last_t, last_month = last["t"], last["month"]
        roll3 = monthly["orders"].tail(3).mean()
        roll6 = monthly["orders"].tail(6).mean()

        forecast = []
        for i in range(1, 13):
            fm   = last_month + i
            t_new= last_t + i
            mn   = fm.month
            row  = [t_new, np.sin(2*np.pi*mn/12), np.cos(2*np.pi*mn/12),
                    np.sin(4*np.pi*mn/12), np.cos(4*np.pi*mn/12), roll3, roll6]
            pred = max(0, int(model.predict([row])[0]))
            forecast.append({"month": str(fm), "orders": pred})
            roll3 = (roll3 * 2 + pred) / 3
            roll6 = (roll6 * 5 + pred) / 6

        # Actuals for chart
        actuals = [[str(r["month"]), int(r["orders"])] for _, r in monthly.iterrows()]

        return {
            "actuals": actuals,
            "forecast": forecast,
            "metrics": {"mape": round(mape, 1), "rmse": round(rmse, 1), "r2": round(r2, 3), "mae": round(mae, 1)},
            "holdout": holdout,
            "model_name": "Ridge + Fourier seasonality + rolling avg",
            "data_months": n,
            "train_months": split,
            "test_months": len(test),
            "model_note": (
                "MAPE > 20% — model is directional only. Upgrade to Prophet/SARIMA "
                "once 36+ months of data and Material-level COOIS export are available."
                if mape > 20 else
                "Model meets production threshold (MAPE < 20%). Suitable for buyer guidance."
            ),
            "production_ready": mape <= 20,
        }

    def lead_time(self):
        valid = self.df[(self.df["lead_days"].notna()) & (self.df["lead_days"] > 0) & (self.df["lead_days"] < 3000)]
        by_q = valid.groupby("quarter")["lead_days"].agg(["mean","median","std"]).reset_index()
        return {
            "overall_mean": int(valid["lead_days"].mean()) if len(valid) else 0,
            "overall_median": int(valid["lead_days"].median()) if len(valid) else 0,
            "overall_std": int(valid["lead_days"].std()) if len(valid) else 0,
            "by_quarter": [{"quarter": str(r["quarter"]), "mean": round(r["mean"], 1),
                            "median": round(r["median"], 1)} for _, r in by_q.iterrows()],
        }

    def summary(self):
        df = self.df
        valid = df[(df["lead_days"].notna()) & (df["lead_days"] > 0) & (df["lead_days"] < 3000)]
        monthly = df.groupby("month").size()
        return {
            "total_orders": len(df), "started": int(df["is_started"].sum()),
            "open": int((~df["is_started"]).sum()),
            "pct_open": round((~df["is_started"]).mean() * 100, 1),
            "avg_lead_days": int(valid["lead_days"].mean()) if len(valid) else 0,
            "median_lead_days": int(valid["lead_days"].median()) if len(valid) else 0,
            "peak_month": str(monthly.idxmax()) if len(monthly) else "—",
            "peak_count": int(monthly.max()) if len(monthly) else 0,
            "avg_monthly": int(monthly.mean()) if len(monthly) else 0,
            "date_start": df["start_date"].min().strftime("%b %Y") if df["start_date"].notna().any() else "—",
            "date_end": df["start_date"].max().strftime("%b %Y") if df["start_date"].notna().any() else "—",
            "mrp_controllers": df["MRP controller"].value_counts().to_dict() if "MRP controller" in df.columns else {},
            "confirmed_prq_links": len(CONFIRMED_LINKS["prq_to_order"]),
        }

    def source_profile(self):
        df = self.raw_df if hasattr(self, "raw_df") else self.df
        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "fields": field_profile(df),
        }

    def source_records(self, limit=200):
        df = self.raw_df if hasattr(self, "raw_df") else self.df
        return records_from_df(df, limit=limit)


# ══════════════════════════════════════════════════════════════════════════════
# MATERIAL DOCUMENT LOADER
# ══════════════════════════════════════════════════════════════════════════════

class MatDocLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        raw = pd.read_excel(self.path, header=1) if self.path.endswith((".xlsx",".xls")) else pd.read_csv(self.path, header=1, low_memory=False)
        self.raw_df = raw.copy()
        df = raw[safe_num(raw["Posting Date"]).notna()].copy()
        df["posting_date"] = df["Posting Date"].apply(excel_serial)
        df["amount"]   = safe_num(df["Amt.in Loc.Cur."]).fillna(0)
        df["qty"]      = safe_num(df["Quantity"]).fillna(0)
        df["stock_qty"]= safe_num(df["Stock Quantity"]).fillna(0)
        df["mvt"]      = df["Movement Type"].astype(str).str.strip()
        df["mvt_label"]= df["mvt"].map(MVT_LABELS).fillna(df["mvt"])
        df["is_gr"]    = df["mvt"].isin(GR_TYPES)
        df["is_gi"]    = df["mvt"].isin(GI_TYPES)
        df["is_xfer"]  = df["mvt"].isin(XFER_TYPES)
        df["month"]    = df["posting_date"].apply(lambda d: d.strftime("%Y-%m") if d else None)
        df["quarter"]  = df["posting_date"].apply(lambda d: f"{d.year}Q{((d.month-1)//3)+1}" if d else None)
        df["mat_str"]  = df["Material"].astype(str).str.strip()
        df["plant"]    = df["Plant"].astype(str).str.strip() if "Plant" in df.columns else "—"
        df["tc"]       = df["Transaction Code"].astype(str).str.strip() if "Transaction Code" in df.columns else "—"
        self.df = df
        return self

    def monthly_spend(self):
        m = self.df.groupby("month")["amount"].sum().reset_index()
        return [[r["month"], round(float(r["amount"]), 2)] for _, r in m[m["month"].notna()].sort_values("month").iterrows()]

    def gr_gi_monthly(self):
        gr = self.df[self.df["is_gr"]].groupby("month")["amount"].sum()
        gi = self.df[self.df["is_gi"]].groupby("month")["amount"].sum()
        months = sorted(set(gr.index) | set(gi.index))
        return [{"month": m, "gr": round(float(gr.get(m, 0)), 2), "gi": round(float(gi.get(m, 0)), 2)} for m in months if m]

    def material_intelligence(self):
        """Core inventory intelligence: excess risk, shortage risk, turnover, unit cost."""
        df = self.df
        mat = df[df["mat_str"].str.isnumeric()].copy()
        agg = mat.groupby("mat_str").agg(
            total_value=("amount", "sum"),
            gr_qty=("qty", lambda x: x[df.loc[x.index, "is_gr"]].sum()),
            gi_qty=("qty", lambda x: x[df.loc[x.index, "is_gi"]].sum()),
            gr_value=("amount", lambda x: x[df.loc[x.index, "is_gr"]].sum()),
            gi_value=("amount", lambda x: x[df.loc[x.index, "is_gi"]].sum()),
            txn_count=("amount", "count"),
            last_movement=("posting_date", "max"),
            first_movement=("posting_date", "min"),
        ).reset_index()

        agg["turnover_ratio"] = (agg["gi_qty"] / agg["gr_qty"].replace(0, np.nan)).fillna(0).clip(0, 10)
        agg["avg_unit_cost"]  = (agg["total_value"] / agg["txn_count"].replace(0, np.nan)).fillna(0)

        # Risk scoring
        v75 = agg["total_value"].quantile(0.75)
        agg["excess_risk"]   = (agg["total_value"] > v75) & (agg["turnover_ratio"] < 0.3)
        agg["shortage_risk"] = (agg["turnover_ratio"] > 0.7) & (agg["gr_qty"] < 5)
        agg["healthy"]       = ~agg["excess_risk"] & ~agg["shortage_risk"]
        agg["risk_label"]    = agg.apply(
            lambda r: "EXCESS" if r["excess_risk"] else ("SHORTAGE" if r["shortage_risk"] else "OK"), axis=1)

        # Days since last movement
        today = datetime.now()
        agg["days_since_last"] = agg["last_movement"].apply(
            lambda d: (today - d).days if isinstance(d, datetime) else 999)
        agg["slow_moving"] = agg["days_since_last"] > 180

        rows = []
        for _, r in agg.sort_values("total_value", ascending=False).head(25).iterrows():
            rows.append({
                "material": r["mat_str"],
                "total_value": round(float(r["total_value"]), 2),
                "gr_qty": round(float(r["gr_qty"]), 2),
                "gi_qty": round(float(r["gi_qty"]), 2),
                "gr_value": round(float(r["gr_value"]), 2),
                "gi_value": round(float(r["gi_value"]), 2),
                "txn_count": int(r["txn_count"]),
                "turnover_ratio": round(float(r["turnover_ratio"]), 3),
                "avg_unit_cost": round(float(r["avg_unit_cost"]), 2),
                "risk_label": r["risk_label"],
                "slow_moving": bool(r["slow_moving"]),
                "days_since_last": int(r["days_since_last"]),
                "last_movement": str(r["last_movement"])[:10] if isinstance(r["last_movement"], datetime) else "—",
            })
        return rows

    def spend_intelligence(self):
        spend = self.df.groupby("month")["amount"].sum()
        avg   = float(spend.mean())
        std   = float(spend.std())
        spikes = {m: round(float(v), 2) for m, v in spend.items() if v > avg + std and m}
        return {
            "monthly_avg": round(avg, 2),
            "monthly_std": round(std, 2),
            "spike_months": spikes,
            "spike_count": len(spikes),
            "total_spend": round(float(self.df["amount"].sum()), 2),
        }

    def movement_breakdown(self):
        m = self.df.groupby("mvt_label").agg(count=("amount","count"), total=("amount","sum"), qty=("qty","sum")).reset_index().sort_values("count", ascending=False)
        return [{"label": r["mvt_label"], "count": int(r["count"]), "amount": round(float(r["total"]),2), "qty": round(float(r["qty"]),2)} for _, r in m.iterrows()]

    def po_summary(self):
        df = self.df[self.df["Purchase order"].notna()]
        p = df.groupby("Purchase order").agg(amount=("amount","sum"), qty=("qty","sum"), docs=("Material Document","count")).reset_index().sort_values("amount", ascending=False)
        return [{"po": str(r["Purchase order"]), "amount": round(float(r["amount"]),2), "qty": round(float(r["qty"]),2), "docs": int(r["docs"])} for _, r in p.head(15).iterrows()]

    def tc_breakdown(self):
        return {str(k): int(v) for k, v in self.df["tc"].value_counts().head(15).items()}

    def supplier_intelligence(self, po_start_dates=None):
        """Build supplier KPIs only when usable supplier data exists in MATDOC extract."""
        df = self.df.copy()

        # Prefer explicit supplier columns from MATDOC export; pandas may suffix duplicates with .1/.2
        supplier_candidates = [
            "Supplier.1", "Supplier.2", "Supplier.3", "Supplier",
            "Supplier for Special Stock.1", "Supplier for Special Stock",
        ]
        sup_col = next((c for c in supplier_candidates if c in df.columns), None)
        if not sup_col:
            return {"available": False, "reason": "No supplier column in source data", "suppliers": []}

        sup = df[sup_col].astype(str).str.strip()
        sup = sup.replace({"": np.nan, "nan": np.nan, "None": np.nan, "Supplier": np.nan, "0": np.nan})
        df["supplier_id"] = sup
        df = df[df["supplier_id"].notna()].copy()
        if df.empty:
            return {"available": False, "reason": "Supplier column exists but has no usable values", "suppliers": []}

        # Location proxy (only if populated in this extract)
        loc_col = None
        for c in ["Unloading Point", "Incoterms Location 1", "Goods Recipient", "Plant"]:
            if c in df.columns:
                non_empty = df[c].dropna().astype(str).str.strip()
                non_empty = non_empty[(non_empty != "") & (non_empty.str.lower() != c.lower())]
                if len(non_empty) >= 5:
                    loc_col = c
                    break

        # Completion signal from SAP if available
        has_completion = "Delivery Completed" in df.columns
        if has_completion:
            df["delivery_completed"] = df["Delivery Completed"].astype(str).str.strip().str.upper().eq("X")
        else:
            df["delivery_completed"] = False

        df["po_str"] = df["Purchase order"].astype(str).str.strip() if "Purchase order" in df.columns else ""
        df["has_po"] = df["po_str"].str.match(r"^\d{8,}$", na=False)

        recs = []
        po_start_dates = po_start_dates or {}

        for sid, grp in df.groupby("supplier_id"):
            docs = int(len(grp))
            po_grp = grp[grp["has_po"]].groupby("po_str") if grp["has_po"].any() else None

            if po_grp is not None:
                po_total = int(po_grp.ngroups)
                po_success = 0
                po_cycles = []
                true_tat_cycles = []
                proxy_tat_cycles = []
                tat_source = "insufficient"
                for po, pg in po_grp:
                    ok = bool(pg["is_gr"].any() or pg["delivery_completed"].any())
                    if ok:
                        po_success += 1
                    dates = pg["posting_date"].dropna()

                    # Preferred: PO date from EBAN to first GR posting in MATDOC
                    gr_dates = pg[pg["is_gr"]]["posting_date"].dropna()
                    po_key = normalize_po(po)
                    po_start = po_start_dates.get(po_key) if po_key else None
                    if isinstance(po_start, datetime) and len(gr_dates) > 0:
                        first_gr = gr_dates.min()
                        if first_gr >= po_start:
                            true_tat_cycles.append((first_gr - po_start).days)

                    # Fallback: activity span inside MATDOC for that PO
                    if len(dates) >= 2:
                        proxy_tat_cycles.append((dates.max() - dates.min()).days)

                if true_tat_cycles:
                    po_cycles = true_tat_cycles
                    tat_source = "po_date_to_first_gr"
                elif proxy_tat_cycles:
                    po_cycles = proxy_tat_cycles
                    tat_source = "posting_span_proxy"
                else:
                    po_cycles = []
                success_rate = round((po_success / max(po_total, 1)) * 100, 1)
                avg_tat = round(float(np.mean(po_cycles)), 1) if po_cycles else None
                order_count = po_total
            else:
                # Fallback when PO linkage unavailable: GR share as success proxy
                success_rate = round((grp["is_gr"].sum() / max(len(grp), 1)) * 100, 1)
                avg_tat = None
                order_count = 0
                tat_source = "insufficient"

            total_amt = round(float(grp["amount"].sum()), 2)
            gr_amt = float(grp[grp["is_gr"]]["amount"].sum())
            gr_share = (gr_amt / max(abs(total_amt), 1.0)) * 100.0

            if avg_tat is None:
                tat_score = 55.0
            else:
                tat_score = max(0.0, 100.0 - min(avg_tat, 90.0) * 1.1)
            reliability = round(max(0.0, min(100.0, success_rate * 0.65 + gr_share * 0.1 + tat_score * 0.25)), 1)

            location = None
            if loc_col:
                loc = grp[loc_col].dropna().astype(str).str.strip()
                loc = loc[(loc != "") & (loc.str.lower() != loc_col.lower())]
                if not loc.empty:
                    location = loc.value_counts().index[0]

            recs.append({
                "supplier": sid,
                "orders": int(order_count),
                "docs": docs,
                "order_success_rate": success_rate,
                "turnaround_days": avg_tat,
                "turnaround_source": tat_source,
                "reliability_score": reliability,
                "total_amount": total_amt,
                "location": location,
            })

        recs = sorted(recs, key=lambda x: (x["reliability_score"], -x["total_amount"]))
        return {
            "available": True,
            "supplier_field": sup_col,
            "location_field": loc_col,
            "suppliers": recs[:25],
            "summary": {
                "supplier_count": len(recs),
                "avg_success_rate": round(float(np.mean([r["order_success_rate"] for r in recs])), 1) if recs else 0,
                "avg_reliability": round(float(np.mean([r["reliability_score"] for r in recs])), 1) if recs else 0,
                "with_location": int(sum(1 for r in recs if r.get("location"))),
                "tat_true_count": int(sum(1 for r in recs if r.get("turnaround_source") == "po_date_to_first_gr")),
                "tat_proxy_count": int(sum(1 for r in recs if r.get("turnaround_source") == "posting_span_proxy")),
            }
        }

    def summary(self):
        df = self.df
        return {
            "total_docs": len(df),
            "total_amount": round(float(df["amount"].sum()), 2),
            "gr_amount": round(float(df[df["is_gr"]]["amount"].sum()), 2),
            "gi_amount": round(float(df[df["is_gi"]]["amount"].sum()), 2),
            "xfer_amount": round(float(df[df["is_xfer"]]["amount"].sum()), 2),
            "unique_materials": int(df["mat_str"].nunique()),
            "unique_pos": int(df["Purchase order"].dropna().nunique()),
            "date_start": str(min(d for d in df["posting_date"] if d))[:10],
            "date_end": str(max(d for d in df["posting_date"] if d))[:10],
            "plant": CONFIRMED_LINKS["shared_plant"],
            "top_tc": str(df["tc"].value_counts().index[0]) if len(df) > 0 else "—",
            "excess_risk_materials": int(df[df["mat_str"].str.isnumeric()].groupby("mat_str")["amount"].sum().gt(df[df["mat_str"].str.isnumeric()].groupby("mat_str")["amount"].sum().quantile(0.75)).sum()),
        }

    def source_profile(self):
        df = self.raw_df if hasattr(self, "raw_df") else self.df
        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "fields": field_profile(df),
        }

    def source_records(self, limit=200):
        df = self.raw_df if hasattr(self, "raw_df") else self.df
        return records_from_df(df, limit=limit)


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSOR — orchestrates all three sources + insights engine
# ══════════════════════════════════════════════════════════════════════════════

class AI4ProcureProcessor:
    def __init__(self, eban_path=None, orders_path=None, matdoc_path=None):
        self.eban_path   = eban_path
        self.orders_path = orders_path
        self.matdoc_path = matdoc_path
        self.eban = self.orders = self.matdoc = None
        self._payload = None

    def source_catalog(self):
        cat = {
            "strategy": {
                "default_mode": "source-first",
                "join_policy": "Join only for explicit analytics (alerts, linkage, supplier TAT). Keep source views independent by default.",
                "notes": [
                    "EBAN, Orders, MATDOC are retained as independent datasets with full field coverage.",
                    "Cross-source joins are selective and declared in payload.linkage + supplier_intelligence.",
                ],
            },
            "sources": {}
        }
        if self.eban:
            cat["sources"]["eban"] = self.eban.source_profile()
        if self.orders:
            cat["sources"]["orders"] = self.orders.source_profile()
        if self.matdoc:
            cat["sources"]["matdoc"] = self.matdoc.source_profile()
        return cat

    def source_records(self, source_name, limit=200):
        source_name = (source_name or "").lower()
        if source_name == "eban" and self.eban:
            return self.eban.source_records(limit=limit)
        if source_name in {"orders", "coois"} and self.orders:
            return self.orders.source_records(limit=limit)
        if source_name in {"matdoc", "inventory"} and self.matdoc:
            return self.matdoc.source_records(limit=limit)
        return []

    def load(self):
        if self.eban_path and Path(self.eban_path).exists():
            self.eban = EBANLoader(self.eban_path).load()
            print(f"  ✓ EBAN  — {self.eban.summary()['total_rows']} rows, plant {CONFIRMED_LINKS['shared_plant']}")
        else:
            print("  ⚠ EBAN  — not found, using demo data")

        if self.orders_path and Path(self.orders_path).exists():
            self.orders = OrdersLoader(self.orders_path).load()
            print(f"  ✓ Orders— {self.orders.summary()['total_orders']} orders, {self.orders.summary()['date_start']}–{self.orders.summary()['date_end']}")
        else:
            print("  ⚠ Orders— not found, using demo data")

        if self.matdoc_path and Path(self.matdoc_path).exists():
            self.matdoc = MatDocLoader(self.matdoc_path).load()
            print(f"  ✓ MatDoc— {self.matdoc.summary()['total_docs']} docs, ${self.matdoc.summary()['total_amount']:,.0f} total value")
        else:
            print("  ⚠ MatDoc— not found, using demo data")

        return self

    def build(self):
        p = {}

        # ── EBAN
        if self.eban:
            p["eban"] = {
                "summary": self.eban.summary(),
                "alerts": self.eban.alerts(),
                "prq_table": self.eban.prq_table(),
            }
        else:
            p["eban"] = self._demo_eban()

        # ── Orders + model
        if self.orders:
            model = self.orders.build_model() if SKLEARN else {"forecast": [], "metrics": {}, "production_ready": False}
            p["orders"] = {
                "summary": self.orders.summary(),
                "monthly": self.orders.monthly(),
                "quarterly": self.orders.quarterly(),
                "weekly": self.orders.weekly()[-52:],
                "model": model,
                "lead_time": self.orders.lead_time(),
            }
        else:
            p["orders"] = self._demo_orders()

        # ── MatDoc
        if self.matdoc:
            po_start_dates = self.eban.po_start_dates() if self.eban else {}
            p["matdoc"] = {
                "summary": self.matdoc.summary(),
                "monthly_spend": self.matdoc.monthly_spend(),
                "gr_gi_monthly": self.matdoc.gr_gi_monthly(),
                "material_intelligence": self.matdoc.material_intelligence(),
                "movement_breakdown": self.matdoc.movement_breakdown(),
                "po_summary": self.matdoc.po_summary(),
                "tc_breakdown": self.matdoc.tc_breakdown(),
                "spend_intelligence": self.matdoc.spend_intelligence(),
                "supplier_intelligence": self.matdoc.supplier_intelligence(po_start_dates=po_start_dates),
            }
        else:
            p["matdoc"] = self._demo_matdoc()

        # ── Cross-file linkage map
        p["linkage"] = {
            "confirmed_links": CONFIRMED_LINKS,
            "prq_to_order_count": len(CONFIRMED_LINKS["prq_to_order"]),
            "shared_plant": CONFIRMED_LINKS["shared_plant"],
            "shared_company": CONFIRMED_LINKS["shared_company"],
            "join_gap": {
                "material_null_in_orders": True,
                "reservation_mismatch": True,
                "po_overlap_eban_matdoc": False,
                "recommendation": "Re-export COOIS with Material field. Use RESB table for Reservation join.",
            },
        }

        # ── Source-first catalog (all fields retained, joins only where needed)
        p["source_catalog"] = self.source_catalog()

        # ── Insights engine
        p["insights"] = self._insights(p)
        p["meta"] = {
            "version": VERSION, "generated_at": datetime.now().isoformat(),
            "data_sources": {
                "eban": bool(self.eban), "orders": bool(self.orders), "matdoc": bool(self.matdoc)
            }
        }

        self._payload = p
        return p

    def _insights(self, p):
        insights = []
        os = p.get("orders", {}).get("summary", {})
        ms = p.get("matdoc", {}).get("summary", {})
        model = p.get("orders", {}).get("model", {})
        spend_intel = p.get("matdoc", {}).get("spend_intelligence", {})
        mats = p.get("matdoc", {}).get("material_intelligence", [])

        open_pct = os.get("pct_open", 0)
        if open_pct > 70:
            insights.append({"type": "warning", "priority": 1,
                "title": f"{open_pct:.0f}% of production orders not started",
                "body": f"{os.get('open',0):,} orders have no actual start date. PRQ deletion risk cannot be confirmed for these orders."})

        spikes = spend_intel.get("spike_months", {})
        if spikes:
            worst = max(spikes, key=spikes.get)
            insights.append({"type": "alert", "priority": 2,
                "title": f"Spend spike: {worst} (${spikes[worst]:,.0f})",
                "body": f"This month was {spikes[worst]/max(spend_intel.get('monthly_avg',1),1):.1f}× the monthly average. Investigate for bulk orders or data anomalies."})

        excess = [m for m in mats if m["risk_label"] == "EXCESS"]
        if excess:
            excess_val = sum(m["total_value"] for m in excess)
            insights.append({"type": "alert", "priority": 1,
                "title": f"{len(excess)} materials flagged for excess inventory",
                "body": f"${excess_val:,.0f} in high-value materials with low consumption (turnover < 0.3). Review for write-down or reallocation."})

        slow = [m for m in mats if m["slow_moving"]]
        if slow:
            insights.append({"type": "warning", "priority": 3,
                "title": f"{len(slow)} slow-moving materials (>180 days inactive)",
                "body": "These materials have had no inventory movement in over 6 months. Flag for obsolescence review."})

        if model.get("metrics", {}).get("mape", 100) > 20:
            insights.append({"type": "info", "priority": 4,
                "title": f"Forecast model: MAPE {model['metrics'].get('mape','—')}% — directional only",
                "body": f"Only {model.get('data_months',0)} months of data available. Upgrade to Prophet once 36+ months and Material-level COOIS export are ready. Phase 1 alerts are fully reliable."})

        insights.append({"type": "gap", "priority": 5,
            "title": "Material field null in production orders",
            "body": "Re-export via SAP COOIS with Material number to unlock per-SKU forecasting. This is the single highest-value data fix."})

        avg_lead = os.get("avg_lead_days", 0)
        if avg_lead > 200:
            insights.append({"type": "warning", "priority": 3,
                "title": f"Average lead time {avg_lead} days — above 200-day threshold",
                "body": "Long supplier lead times increase PRQ deletion risk window. Buyers need earlier visibility into demand changes."})

        return sorted(insights, key=lambda x: x["priority"])

    def export(self, path="ai4procure_data.json"):
        if not self._payload: self.build()
        with open(path, "w") as f:
            json.dump(self._payload, f, indent=2, default=str)
        print(f"  ✓ Exported → {path}")
        return path

    # ── Demo data fallbacks ───────────────────────────────────────────────────
    def _demo_eban(self):
        return {
            "summary": {"total_rows":95,"unique_prqs":12,"procurement_prqs":2,"total_value":28985.40,"deleted_count":0,"confirmed_order_links":6,"plant":"USD6","high_risk_value":21038.86},
            "alerts": [
                {"prq":"10672166","items":["HMND 1418BW20 N12 Freestanding Enclosure","HMND 90BWFW Inner Panel Full Height"],"item_count":2,"total_value":4197.84,"purchase_orders":["4500195093"],"deleted_count":0,"risk":"HIGH","req_date":"2024-06-13","is_prod_order_link":False,"material_group":"1710120","status":"B","plant":"USD6"},
                {"prq":"10795703","items":["AP11 Heater Board Test Fixture","AP11 New P/S Prototype","AP11 OMRON KIT"],"item_count":5,"total_value":9138.86,"purchase_orders":["4500191357","4500198996","4500200336","4500201081"],"deleted_count":0,"risk":"HIGH","req_date":"2024-04-25","is_prod_order_link":False,"material_group":"1750100","status":"B","plant":"USD6"},
                {"prq":"20006722","items":["Production Order linked PRQ"],"item_count":1,"total_value":0,"purchase_orders":[],"deleted_count":0,"risk":"MEDIUM","req_date":"2024-03-27","is_prod_order_link":True,"material_group":"—","status":"—","plant":"USD6"},
            ],
            "prq_table": [
                {"prq":"10672166","item":"1510","description":"HMND 1418BW20 N12 Freestanding Enclosure","qty":1,"price":3254.95,"total_value":3254.95,"req_date":"2024-06-13","delivery_date":"—","purchase_order":"4500195093","material_group":"1710120","status":"B","plant":"USD6","is_deleted":False,"is_prod_order_link":False,"risk":"HIGH"},
                {"prq":"10672166","item":"1520","description":"HMND 90BWFW Inner Panel Full Height","qty":1,"price":942.89,"total_value":942.89,"req_date":"2024-06-13","delivery_date":"—","purchase_order":"4500195093","material_group":"1710120","status":"B","plant":"USD6","is_deleted":False,"is_prod_order_link":False,"risk":"LOW"},
                {"prq":"10672166","item":"1530","description":"Resistor Panels & Materials","qty":1,"price":3871.90,"total_value":3871.90,"req_date":"2024-06-26","delivery_date":"—","purchase_order":"4500196059","material_group":"1750100","status":"B","plant":"USD6","is_deleted":False,"is_prod_order_link":False,"risk":"MEDIUM"},
                {"prq":"10795703","item":"420","description":"AP11 Heater Board Test Fixture","qty":2,"price":1690.00,"total_value":3380.00,"req_date":"2024-04-25","delivery_date":"—","purchase_order":"4500191357","material_group":"1750100","status":"B","plant":"USD6","is_deleted":False,"is_prod_order_link":False,"risk":"MEDIUM"},
                {"prq":"10795703","item":"430","description":"AP11 New P/S Prototype","qty":4,"price":3023.38,"total_value":12093.52,"req_date":"2024-08-01","delivery_date":"—","purchase_order":"4500198996","material_group":"3430999","status":"B","plant":"USD6","is_deleted":False,"is_prod_order_link":False,"risk":"HIGH"},
                {"prq":"10795703","item":"440","description":"LF-SW008104 WIREEL_AP11 PANEL LEFT 12C","qty":8,"price":25.55,"total_value":204.40,"req_date":"2024-08-16","delivery_date":"—","purchase_order":"4500200336","material_group":"1750100","status":"B","plant":"USD6","is_deleted":False,"is_prod_order_link":False,"risk":"LOW"},
                {"prq":"10795703","item":"450","description":"LF-SW008069 WIREEL_AP11 PANEL I/O LEFT","qty":8,"price":21.28,"total_value":170.24,"req_date":"2024-08-16","delivery_date":"—","purchase_order":"4500200336","material_group":"1750100","status":"B","plant":"USD6","is_deleted":False,"is_prod_order_link":False,"risk":"LOW"},
                {"prq":"10795703","item":"460","description":"AP11 OMRON KIT","qty":10,"price":506.75,"total_value":5067.50,"req_date":"2024-08-26","delivery_date":"—","purchase_order":"4500201081","material_group":"1750100","status":"B","plant":"USD6","is_deleted":False,"is_prod_order_link":False,"risk":"HIGH"},
            ]
        }

    def _demo_orders(self):
        actuals = [["2024-03",22],["2024-04",38],["2024-05",49],["2024-06",38],["2024-07",49],["2024-08",73],["2024-09",54],["2024-10",56],["2024-11",103],["2024-12",36],["2025-01",20],["2025-02",57],["2025-03",72],["2025-04",87],["2025-05",266],["2025-06",54],["2025-07",40],["2025-08",63],["2025-09",106],["2025-10",78],["2025-11",80],["2025-12",61],["2026-01",119],["2026-02",95]]
        return {
            "summary": {"total_orders":1730,"started":449,"open":1281,"pct_open":74.0,"avg_lead_days":216,"median_lead_days":193,"peak_month":"2025-05","peak_count":266,"avg_monthly":69,"date_start":"Mar 2024","date_end":"Mar 2026","mrp_controllers":{"PS1":1728,"001":2},"confirmed_prq_links":6},
            "monthly": actuals,
            "quarterly": [["2024Q1",22],["2024Q2",125],["2024Q3",176],["2024Q4",195],["2025Q1",149],["2025Q2",407],["2025Q3",209],["2025Q4",219],["2026Q1",228]],
            "weekly": [],
            "model": {"actuals": actuals, "forecast":[{"month":"2026-03","orders":121},{"month":"2026-04","orders":129},{"month":"2026-05","orders":135},{"month":"2026-06","orders":137},{"month":"2026-07","orders":135},{"month":"2026-08","orders":130},{"month":"2026-09","orders":125},{"month":"2026-10","orders":121},{"month":"2026-11","orders":121},{"month":"2026-12","orders":125},{"month":"2027-01","orders":133},{"month":"2027-02","orders":143}],"metrics":{"mape":42.3,"rmse":36.9,"r2":-2.6,"mae":32.5},"holdout":[],"model_name":"Ridge + Fourier seasonality + rolling avg","data_months":24,"train_months":19,"test_months":5,"model_note":"MAPE > 20% — model is directional only. Upgrade to Prophet/SARIMA once 36+ months of data and Material-level COOIS export are available.","production_ready":False},
            "lead_time": {"overall_mean":216,"overall_median":193,"overall_std":181,"by_quarter":[{"quarter":"2024Q1","mean":181.2,"median":174},{"quarter":"2024Q2","mean":236.7,"median":214},{"quarter":"2024Q3","mean":230.4,"median":210},{"quarter":"2024Q4","mean":228.5,"median":209},{"quarter":"2025Q1","mean":193.4,"median":182},{"quarter":"2025Q2","mean":243.3,"median":220},{"quarter":"2025Q3","mean":195.8,"median":185},{"quarter":"2025Q4","mean":179.5,"median":168}]},
        }

    def _demo_matdoc(self):
        return {
            "summary": {"total_docs":243,"total_amount":289258.06,"gr_amount":198650.76,"gi_amount":37871.55,"xfer_amount":52735.75,"unique_materials":126,"unique_pos":40,"date_start":"2024-01-02","date_end":"2026-03-20","plant":"USD6","top_tc":"CO27","excess_risk_materials":8},
            "monthly_spend": [["2024-01",3433.11],["2024-02",9066.77],["2024-03",138.10],["2024-04",4194.39],["2024-05",482.14],["2024-06",6600.99],["2024-07",19765.06],["2024-08",6139.93],["2024-09",2249.08],["2024-10",32728.14],["2024-11",637.14],["2024-12",3938.54],["2025-01",1364.64],["2025-02",0.0],["2025-03",37590.46],["2025-04",9247.44],["2025-05",29847.18],["2025-06",107.23],["2025-07",32202.60],["2025-08",219.26],["2025-09",21754.50],["2025-10",3.70],["2025-11",348.63],["2025-12",43014.50],["2026-01",5136.34],["2026-03",19048.19]],
            "gr_gi_monthly": [{"month":"2024-01","gr":725.0,"gi":595.0},{"month":"2024-02","gr":725.0,"gi":595.0},{"month":"2024-04","gr":2118.94,"gi":0},{"month":"2024-07","gr":17151.95,"gi":900.0},{"month":"2024-08","gr":4343.74,"gi":900.0},{"month":"2024-09","gr":2249.95,"gi":0},{"month":"2024-10","gr":29826.60,"gi":0},{"month":"2025-03","gr":29826.60,"gi":7065.0},{"month":"2025-04","gr":9247.44,"gi":0},{"month":"2025-05","gr":29826.60,"gi":0},{"month":"2025-07","gr":29878.99,"gi":7065.0},{"month":"2025-09","gr":21754.50,"gi":0},{"month":"2025-12","gr":40397.50,"gi":0},{"month":"2026-01","gr":5136.34,"gi":0},{"month":"2026-03","gr":19048.19,"gi":0}],
            "material_intelligence": [{"material":"2312458","total_value":29878.99,"gr_qty":1.0,"gi_qty":0.0,"gr_value":29878.99,"gi_value":0,"txn_count":1,"turnover_ratio":0.0,"avg_unit_cost":29878.99,"risk_label":"EXCESS","slow_moving":False,"days_since_last":40,"last_movement":"2026-02-18"},{"material":"2310125","total_value":29826.60,"gr_qty":1.0,"gi_qty":0.0,"gr_value":29826.60,"gi_value":0,"txn_count":1,"turnover_ratio":0.0,"avg_unit_cost":29826.60,"risk_label":"EXCESS","slow_moving":False,"days_since_last":45,"last_movement":"2026-02-13"},{"material":"238951","total_value":28899.64,"gr_qty":1.0,"gi_qty":0.0,"gr_value":28899.64,"gi_value":0,"txn_count":1,"turnover_ratio":0.0,"avg_unit_cost":28899.64,"risk_label":"EXCESS","slow_moving":False,"days_since_last":28,"last_movement":"2026-03-02"},{"material":"2304795","total_value":21754.50,"gr_qty":25.0,"gi_qty":0.0,"gr_value":21754.50,"gi_value":0,"txn_count":1,"turnover_ratio":0.0,"avg_unit_cost":21754.50,"risk_label":"EXCESS","slow_moving":False,"days_since_last":192,"last_movement":"2025-09-19"},{"material":"2310196","total_value":15008.00,"gr_qty":0.0,"gi_qty":10.0,"gr_value":0,"gi_value":15008.00,"txn_count":2,"turnover_ratio":0.0,"avg_unit_cost":7504.00,"risk_label":"EXCESS","slow_moving":False,"days_since_last":110,"last_movement":"2025-12-10"},{"material":"183573","total_value":8187.00,"gr_qty":100.0,"gi_qty":0.0,"gr_value":8187.00,"gi_value":0,"txn_count":1,"turnover_ratio":0.0,"avg_unit_cost":8187.00,"risk_label":"EXCESS","slow_moving":True,"days_since_last":242,"last_movement":"2025-07-31"},{"material":"178240","total_value":7065.00,"gr_qty":0.0,"gi_qty":1.0,"gr_value":0,"gi_value":7065.00,"txn_count":1,"turnover_ratio":0.0,"avg_unit_cost":7065.00,"risk_label":"OK","slow_moving":False,"days_since_last":155,"last_movement":"2025-10-27"},{"material":"160707","total_value":6717.00,"gr_qty":20.0,"gi_qty":0.0,"gr_value":6717.00,"gi_value":0,"txn_count":1,"turnover_ratio":0.0,"avg_unit_cost":6717.00,"risk_label":"EXCESS","slow_moving":True,"days_since_last":242,"last_movement":"2025-07-31"},{"material":"2312157","total_value":5610.02,"gr_qty":25.0,"gi_qty":4.0,"gr_value":5285.00,"gi_value":325.02,"txn_count":2,"turnover_ratio":0.16,"avg_unit_cost":2805.01,"risk_label":"EXCESS","slow_moving":False,"days_since_last":90,"last_movement":"2025-12-31"},{"material":"2312369","total_value":5006.76,"gr_qty":1.0,"gi_qty":0.0,"gr_value":5006.76,"gi_value":0,"txn_count":2,"turnover_ratio":0.0,"avg_unit_cost":2503.38,"risk_label":"EXCESS","slow_moving":True,"days_since_last":210,"last_movement":"2025-09-02"}],
            "movement_breakdown": [{"label":"GI for Production Order","count":118,"amount":19574.44,"qty":413.8},{"label":"GR from Delivery (WM)","count":29,"amount":49103.40,"qty":138.0},{"label":"GR from Purchase Order","count":26,"amount":149547.36,"qty":910.0},{"label":"Transfer to Unrestricted","count":10,"amount":1688.86,"qty":298.0},{"label":"Release from QI Stock","count":9,"amount":0,"qty":62.0},{"label":"GI Stock Transfer","count":9,"amount":17701.60,"qty":346.0},{"label":"GR w/o Purchase Order","count":7,"amount":0,"qty":433.0},{"label":"Physical Inventory","count":4,"amount":3048.54,"qty":1423.6}],
            "po_summary": [{"po":"4700102114","amount":40397.50,"qty":143.0,"docs":1},{"po":"4500186459","amount":29421.50,"qty":225.0,"docs":2},{"po":"4800085741","amount":21754.50,"qty":25.0,"docs":1},{"po":"4800090427","amount":16659.72,"qty":18.0,"docs":5},{"po":"4800091504","amount":9247.44,"qty":8.0,"docs":8},{"po":"4500185796","amount":8187.00,"qty":100.0,"docs":1},{"po":"4500171829","amount":6717.00,"qty":20.0,"docs":1},{"po":"4500150263","amount":5006.76,"qty":2.0,"docs":2}],
            "tc_breakdown": {"CO27":65,"COGI":35,"MIGO_TR":30,"VLPOD":29,"MIGO_GR":23,"MIGO_GI":19,"CO11N":12,"VL02N":10,"QA11":9,"MI07":5},
            "spend_intelligence": {"monthly_avg":11124.54,"monthly_std":13563.12,"spike_months":{"2024-10":32728.14,"2025-03":37590.46,"2025-05":29847.18,"2025-07":32202.60,"2025-12":43014.50},"spike_count":5,"total_spend":289258.06},
            "supplier_intelligence": {"available": False, "reason": "Demo mode does not include supplier-level source columns", "suppliers": []},
        }


# ══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════════════════

def create_app(processor):
    app = Flask(__name__)
    payload = processor.build()
    base_dir = Path(__file__).parent
    assets_dir = base_dir / "assets"

    @app.route("/")
    def index():
        html = base_dir / "ai4procure_dashboard.html"
        return html.read_text(encoding="utf-8", errors="replace") if html.exists() else "<h1>AI4Procure v2</h1><p>Place ai4procure_dashboard.html in the same directory.</p>"

    @app.route("/SemiDeep.png")
    def semideep_logo():
        return send_from_directory(base_dir, "SemiDeep.png")

    # Linux hosts are case-sensitive; keep both routes to avoid broken logo URLs.
    @app.route("/semideep.png")
    def semideep_logo_lower():
        return send_from_directory(base_dir, "SemiDeep.png")

    @app.route("/assets/<path:filename>")
    def assets_file(filename):
        return send_from_directory(assets_dir, filename)

    @app.route("/api/data")           
    def data():           return jsonify(payload)
    @app.route("/api/summary")        
    def summary():        return jsonify({"eban": payload["eban"]["summary"], "orders": payload["orders"]["summary"], "matdoc": payload["matdoc"]["summary"], "linkage": payload["linkage"], "insights": payload["insights"], "meta": payload["meta"], "source_catalog": payload.get("source_catalog", {})})
    @app.route("/api/catalog")
    def catalog():        return jsonify(payload.get("source_catalog", processor.source_catalog()))
    @app.route("/api/source/<source_name>")
    def source_data(source_name):
        limit = request.args.get("limit", default=200, type=int)
        return jsonify({"source": source_name, "limit": limit, "rows": processor.source_records(source_name, limit=limit)})
    @app.route("/api/alerts")         
    def alerts():         return jsonify(payload["eban"]["alerts"])
    @app.route("/api/forecast")       
    def forecast():       return jsonify(payload["orders"]["model"])
    @app.route("/api/materials")      
    def materials():      return jsonify(payload["matdoc"]["material_intelligence"])
    @app.route("/api/inventory")      
    def inventory():      return jsonify({"gr_gi": payload["matdoc"]["gr_gi_monthly"], "spend": payload["matdoc"]["monthly_spend"], "movements": payload["matdoc"]["movement_breakdown"]})
    @app.route("/api/suppliers")
    def suppliers():      return jsonify(payload["matdoc"].get("supplier_intelligence", {"available": False, "suppliers": []}))
    @app.route("/api/linkage")        
    def linkage():        return jsonify(payload["linkage"])
    @app.route("/api/health")         
    def health():         return jsonify({"status": "ok", "version": VERSION, "generated_at": payload["meta"]["generated_at"]})

    return app


def _build_processor_from_env():
    """Build processor for WSGI servers (Gunicorn) using env/common file names."""
    eban_path = os.getenv("EBAN_PATH")
    orders_path = os.getenv("ORDERS_PATH")
    matdoc_path = os.getenv("MATDOC_PATH")

    if not eban_path and os.path.exists("Headers_xlsx_Sheet1.csv"):
        eban_path = "Headers_xlsx_Sheet1.csv"
    if not orders_path and os.path.exists("Book2_xlsx_Sheet1.csv"):
        orders_path = "Book2_xlsx_Sheet1.csv"
    if not matdoc_path and os.path.exists("Book5.xlsx"):
        matdoc_path = "Book5.xlsx"

    # Prefer real data when present, otherwise use demo mode so service still boots.
    if eban_path or orders_path or matdoc_path:
        return AI4ProcureProcessor(
            eban_path=eban_path,
            orders_path=orders_path,
            matdoc_path=matdoc_path,
        ).load()

    return AI4ProcureProcessor().load()


if FLASK:
    try:
        _wsgi_processor = _build_processor_from_env()
        app = create_app(_wsgi_processor)
    except Exception as exc:
        # Last-resort fallback so platform health checks still succeed with a clear message.
        app = Flask(__name__)

        @app.route("/")
        def _startup_error_index():
            return (
                "<h1>AI4Procure startup error</h1>"
                f"<pre>{exc}</pre>"
                "<p>Set EBAN_PATH / ORDERS_PATH / MATDOC_PATH or include default files.</p>"
            ), 500

        @app.route("/api/health")
        def _startup_error_health():
            return jsonify({"status": "error", "message": str(exc), "version": VERSION}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="AI4Procure v2 — Supply Chain Intelligence")
    p.add_argument("--eban",   default=None)
    p.add_argument("--orders", default=None)
    p.add_argument("--matdoc", default=None)
    p.add_argument("--port",   type=int, default=int(os.getenv("PORT", 5000)))
    p.add_argument("--host",   default="0.0.0.0")
    p.add_argument("--export", action="store_true")
    p.add_argument("--debug",  action="store_true")
    p.add_argument("--demo",   action="store_true")
    args = p.parse_args()

    print(f"\n{'='*62}")
    print(f"  AI4Procure v{VERSION} — Supply Chain Procurement Intelligence")
    print(f"  Phase 1 + Phase 2  |  SAP EBAN + COOIS + MATDOC")
    print(f"{'='*62}\n")

    # Auto-detect data files from CLI args or environment
    eban_path = args.eban or os.getenv("EBAN_PATH")
    orders_path = args.orders or os.getenv("ORDERS_PATH")
    matdoc_path = args.matdoc or os.getenv("MATDOC_PATH")
    
    # Try common file names if not provided
    if not eban_path and os.path.exists("Headers_xlsx_Sheet1.csv"):
        eban_path = "Headers_xlsx_Sheet1.csv"
    if not orders_path and os.path.exists("Book2_xlsx_Sheet1.csv"):
        orders_path = "Book2_xlsx_Sheet1.csv"
    if not matdoc_path and os.path.exists("Book5.xlsx"):
        matdoc_path = "Book5.xlsx"

    processor = AI4ProcureProcessor(
        eban_path   = None if args.demo else eban_path,
        orders_path = None if args.demo else orders_path,
        matdoc_path = None if args.demo else matdoc_path,
    ).load()

    if args.export or not FLASK:
        processor.build()
        processor.export()
        return

    app = create_app(processor)
    print(f"\n  Server: http://{args.host}:{args.port}")
    print(f"  Endpoints: /api/data  /api/summary  /api/catalog")
    print(f"             /api/source/<eban|orders|matdoc>?limit=200")
    print(f"             /api/alerts  /api/forecast  /api/materials")
    print(f"             /api/inventory  /api/suppliers  /api/linkage\n")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
