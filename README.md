# AI4Procure — DS&S NA Procurement Intelligence Platform
**Version 1.0.0 · Phase 1 + Phase 2 · Pilot: Aug 2025 – Feb 2026**

## Overview
AI4Procure is a procurement intelligence tool for DS&S NA Procurement, built on three SAP data sources:
- **File 1 (EBAN)** — Purchase Requisitions
- **File 2 (COOIS)** — Production Orders
- **File 3 (MATDOC)** — Material Documents (Inventory Movements)

### Capabilities
| Panel | Description |
|-------|-------------|
| Dashboard | KPIs, order trend, spend overview, insights |
| PRQ Alerts | Real-time deleted PRQ risk detection |
| PRQ × Orders | Cross-match table with risk scoring |
| Alert Email | Auto-generated buyer alert email |
| Forecast | Order demand prediction (linear → Prophet upgrade path) |
| Inventory Moves | GR/GI analysis, movement types, PO spend |
| Materials | Per-material inventory analysis (126 unique SKUs) |
| Buyer Actions | 8-step action guide across Phase 1 & Phase 2 |

---

## Local Setup

```bash
# 1. Clone / place files in a directory
cd ai4procure/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run with your data files
python ai4procure.py \
  --eban    path/to/Headers_xlsx_Sheet1.csv \
  --orders  path/to/Book2_xlsx_Sheet1.csv \
  --matdoc  path/to/Book5.xlsx \
  --port    5000

# 4. Open browser
open http://localhost:5000

# Run as standalone HTML (no server needed — uses embedded fallback data)
open ai4procure_dashboard.html
```

## Export JSON Only (no server)
```bash
python ai4procure.py \
  --eban path/to/eban.csv \
  --orders path/to/orders.csv \
  --matdoc path/to/matdoc.xlsx \
  --export
# Writes: ai4procure_data.json
```

---

## Cloud Deployment

### Option A — Heroku
```bash
heroku create ai4procure-dss
git init && git add . && git commit -m "AI4Procure v1.0.0"
git push heroku main
heroku open
```

### Option B — Railway
```bash
# Push repo to GitHub, then:
# railway.app → New Project → Deploy from GitHub → select repo
# Set env var: PORT=8080
```

### Option C — Render
```bash
# render.com → New Web Service → connect GitHub repo
# Build command: pip install -r requirements.txt
# Start command: gunicorn ai4procure:app --workers 2 --bind 0.0.0.0:$PORT
```

### Option D — Azure App Service
```bash
az webapp up --name ai4procure-dss --runtime PYTHON:3.11 --sku B1
```

### Option E — AWS Elastic Beanstalk
```bash
eb init ai4procure --platform python-3.11
eb create ai4procure-prod
eb deploy
```

### Docker
```bash
docker build -t ai4procure .
docker run -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -e EBAN_PATH=/app/data/eban.csv \
  -e ORDERS_PATH=/app/data/orders.csv \
  -e MATDOC_PATH=/app/data/matdoc.xlsx \
  ai4procure
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard HTML |
| `GET /api/data` | Full JSON payload |
| `GET /api/summary` | KPI summary only |
| `GET /api/alerts` | PRQ alerts |
| `GET /api/forecast` | Order forecast |
| `GET /api/materials` | Material summary |
| `GET /api/health` | Health check |

---

## File Format Reference

### File 1 — EBAN (Purchase Requisitions)
- Format: CSV exported from SAP (headers on row 2)
- Key columns: Purchase Requisition, Item of requisition, Short Text, Material, Quantity requested, Valuation Price, Requisition date, Purchase order, Deletion Indicator, Processing status, Material Group

### File 2 — Production Orders (COOIS)
- Format: CSV exported from SAP COOIS
- Key columns: Order, Basic Start Date, Basic finish date, Actual start date, Total Order Quantity, MRP controller, Reservation, Scheduling type
- **Known issue:** Material column is currently 100% null — re-export needed for Phase 2

### File 3 — Material Documents (MATDOC)
- Format: Excel (.xlsx), headers on row 2
- Key columns: Material Document, Posting Date, Movement Type, Material, Quantity, Amt.in Loc.Cur., Purchase order, Transaction Code, Debit/Credit ind

---

## Known Data Gaps & Roadmap

| Issue | Impact | Resolution |
|-------|--------|------------|
| Material null in COOIS | Blocks product-level forecast | Re-export SAP COOIS with material |
| Reservation mismatch (EBAN vs Orders) | Breaks PRQ-order join | SAP RESB table linkage |
| May 2025 spike (266 orders) | Inflates linear model | Root cause investigation needed |
| Linear model R²=0.41 | Limited forecast accuracy | Upgrade to Prophet/SARIMA |

---

## Project Info
- **Project Manager:** Kendra Valton
- **Scope:** DS&S NA direct procurement (global rollout if pilot successful)
- **SAP S/4 ERP Integration:** 2025
- **Mobile App Rollout:** 2026
- **Budget:** TBD
