# 🛍️ Synthetic Grocery Supply Chain Data Generator  
> **High-Fidelity Synthetic Engine for Inventory Optimization & Demand Forecasting**  
> *Now with **Pandas** and **Polars** implementations – choose your performance level.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/roberto-balbinotti)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/robertobalbinotti/synthetic-grocery-data)

---

## 🎯 Context & Strategic Objectives

In the retail (grocery) sector, the scarcity of clean historical data or the confidentiality of real data hinders the agile development of AI models. This project fills that gap by providing a **Digital Twin** of the supply chain, simulating complex operations and enabling **Machine Learning model testing** in demand forecasting and inventory optimization scenarios.

**Main objectives:**
1. **Massive data generation:** Foundation for the AI project [Smart Supply Chain AI](https://github.com/rbalbinotti/smart-supply-chain-ai).
2. **Technical portfolio:** Showcase proficiency in data engineering, time-series modeling, and Python-based pipeline development.

---

## ⚡ Choose Your Engine: Pandas vs. Polars

This repository offers **two complete implementations** of the data generation pipeline, each tailored to different performance and realism requirements:

| Feature | Pandas Implementation | Polars Implementation |
|---------|----------------------|------------------------|
| **Notebook** | `synthetic_grocery.ipynb` | `synthetic_grocery_polars.ipynb` |
| **Row count** | ~100,192 rows | ~300,605 rows (scalable) |
| **Final columns** | 30 columns | 31 columns |
| **Performance** | Single-threaded, standard memory | Multi-threaded, Arrow-backed, lazy evaluation |
| **Best for** | Prototyping, smaller datasets | Production-grade, large-scale simulation |
| **Key libraries** | `pandas`, `numpy` | `polars`, `fastparquet` |

Both versions share the same modular logic (functions in `create_data_functions.py` and `weather_conditions.py`) but differ in implementation details, resulting in **different schemas and feature richness**.

---

## 📊 Schema Comparison & Key Additions in Polars

The Polars implementation introduces several enhancements that make the dataset more realistic and detailed:

### Pandas Schema (30 columns)
- Standard columns: `received_date`, `product_id`, `product`, `category`, `sub_category`, `shelf_life_days`, `maximum_days_on_sale`, `seasonality`, `storage_recommendation`, `unit_of_measurement`, `supplier_id`, `supplier`, `supplier_rating`, `distance_km`, `moq`, `in_season`, `is_holiday`, `day_classification`, `is_weekend`, `sales_demand`, `sales_volume`, `delivery_days`, `min_stock`, `max_stock`, `stock_quantity`, `temperature_classification`, `precipitation_classification`, `wind_classification`, `weather_severity`, plus `lpo` and `lead_time` (in intermediate steps).

### Polars Schema (31 columns – final output)
- Standard columns: `order_purchase_date`, `received_date`, `product_id`, `product`, `category`, `sub_category`, `sales_demand`, `sales_volume`, `seasonality`, `storage_recommendation`, `unit_of_measurement`, `shelf_life_days`, `maximum_days_on_sale`, `supplier_id`, `supplier`, `supplier_rating`, `distance_km`, `moq`, `delivery_days`, `transit_time`, `in_season`, `is_holiday`, `day_classification`, `is_weekend`, `min_stock`, `max_stock`, `stock_quantity`, `temperature_classification`, `precipitation_classification`, `wind_classification`, `weather_severity`.


### ✨ Key Enhancements in Polars
- **Granular Road Segmentation:** The pipeline explicitly models **urban_km**, **highway_km**, and **off_road_km** based on product category and distance, then simulates realistic speed distributions for each segment to compute **transit_time**.
- **Safety Stock Logic:** `min_stock` and `max_stock` are calculated using lead time variability and supplier reliability ratings, following a service-level approach (95% with safety factors).
- **Purchase Order Simulation:** The `order_purchase_date` is derived from supplier rating, seasonality, and delivery days, providing a realistic procurement timeline.
- **Quantization:** Numeric columns are cast to smaller integer/float types (UInt16, Float16) to optimize memory and storage without losing significant precision.

---

## 🔬 Methodology & Statistical Rigor

The simulation follows **Time Series Decomposition** principles, modeling demand $D(t)$ as a multivariate function:

$$D(t) = T(t) + S(t) + \sum \beta_i X_i(t) + \epsilon$$

- **$T(t)$**: Deterministic growth trend.
- **$S(t)$:** Weekly and annual seasonality.
- **$X_i(t)$:** Exogenous variables (price, real INMET weather, holidays).
- **$\epsilon$:** Gaussian noise simulating market uncertainties.

### Technical Differentiator: Real Weather Data
Unlike common synthetic generators, this project incorporates **real meteorological data** from INMET/BDMEP, enriched with feature engineering to map climate severity and capture real correlations between temperature, precipitation, wind, and perishable demand.

---

## ✨ Pipeline Components (Shared Logic)

### Time Series (`create_data_functions.py`)
- **Base series:** DataFrame with dates (`ds`), IDs, and target values (demand/sales).
- **Trend & seasonality:** Growth and weekly/annual cycles.
- **Lag features:** `LagFeatureCreator` adds temporal dependencies (e.g., previous week’s sales).
- **Events & holidays:** Impacts from promotions and special dates.
- **Price:** Inverse relationship between price and demand.

### Exogenous Weather Variables (`weather_conditions.py`)
- **Temperature:** Classified into ranges (Very Cold, Temperate, Hot).
- **Precipitation:** Intensity (No rain → Violent rainfall).
- **Wind:** Classified by speed.
- **Seasonal simulation:** Adjustments based on months and seasons.

---

## 🛠️ Data Engineering & MLOps

- **Modularization:** Core logic is shared across both implementations via `create_data_functions.py` and `weather_conditions.py`.
- **Optimized format:** Output saved as `.parquet` for Big Data pipelines.
- **Deployment-ready:** Dockerfile for environment isolation.
- **Dependency management:** `pyproject.toml` with **UV** (fast, modern Python package manager) – replaces Conda for lighter, faster setups.

---

## 📂 Directory Structure

```text
.
├── create_data_functions.py          # Shared core logic (both versions)
├── weather_conditions.py             # Weather classification module
├── data
│   ├── external                      # INMET weather CSV
│   ├── processed                     # Output Parquet files
│   └── raw                           # JSON catalogs (products, suppliers)
├── synthetic_grocery.ipynb           # Pandas implementation
├── synthetic_grocery_polars.ipynb    # Polars high-performance implementation
├── Dockerfile
├── pyproject.toml                    # UV-managed dependencies
├── uv.lock                           # Locked dependency versions
├── LICENSE
├── README.md
└── README_PT.md
```

---

## 📚 Stack & References

- **Core (Pandas):** `pandas`, `numpy`, `scikit-learn`, `fastparquet`.
- **Core (Polars):** `polars` – leverages Arrow and parallel execution.
- **Statistics:** `holidays`, `workalendar`.
- **Weather source:** Real data from [**INMET/BDMEP**](https://bdmep.inmet.gov.br/).
- **Associated project:** [Smart Supply Chain AI](https://github.com/rbalbinotti/smart-supply-chain-ai).

---

## 🚀 Getting Started

Clone the repository and install the dependencies (choose your preferred engine):

```bash
git clone https://github.com/rbalbinotti/synthetic-grocery-data.git
cd synthetic-grocery-data

# Using UV (recommended)
uv sync

# Or with pip
pip install -e .
```

Then open the respective notebook and run all cells.

---

*Developed by **Roberto Rosário Balbinotti** – ML Architect & Data Specialist.*  
E-mail: rbalbinotti@gmail.com