# 🛍️ Synthetic Grocery Supply Chain Data Generator  
> **High-Fidelity Synthetic Engine for Inventory Optimization & Demand Forecasting**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/roberto-balbinotti)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/robertobalbinotti/synthetic-grocery-data)
---

## 🎯 Context & Strategic Objectives

In the retail (grocery) sector, the scarcity of clean historical data or the confidentiality of real data hinders the agile development of AI models. This project fills that gap by providing a **Digital Twin** of the supply chain, simulating complex operations and enabling **Machine Learning model testing** in demand forecasting and inventory optimization scenarios.  

**Main objectives:**
1. **Massive data generation:** Foundation for the AI project [Smart Supply Chain AI](https://github.com/rbalbinotti/smart-supply-chain-ai).  
2. **Technical portfolio:** Showcase proficiency in data engineering, time-series modeling, and Python-based pipeline development.  

---

## 🔬 Methodology & Statistical Rigor

The simulation follows **Time Series Decomposition** principles, modeling demand $D(t)$ as a multivariate function:

$$D(t) = T(t) + S(t) + \sum \beta_i X_i(t) + \epsilon$$

- **$T(t)$**: Deterministic growth trend.  
- **$S(t)$:** Weekly and annual seasonality.  
- **$X_i(t)$:** Exogenous variables (price, real INMET weather, holidays).  
- **$\epsilon$:** Gaussian noise simulating market uncertainties.  

### Technical Differentiator: Real Weather Data
Unlike common synthetic generators, this project incorporates **real meteorological data** (INMET/BDMEP), enriched with feature engineering to map climate severity and capture real correlations between temperature and perishable demand.  

---

## ✨ Pipeline Components

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

## 📊 Structure & Final Result

The final dataset is saved in **Parquet** format for high performance, containing **100,192 rows and 29 columns**.  

Sample:

| received_date | product | category | sub_category | shelf_life_days | supplier | distance_km | temp_class | precip_class | wind_class | is_holiday | sales_demand | sales_volume | stock_qty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-02-04 | Egg (Chicken) | Dairy | Eggs | 28 | FreshEggs Co. | 65 | Warm | No precip. | Gentle Breeze | False | High | 318 | 1096 |
| 2023-01-03 | Sugar | Pantry | Baking | 730 | Wholesale | 25 | Warm | No precip. | Gentle Breeze | False | High | 10 | 33 |

> **Note:** The complete `grocery_data.parquet` file with all **100,192 rows** is available for download on **[Kaggle](https://www.kaggle.com/datasets/robertobalbinotti/synthetic-grocery-data)**.

---

## 🛠️ Data Engineering & MLOps

- **Modularization:** Logic separated into `create_data_functions.py` and `weather_conditions.py`.  
- **Optimized format:** `.parquet` for Big Data pipelines.  
- **Deployment-ready:** Dockerfile for environment isolation.  
- **Dependency management:** `pyproject.toml` with PDM.  

---

## 📂 Directory Structure

```text
.
├── create_data_functions.py
├── data
│   ├── external
│   ├── processed
│   └── raw
├── Dockerfile
├── LICENSE
├── pdm.lock
├── pyproject.toml
├── README.md
├── README_PT.md
├── synthetic_grocery.ipynb
└── weather_conditions.py
```

---

## 📚 Stack & References

- **Core:** `Pandas`, `NumPy`, `Scikit-Learn`, `fastparquet`.  
- **Statistics:** `holidays`, `workalendar`.  
- **Weather source:** Real data from [**INMET/BDMEP**](https://bdmep.inmet.gov.br/)
- **Associated project:** [Smart supply Chain AI](https://github.com/rbalbinotti/smart-supply-chain-ai) 
---

*Developed by **Roberto Rosário Balbinotti** – ML Architect & Data Specialist.*  
E-mail: rbalbinotti@gmail.com

---
