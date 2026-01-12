

## 🛍️ Synthetic Grocery Supply Chain Data Generator

### Author: Roberto Rosário Balbinotti
Contact:
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/roberto-balbinotti)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:rbalbinotti@gmail.com)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/robertobalbinotti/synthetic-grocery-data)

### 🎯 Project Overview

This project consists of a robust pipeline for generating a high-fidelity synthetic dataset to simulate complex operations within a supermarket supply chain (**Grocery Supply Chain**).

The goal is to create a feature-rich database suitable for developing and testing Machine Learning models in demand forecasting and inventory optimization scenarios.

---

### 🚀 Main Objectives

1. **Data Mass Generation:** Serve as the primary data source for the supply chain artificial intelligence project: **[`smart-supply-chain-ai`](https://github.com/rbalbinotti/smart-supply-chain-ai)**.
2. **Technical Portfolio:** Demonstrate proficiency in data engineering, complex time-series modeling, and building data pipelines in Python.

---

### ✨ Dataset Features and Methodology

The synthetic dataset is a time series that incorporates real-world complexities to challenge forecasting models. The methodology encompasses the simulation of internal trends and the inclusion of exogenous variables.

#### Time Series Components (`create_data_functions.py` module)

* **Base Time Series:** Creation of a base DataFrame with dates (`ds`), unique IDs, and the target value (demand or sales).
* **Trend and Seasonality:** Inclusion of growth components (trend) and annual and weekly cyclic patterns (seasonality) to simulate sales behavior over time.
* **Lag Features:** Utilizes the `LagFeatureCreator` (a scikit-learn compatible transformer) to add lagged features (e.g., sales from the previous week or month) to capture temporal dependencies.
* **Holidays and Events:** Inclusion of impacts from events such as holidays and promotions, which increase or decrease demand on specific dates.
* **Price Feature:** Adds a price column as an exogenous variable, simulating an inversely proportional relationship to demand (price increases generally reduce demand).

#### Weather Exogenous Variables - **Real Data** (`weather_conditions.py` module)

The original weather data consists of **real data** collected via INMET. They are processed to be used as predictive features:

* **Temperature:** Classified into categories such as "Very Cold", "Mild to Temperate", and "Hot".
* **Precipitation:** Classified by intensity, from "No precipitation" to "Violent Rainfall".
* **Wind:** Classified by speed.
* **Seasonal Simulation:** The module also includes logic to simulate weather conditions based on months and seasons of the year.

---

### 📊 Final Result Sample

The final generated dataset is a Parquet file with **100,192 rows and 29 columns** (at the time of the notebook execution). It integrates all catalog, weather, and time-series information.

Below is a detailed sample:

|  | received_date | lpo | in_season | product | product_id | category | sub_category | shelf_life_days | max_days_on_sale | unit | supplier_rating | supplier | supplier_id | distance_km | moq | storage | temp_class | precip_class | wind_class | weather_severity | day_class | is_holiday | is_weekend | sales_demand | sales_volume | lead_time | min_stock | max_stock | stock_qty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **89076** | 2025-02-04 | 2025-02-01 | False | Egg (Chicken) | 1135024 | P | Dairy | Eggs | 28 | 14 | unit | 3 | FreshEggs Co. | 1716146 | S | 65 | 30 | Refrig. | Warm | No precip. | Gentle Breeze | Moderate | Weekdays | False | False | High | 318 | 4 | 1096 |
| **76409** | 2023-01-03 | 2022-12-31 | False | Sugar | 1510142 | P | Pantry | Baking | 730 | 180 | lb | 4 | Wholesale | 1471068 | S | 25 | 300 | Room Temp | Warm | No precip. | Gentle Breeze | Moderate | Weekdays | False | False | High | 10 | 3 | 33 |
| **85256** | 2025-07-29 | 2025-07-23 | False | Oatmeal Biscuit | 1703249 | P | Pantry | Snacks | 90 | 30 | unit | 5 | Biscuit Boutique | 1339404 | S | 92 | 65 | Room Temp | Cool | Light Rain | Gentle Breeze | Moderate | Weekdays | False | False | Normal | 124 | 4 | 856 |
| **97832** | 2024-08-16 | 2024-08-11 | False | Shrimp | 1896390 | P | Fresh | Seafood | 2 | 1 | lb | 4 | OceanHarvest | 1838472 | S | 180 | 40 | Refrig. | Cool | No precip. | Gentle Breeze | Moderate | Weekdays | False | False | Normal | 16 | 6 | 155 |
| **31962** | 2023-07-15 | 2023-07-10 | False | Canned Tuna | 1106138 | P | Pantry | Canned Fish | 1095 | 90 | unit | 4 | PantryEssentials | 1344994 | S | 95 | 130 | Room Temp | Cold | Light Rain | Light Breeze | Moderate | Saturday | False | True | High | 10 | 6 | 56 |

> **Note:** The complete `grocery_data.parquet` file with all **100,192 rows** is available for download on **[Kaggle](https://www.kaggle.com/datasets/robertobalbinotti/synthetic-grocery-data)**.

---

### 🛠️ Usage (Installation and Execution)

To reproduce the data generation, follow the steps below:

#### 1. Clone the Repository

```bash
git clone [YOUR_REPOSITORY_URL]
cd [REPOSITORY_NAME]

```

#### 2. Install Dependencies

This project requires essential libraries for data manipulation, time series, and file system operations. Using a virtual environment (`venv` or `conda`) is recommended.

**Main Dependencies:**

* `pandas`
* `numpy`
* `fastparquet`
* `holidays`
* `workalendar`

You can install the dependencies via `pip`:

```bash
pip install pandas numpy fastparquet holidays workalendar jupyter

```

#### 3. Directory Structure

The pipeline expects the following folder structure to exist:

```
.
├── synthetic_grocery.ipynb
├── create_data_functions.py
├── weather_conditions.py
└── data
    ├── raw/
    ├── external/
    └── processed/

```

Create the `data` folder and its subdirectories before running the notebook.

#### 4. Run the Pipeline

Open the `synthetic_grocery.ipynb` notebook and execute all cells in sequence. The process will:

1. Generate product catalog and supplier data.
2. Apply feature engineering functions (trend, seasonality, events, price, and weather).
3. Save the final dataset in the `data/processed/` folder as `grocery_data.parquet`.

```bash
jupyter notebook synthetic_grocery.ipynb

```

---

### 📚 References and Sources

This project was built using the following libraries and sources:

| Resource | Description |
| --- | --- |
| **AI Author/Expert** | Roberto Rosário Balbinotti |
| **Main Tools** | `Pandas`, `NumPy`, `fastparquet` |
| **Weather Data** | **Real data**, collected from the **INMET/BDMEP** website ([https://bdmep.inmet.gov.br/](https://bdmep.inmet.gov.br/)). |
| **Time Series** | `holidays` and `workalendar` (Used to generate event features in the `create_data_functions.py` module). |
| **Weather Modeling** | Custom functions and feature classification in the `weather_conditions.py` module. |
| **Associated Project** | This dataset is the foundation for the Machine Learning project [`smart-supply-chain-ai`](https://github.com/rbalbinotti/smart-supply-chain-ai). |

