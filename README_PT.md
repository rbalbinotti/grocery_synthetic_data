## 🛍️ Synthetic Grocery Supply Chain Data Generator
### Autor: Roberto Rosário Balbinotti
Contato:  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/roberto-balbinotti)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:rbalbinotti@gmail.com)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/robertobalbinotti/synthetic-grocery-data)

### 🎯 Visão Geral do Projeto

Este projeto consiste em um pipeline robusto para a geração de um *dataset* sintético de alta fidelidade para simular operações complexas de uma cadeia de suprimentos de supermercado (*Grocery Supply Chain*).

O objetivo é criar uma base de dados rica em features, adequada para o desenvolvimento e teste de modelos de Machine Learning em cenários de previsão de demanda e otimização de estoque.

---

### 🚀 Objetivos Principais

1. **Criação de Massa de Dados:** Servir como a principal fonte de dados para o projeto de inteligência artificial na cadeia de suprimentos: [**`smart-supply-chain-ai`**](https://github.com/rbalbinotti/smart-supply-chain-ai).
2. **Portfólio Técnico:** Demonstrar proficiência na engenharia de dados, modelagem de séries temporais complexas e criação de *pipelines* de dados em Python.

---

### ✨ Features do Dataset e Metodologia

O conjunto de dados sintético é uma série temporal que incorpora complexidades reais para desafiar modelos de previsão. A metodologia abrange a simulação de tendências internas e a inclusão de variáveis exógenas.

#### Componentes de Séries Temporais (Módulo `create_data_functions.py`)

* **Série Temporal Base:** Criação de um DataFrame base com datas (`ds`), IDs exclusivos e o valor alvo (demanda ou vendas).
* **Tendência e Sazonalidade:** Inclusão de componentes de crescimento (tendência) e padrões cíclicos anuais e semanais (sazonalidade) para simular o comportamento de vendas ao longo do tempo.
* **Recursos de Lag:** Utiliza o `LagFeatureCreator` (um transformador compatível com scikit-learn) para adicionar recursos defasados (por exemplo, vendas da última semana ou mês) para capturar dependências temporais.
* **Feriados e Eventos:** Inclusão de impactos de eventos como feriados e promoções, que aumentam ou diminuem a demanda em datas específicas.
* **Feature de Preço:** Adiciona uma coluna de preço como variável exógena, simulando uma relação inversamente proporcional à demanda (o aumento do preço geralmente reduz a demanda).

#### Variáveis Exógenas Meteorológicas - **Dados Reais** (Módulo `weather_conditions.py`)

Os dados meteorológicos originais são **dados reais**, coletados através do INMET. Eles são processados para serem usados como *features* preditivas:

* **Temperatura:** Classificada em categorias como "Very Cold", "Mild to Temperate", "Hot".
* **Precipitação:** Classificada por intensidade, de "No precipitation" a "Violent Rainfall".
* **Vento:** Classificado pela velocidade.
* **Simulação Sazonal:** O módulo também inclui lógica para simular condições climáticas com base nos meses e estações do ano.

---

### 📊 Amostra do Resultado Final

O dataset final gerado é um arquivo Parquet com **100.192 linhas e 29 colunas** (no momento da execução do notebook). Ele integra todas as informações de catálogo, clima e séries temporais.

Abaixo, uma amostra detalhada:

| | received_date | lpo | in_season | product | product_id | category | sub_category | shelf_life_days | max_days_on_sale | unit | supplier_rating | supplier | supplier_id | distance_km | moq | storage | temp_class | precip_class | wind_class | weather_severity | day_class | is_holiday | is_weekend | sales_demand | sales_volume | lead_time | min_stock | max_stock | stock_qty |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **89076** | 2025-02-04 | 2025-02-01 | False | Egg (Chicken) | 1135024\|P | Dairy | Eggs | 28 | 14 | unit | 3 | FreshEggs Co. | 1716146\|S | 65 | 30 | Refrig. | Warm | No precip. | Gentle Breeze | Moderate | Weekdays | False | False | High | 318 | 4 | 1096 | 1370 | 1491 |
| **76409** | 2023-01-03 | 2022-12-31 | False | Sugar | 1510142\|P | Pantry | Baking | 730 | 180 | lb | 4 | Wholesale | 1471068\|S | 25 | 300 | Room Temp | Warm | No precip. | Gentle Breeze | Moderate | Weekdays | False | False | High | 10 | 3 | 33 | 300 | 412 |
| **85256** | 2025-07-29 | 2025-07-23 | False | Oatmeal Biscuit | 1703249\|P | Pantry | Snacks | 90 | 30 | unit | 5 | Biscuit Boutique | 1339404\|S | 92 | 65 | Room Temp | Cool | Light Rain | Gentle Breeze | Moderate | Weekdays | False | False | Normal | 124 | 4 | 856 | 1070 | 1148 |
| **97832** | 2024-08-16 | 2024-08-11 | False | Shrimp | 1896390\|P | Fresh | Seafood | 2 | 1 | lb | 4 | OceanHarvest | 1838472\|S | 180 | 40 | Refrig. | Cool | No precip. | Gentle Breeze | Moderate | Weekdays | False | False | Normal | 16 | 6 | 155 | 186 | 184 |
| **31962** | 2023-07-15 | 2023-07-10 | False | Canned Tuna | 1106138\|P | Pantry | Canned Fish | 1095 | 90 | unit | 4 | PantryEssentials | 1344994\|S | 95 | 130 | Room Temp | Cold | Light Rain | Light Breeze | Moderate | Saturday | False | True | High | 10 | 6 | 56 | 130 | 61 |


> **Nota:** O arquivo completo `grocery_data.parquet` com todas as **100.192 linhas** está disponível para download no [**Kaggle**](https://www.kaggle.com/datasets/robertobalbinotti/synthetic-grocery-data).

---

### 🛠️ Como Utilizar (Instalação e Execução)

Para reproduzir a geração de dados, siga os passos abaixo:

#### 1. Clonar o Repositório

```bash
git clone [URL_DO_SEU_REPOSITORIO]
cd [NOME_DO_REPOSITORIO]
```

#### 2. Instalar Dependências

Este projeto requer as bibliotecas essenciais para manipulação de dados, séries temporais e operações de *file system*. Recomenda-se o uso de um ambiente virtual (`venv` ou `conda`).

**Dependências Principais:**

* `pandas`
* `numpy`
* `fastparquet`
* `holidays`
* `workalendar`

Você pode instalar as dependências via `pip`:

```bash
pip install pandas numpy fastparquet holidays workalendar jupyter

```

#### 3. Estrutura de Diretórios

O *pipeline* espera que a seguinte estrutura de pastas exista:

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

Crie a pasta `data` e seus subdiretórios antes de executar o *notebook*.

#### 4. Executar o Pipeline

Abra o notebook `synthetic_grocery.ipynb` e execute todas as células em sequência. O processo irá:

1. Gerar dados de catálogo de produtos e fornecedores.
2. Aplicar as funções de engenharia de features (tendência, sazonalidade, eventos, preço e clima).
3. Salvar o conjunto de dados final na pasta `data/processed/` com o nome `grocery_data.parquet`.

```bash
jupyter notebook synthetic_grocery.ipynb

```

---

### 📚 Referências e Fontes

Este projeto foi construído utilizando as seguintes bibliotecas e fontes:

| Recurso | Descrição |
| --- | --- |
| **Autor/Especialista em IA** | Roberto Rosário Balbinotti |
| **Ferramentas Principais** | `Pandas`, `NumPy`, `fastparquet` |
| **Dados Meteorológicos** | **Reais**, coletados do site **INMET/BDMEP** ([https://bdmep.inmet.gov.br/](https://bdmep.inmet.gov.br/)). |
| **Séries Temporais** | `holidays` e `workalendar` (Utilizadas para gerar *features* de eventos no módulo `create_data_functions.py`) |
| **Modelagem de Clima** | Funções personalizadas e classificação de features no módulo `weather_conditions.py` |
| **Projeto Associado** | O *dataset* é a base para o projeto de Machine Learning [`smart-supply-chain-ai`](https://github.com/rbalbinotti/smart-supply-chain-ai). |

---

