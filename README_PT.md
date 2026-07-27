# 🛍️ Gerador de Dados Sintéticos para Cadeia de Suprimentos de Supermercados  
> **Motor Sintético de Alta Fidelidade para Otimização de Estoque & Previsão de Demanda**  
> *Agora com implementações em **Pandas** e **Polars** – escolha seu nível de desempenho.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/roberto-balbinotti)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/robertobalbinotti/synthetic-grocery-data)

---

## 🎯 Contexto & Objetivos Estratégicos

No setor varejista (supermercados), a escassez de dados históricos limpos ou a confidencialidade dos dados reais dificultam o desenvolvimento ágil de modelos de IA. Este projeto preenche essa lacuna ao fornecer um **Gêmeo Digital** da cadeia de suprimentos, simulando operações complexas e permitindo **testes de modelos de Machine Learning** em cenários de previsão de demanda e otimização de estoque.

**Principais objetivos:**
1. **Geração massiva de dados:** Base para o projeto de IA [Smart Supply Chain AI](https://github.com/rbalbinotti/smart-supply-chain-ai).  
2. **Portfólio técnico:** Demonstrar proficiência em engenharia de dados, modelagem de séries temporais e desenvolvimento de pipelines em Python.

---

## ⚡ Escolha Seu Motor: Pandas vs. Polars

Este repositório oferece **duas implementações completas** do pipeline de geração de dados, cada uma adaptada a diferentes requisitos de desempenho e realismo:

| Recurso | Implementação Pandas | Implementação Polars |
|---------|----------------------|----------------------|
| **Notebook** | `synthetic_grocery.ipynb` | `synthetic_grocery_polars.ipynb` |
| **Linhas** | ~100.192 | ~300.605 (escalável) |
| **Colunas finais** | 30 | 31 |
| **Desempenho** | Single-thread, memória padrão | Multi-thread, Arrow, avaliação preguiçosa |
| **Melhor uso** | Prototipagem, datasets menores | Produção, simulação em larga escala |
| **Bibliotecas-chave** | `pandas`, `numpy` | `polars`, `fastparquet` |

Ambas compartilham a mesma lógica modular (`create_data_functions.py` e `weather_conditions.py`), mas diferem nos detalhes de implementação, resultando em **esquemas e riqueza de atributos distintos**.

---

## 📊 Comparação de Esquema & Adições no Polars

A versão Polars traz melhorias que tornam o dataset mais realista e detalhado:

### Esquema Pandas (30 colunas)
Inclui colunas como: `received_date`, `product_id`, `product`, `category`, `sub_category`, `shelf_life_days`, `maximum_days_on_sale`, `seasonality`, `storage_recommendation`, `unit_of_measurement`, `supplier_id`, `supplier`, `supplier_rating`, `distance_km`, `moq`, `in_season`, `is_holiday`, `day_classification`, `is_weekend`, `sales_demand`, `sales_volume`, `delivery_days`, `min_stock`, `max_stock`, `stock_quantity`, `temperature_classification`, `precipitation_classification`, `wind_classification`, `weather_severity`.

### Esquema Polars (31 colunas – saída final)
Inclui: `order_purchase_date`, `received_date`, `product_id`, `product`, `category`, `sub_category`, `sales_demand`, `sales_volume`, `seasonality`, `storage_recommendation`, `unit_of_measurement`, `shelf_life_days`, `maximum_days_on_sale`, `supplier_id`, `supplier`, `supplier_rating`, `distance_km`, `moq`, `delivery_days`, `transit_time`, `in_season`, `is_holiday`, `day_classification`, `is_weekend`, `min_stock`, `max_stock`, `stock_quantity`, `temperature_classification`, `precipitation_classification`, `wind_classification`, `weather_severity`.

### ✨ Melhorias no Polars
- **Segmentação de Estradas:** Modela `urban_km`, `highway_km`, `off_road_km` e calcula **transit_time** com distribuições de velocidade realistas.  
- **Estoque de Segurança:** `min_stock` e `max_stock` baseados em variabilidade de lead time e confiabilidade do fornecedor.  
- **Simulação de Pedido de Compra:** `order_purchase_date` derivado de rating do fornecedor, sazonalidade e dias de entrega.  
- **Quantização:** Colunas numéricas convertidas para tipos menores (UInt16, Float16) para otimização de memória.

---

## 🔬 Metodologia & Rigor Estatístico

A simulação segue princípios de **Decomposição de Séries Temporais**, modelando a demanda \(D(t)\) como função multivariada:

\[
D(t) = T(t) + S(t) + \sum \beta_i X_i(t) + \epsilon
\]

- **\(T(t)\):** Tendência de crescimento.  
- **\(S(t)\):** Sazonalidade semanal e anual.  
- **\(X_i(t)\):** Variáveis exógenas (preço, clima real INMET, feriados).  
- **\(\epsilon\):** Ruído gaussiano simulando incertezas de mercado.  

**Diferencial técnico:** Uso de dados meteorológicos reais do INMET/BDMEP, enriquecidos com engenharia de atributos para capturar correlações entre clima e demanda de perecíveis.

---

## ✨ Componentes do Pipeline

### Séries Temporais (`create_data_functions.py`)
- Série base com datas, IDs e valores alvo.  
- Tendência e sazonalidade.  
- Features de defasagem (`LagFeatureCreator`).  
- Eventos e feriados.  
- Preço com relação inversa à demanda.  

### Variáveis Climáticas (`weather_conditions.py`)
- Temperatura (Muito Frio, Temperado, Quente).  
- Precipitação (Sem chuva → Chuva violenta).  
- Vento por velocidade.  
- Ajustes sazonais por mês/estação.  

---

## 🛠️ Engenharia de Dados & MLOps

- **Modularização:** Lógica compartilhada entre versões.  
- **Formato otimizado:** Saída em `.parquet`.  
- **Pronto para deploy:** Dockerfile para isolamento.  
- **Gerenciamento de dependências:** `pyproject.toml` com **UV** (substitui Conda).  

---

## 📂 Estrutura de Diretórios

```text
.
├── create_data_functions.py
├── weather_conditions.py
├── data
│   ├── external
│   ├── processed
│   └── raw
├── synthetic_grocery.ipynb
├── synthetic_grocery_polars.ipynb
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── LICENSE
├── README.md
└── README_PT.md
```

---

## 📚 Stack & Referências

- **Core (Pandas):** `pandas`, `numpy`, `scikit-learn`, `fastparquet`.  
- **Core (Polars):** `polars` com Arrow e execução paralela.  
- **Estatística:** `holidays`, `workalendar`.  
- **Clima:** Dados reais do [**INMET/BDMEP**](https://bdmep.inmet.gov.br/).  
- **Projeto associado:** [Smart Supply Chain AI](https://github.com/rbalbinotti/smart-supply-chain-ai).  

---

## 🚀 Primeiros Passos

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/rbalbinotti/synthetic-grocery-data.git
cd synthetic-grocery-data

# Usando UV (recomendado)
uv sync

# Ou com pip
pip install -e .
```

Depois, abra o notebook correspondente e execute todas as células.

---

*Desenvolvido por **Roberto Rosário Balbinotti** – Arquiteto de ML & Especialista em Dados.*  
E-mail: rbalbinotti@gmail.com  
