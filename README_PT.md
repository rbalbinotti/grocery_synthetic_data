# 🛍️ Gerador de Dados Sintéticos para Cadeia de Suprimentos de Mercado  
> **Motor de Alta Fidelidade para Otimização de Estoque e Previsão de Demanda**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/roberto-balbinotti)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/robertobalbinotti/synthetic-grocery-data)
---

## 🎯 Contexto & Objetivos Estratégicos

No setor varejista (mercado), a escassez de dados históricos limpos ou a confidencialidade de dados reais dificulta o desenvolvimento ágil de modelos de IA. Este projeto preenche essa lacuna fornecendo um **Gêmeo Digital** da cadeia de suprimentos, simulando operações complexas e permitindo o **teste de modelos de Machine Learning** em cenários de previsão de demanda e otimização de inventário.

**Principais objetivos:**
1.  **Geração massiva de dados:** Base para o projeto de IA [Smart Supply Chain AI](https://github.com/rbalbinotti/smart-supply-chain-ai).
2.  **Portfólio técnico:** Demonstrar proficiência em engenharia de dados, modelagem de séries temporais e desenvolvimento de pipelines em Python.

---

## 🔬 Metodologia & Rigor Estatístico

A simulação segue os princípios de **Decomposição de Série Temporal**, modelando a demanda $D(t)$ como uma função multivariada:

$$D(t) = T(t) + S(t) + \sum \beta_i X_i(t) + \epsilon$$

-   **$T(t)$:** Tendência de crescimento determinística.
-   **$S(t)$:** Sazonalidade semanal e anual.
-   **$X_i(t)$:** Variáveis exógenas (preço, dados climáticos reais do INMET, feriados).
-   **$\epsilon$:** Ruído gaussiano simulando incertezas do mercado.

### Diferencial Técnico: Dados Climáticos Reais
Diferente de geradores sintéticos comuns, este projeto incorpora **dados meteorológicos reais** (INMET/BDMEP), enriquecidos com feature engineering para mapear severidade climática e capturar correlações reais entre temperatura e demanda de perecíveis.

---

## ✨ Componentes do Pipeline

### Série Temporal (`create_data_functions.py`)
-   **Série base:** DataFrame com datas (`ds`), IDs e valores-alvo (demanda/vendas).
-   **Tendência & sazonalidade:** Crescimento e ciclos semanais/anuais.
-   **Features de lag:** `LagFeatureCreator` adiciona dependências temporais (ex.: vendas da semana anterior).
-   **Eventos & feriados:** Impactos de promoções e datas especiais.
-   **Preço:** Relação inversa entre preço e demanda.

### Variáveis Exógenas Climáticas (`weather_conditions.py`)
-   **Temperatura:** Classificada em faixas (Muito Frio, Ameno, Quente).
-   **Precipitação:** Intensidade (Sem chuva → Chuva Violenta).
-   **Vento:** Classificado por velocidade.
-   **Simulação sazonal:** Ajustes baseados em meses e estações.

---

## 📊 Estrutura & Resultado Final

O conjunto de dados final é salvo no formato **Parquet** para alta performance, contendo **100.192 linhas e 29 colunas**.

Amostra:

| data_recebimento | produto | categoria | sub_categoria | dias_validade | fornecedor | distancia_km | classe_temp | classe_precip | classe_vento | feriado | demanda_vendas | volume_vendas | qtd_estoque |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2025-02-04 | Ovo (Galinha) | Laticínios | Ovos | 28 | FreshEggs Co. | 65 | Quente | Sem precip. | Brisa Leve | Falso | Alta | 318 | 1096 |
| 2023-01-03 | Açúcar | Despensa | Assados | 730 | Atacadista | 25 | Quente | Sem precip. | Brisa Leve | Falso | Alta | 10 | 33 |

> **Nota:** O arquivo completo `grocery_data.parquet` com todas as **100.192 linhas** está disponível para download no **[Kaggle](https://www.kaggle.com/datasets/robertobalbinotti/synthetic-grocery-data)**.

---

## 🛠️ Engenharia de Dados & MLOps

-   **Modularização:** Lógica separada em `create_data_functions.py` e `weather_conditions.py`.
-   **Formato otimizado:** `.parquet` para pipelines de Big Data.
-   **Pronto para implantação:** Dockerfile para isolamento de ambiente.
-   **Gestão de dependências:** `pyproject.toml` com PDM.

---

## 📂 Estrutura de Diretórios

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

## 📚 Stack & Referências

-   **Core:** `Pandas`, `NumPy`, `Scikit-Learn`, `fastparquet`.
-   **Estatística:** `holidays`, `workalendar`.
-   **Fonte climática:** Dados reais do [**INMET/BDMEP**](https://bdmep.inmet.gov.br/)
-   **Projeto associado:** [Smart supply Chain AI](https://github.com/rbalbinotti/smart-supply-chain-ai)

---

*Desenvolvido por **Roberto Rosário Balbinotti** – Arquiteto de ML & Especialista em Dados.*
E-mail: rbalbinotti@gmail.com

---