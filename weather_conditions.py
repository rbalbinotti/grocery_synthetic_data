"""
Weather Conditions Data Preprocessing Utilities

This module provides functions for cleaning, transforming, and preparing weather-related
features (such as temperature, precipitation, and general conditions) for use as
exogenous variables in time series forecasting models.

Author: Roberto Rosário Balbinotti
Created: 2025
Version: 1.1
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime

class WeatherConditions:

    def __init__(self, df: pd.DataFrame):
        self.df = df


    def _classify_temperature(self, avg_temp: float) -> str:
        # Categorize temperature based on average value
        if avg_temp < 5:
            return "Very Cold"
        elif avg_temp <= 11:
            return "Cold"
        elif avg_temp <= 17:
            return "Cool"
        elif avg_temp <= 24:
            return "Mild to Temperate"
        elif avg_temp <= 29:
            return "Warm"
        elif avg_temp <= 35:
            return "Hot"
        else:
            return "Very Hot"

    def _classify_precipitation(self, precipitation: float) -> str:
        # Categorize precipitation intensity
        if precipitation == 0:
            return "No precipitation"
        elif precipitation < 2.5:
            return "Light Rain"
        elif precipitation < 10:
            return "Moderate Rain"
        elif precipitation < 50:
            return "Heavy Rain"
        else:
            return "Violent Rainfall"

    def _classify_wind(self, wind: float) -> str:
        # Categorize wind strength
        if wind <= 1.5:
            return "Calm / Light Breeze"
        elif wind <= 5.4:
            return "Gentle to Fresh Breeze"
        elif wind <= 10.7:
            return "Moderate to Strong Wind"
        elif wind <= 24.4:
            return "Very Strong Wind / Gale"
        else:
            return "Storm / Hurricane Force"

    def _classify_severity(self, row: pd.Series) -> str:
        
        avg_temp = row['daily_average_temperature_c']
        precipitation = row['daily_total_precipitation_mm']
        wind = row['daily_average_wind_speed_mps']

        temp_extreme = avg_temp < -5 or avg_temp > 40
        temp_intense = avg_temp < 0 or avg_temp > 30
        rain_severe = precipitation >= 10
        rain_extreme = precipitation >= 50
        wind_severe = wind >= 10.8
        wind_extreme = wind >= 24.5

        if temp_extreme or (rain_extreme and wind_extreme):
            return "Catastrophic"
        elif temp_intense and (rain_extreme or wind_extreme):
            return "Extreme"
        elif temp_intense or rain_severe or wind_severe:
            return "Severe"
        elif (avg_temp < 10 or avg_temp > 25) or \
            (0 < precipitation < 10) or \
            (1.5 < wind <= 10.7):
            return "Moderate"
        else:
            return "Normal"

    def classify_weather(self) -> pd.DataFrame:

        if self.df is None:
            raise ValueError("O DataFrame não pode ser None. Inicialize a classe com um DataFrame válido.")

        df_copy = self.df.copy()


        if "daily_average_temperature_c" not in df_copy.columns:
            df_copy["daily_average_temperature_c"] = (
                df_copy["daily_maximum_temperature_c"] + df_copy["daily_minimum_temperature_c"]
            ) / 2

        df_copy["temperature_classification"] = df_copy["daily_average_temperature_c"].apply(self._classify_temperature)
        df_copy["precipitation_classification"] = df_copy["daily_total_precipitation_mm"].apply(self._classify_precipitation)
        df_copy["wind_classification"] = df_copy["daily_average_wind_speed_mps"].apply(self._classify_wind)

        df_copy["weather_severity"] = df_copy.apply(self._classify_severity, axis=1)

        return df_copy

    @staticmethod
    def simulate_weather(date: datetime) -> tuple[int, int, str]:
        """
        Simula condições climáticas com base na data, considerando padrões sazonais no Hemisfério Sul.
        Este é um método estático porque não depende de nenhuma instância de `WeatherConditions`.
        """
        month = date.month

        # Simula verão (Dez-Mar)
        if month in [12, 1, 2, 3]:
            temperature = np.random.randint(15, 41)
            precipitation = np.random.choice([0, 5, 10, 20], p=[0.5, 0.25, 0.15, 0.1])
            condition = 'Rain and Heat' if precipitation > 0 else 'Sun and Heat'
        # Simula outono (Abr-Jun)
        elif month in [4, 5, 6]:
            temperature = np.random.randint(5, 34)
            precipitation = np.random.choice([0, 2, 5], p=[0.7, 0.2, 0.1])
            condition = 'Rainy' if precipitation > 0 else 'Pleasant'
        # Simula inverno (Jul-Set)
        elif month in [7, 8, 9]:
            temperature = np.random.randint(-8, 30)
            precipitation = np.random.choice([0, 1, 3], p=[0.8, 0.15, 0.05])
            condition = 'Cold and Rainy' if precipitation > 0 else 'Cold'
        # Simula primavera (Out-Nov)
        else:  # month in [10, 11]
            temperature = np.random.randint(10, 39)
            precipitation = np.random.choice([0, 5, 10], p=[0.6, 0.3, 0.1])
            condition = 'Unstable' if precipitation > 0 else 'Pleasant'

        return temperature, precipitation, condition



class PolarsWeatherConditions:
    """
    Optimized version for Polars (Lazy and Eager) using native expressions.
    """
    def __init__(self, lf: pl.LazyFrame | pl.DataFrame):
        # Ensure we work internally with LazyFrame for maximum optimization
        self.lf = lf.lazy() if isinstance(lf, pl.DataFrame) else lf

    def _classify_temperature(self) -> pl.Expr:
        # Replace if/elif with pl.when().then().otherwise()
        # which run natively and in parallel in Polars
        col = pl.col("daily_average_temperature_c")
        return (
            pl.when(col < 5).then(pl.lit("Very Cold"))
            .when(col <= 11).then(pl.lit("Cold"))
            .when(col <= 17).then(pl.lit("Cool"))
            .when(col <= 24).then(pl.lit("Mild to Temperate"))
            .when(col <= 29).then(pl.lit("Warm"))
            .when(col <= 35).then(pl.lit("Hot"))
            .otherwise(pl.lit("Very Hot"))
        )

    def _classify_precipitation(self) -> pl.Expr:
        col = pl.col("daily_total_precipitation_mm")
        return (
            pl.when(col == 0).then(pl.lit("No precipitation"))
            .when(col < 2.5).then(pl.lit("Light Rain"))
            .when(col < 10).then(pl.lit("Moderate Rain"))
            .when(col < 50).then(pl.lit("Heavy Rain"))
            .otherwise(pl.lit("Violent Rainfall"))
        )

    def _classify_wind(self) -> pl.Expr:
        col = pl.col("daily_average_wind_speed_mps")
        return (
            pl.when(col <= 1.5).then(pl.lit("Calm / Light Breeze"))
            .when(col <= 5.4).then(pl.lit("Gentle to Fresh Breeze"))
            .when(col <= 10.7).then(pl.lit("Moderate to Strong Wind"))
            .when(col <= 24.4).then(pl.lit("Very Strong Wind / Gale"))
            .otherwise(pl.lit("Storm / Hurricane Force"))
        )

    def _classify_severity(self) -> pl.Expr:
        avg_temp = pl.col('daily_average_temperature_c')
        precipitation = pl.col('daily_total_precipitation_mm')
        wind = pl.col('daily_average_wind_speed_mps')

        # Create boolean masks using Polars logical operators (&, |)
        temp_extreme = (avg_temp < -5) | (avg_temp > 40)
        temp_intense = (avg_temp < 0) | (avg_temp > 30)
        rain_severe = precipitation >= 10
        rain_extreme = precipitation >= 50
        wind_severe = wind >= 10.8
        wind_extreme = wind >= 24.5

        return (
            pl.when(temp_extreme | (rain_extreme & wind_extreme)).then(pl.lit("Catastrophic"))
            .when(temp_intense & (rain_extreme | wind_extreme)).then(pl.lit("Extreme"))
            .when(temp_intense | rain_severe | wind_severe).then(pl.lit("Severe"))
            .when(
                (avg_temp < 10) | (avg_temp > 25) |
                ((precipitation > 0) & (precipitation < 10)) |
                ((wind > 1.5) & (wind <= 10.7))
            ).then(pl.lit("Moderate"))
            .otherwise(pl.lit("Normal"))
        )

    def classify_weather(self) -> pl.LazyFrame:
        lf_temp = self.lf
        
        # Correction: Check if the average column already exists in the metadata structure
        if "daily_average_temperature_c" not in lf_temp.columns:
            lf_temp = lf_temp.with_columns(
                ((pl.col("daily_maximum_temperature_c") + pl.col("daily_minimum_temperature_c")) / 2)
                .alias("daily_average_temperature_c")
            )

        # Now that the column definitely exists in the plan, we can safely proceed:
        return (
            lf_temp
            # 1. Add all new classifications in parallel
            .with_columns([
                self._classify_temperature().alias("temperature_classification"),
                self._classify_precipitation().alias("precipitation_classification"),
                self._classify_wind().alias("wind_classification"),
            ])
            # 2. Add severity based on the columns from the previous step
            .with_columns(
                self._classify_severity().alias("weather_severity")
            )
        )



if __name__ == '__main__':
    
    import pandas as pd
    from datetime import datetime

    # Sample weather data
    data = {
        'daily_maximum_temperature_c': [30, 22, 10],
        'daily_minimum_temperature_c': [20, 14, 5],
        'daily_total_precipitation_mm': [0, 12, 55],
        'daily_average_wind_speed_mps': [2.0, 11.0, 25.0]
    }

    df_weather = pd.DataFrame(data)

    # Initialize the classifier
    weather_analyzer = WeatherConditions(df_weather)

    # Apply classification
    classified_df = weather_analyzer._classify_weather()

    # Display results
    print(classified_df[['daily_average_temperature_c', 'temperature_classification',
                        'precipitation_classification', 'wind_classification',
                        'weather_severity']])
