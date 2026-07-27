"""
Utility Functions for Synthetic Demand Data Generation (Grocery/Retail)

This module provides functions to create a synthetic time series dataset,
simulating product sales or demand. The goal is to generate complex data,
incorporating trend, seasonality, events (such as holidays or promotions),
and price variations, making it suitable for forecasting modeling tasks.

Function/Class:	Purpose
- get_product_ids:	
    Creates unique product IDs and product names based on categories (e.g., "A1", "A2", "B1", etc.).
- create_base_df:	
    Constructs the base time series DataFrame with essential columns: unique_id, ds (date/timestamp), and y (target value, e.g., demand or sales).
- add_trend_seasonality:	
    Adds trend (growth) and seasonality (annual, weekly) components to the target value (y) using functions like sine/cosine or simple models.
- add_events:
	Introduces the effect of events (such as holidays, promotions, or price changes) into the data, increasing or decreasing demand on specific dates.
- add_price_feature:
	Adds a price column as an exogenous variable, typically with an inverse relationship to demand (demand falls when price rises).
- create_synthetic_data:
	The main function that orchestrates the calls to all the above functions to build the complete synthetic time series dataset.

Author: Roberto Rosário Balbinotti
Created: 2025
Version: 1.0
"""

import polars as pl
import pandas as pd
import numpy as np
import ast
import holidays
import math
import importlib

from scipy.stats import truncnorm
from workalendar.america import Brazil
from sklearn.base import BaseEstimator, TransformerMixin


def create_purchase_order_polars(df: pl.DataFrame | pl.LazyFrame, seed: int = None) -> pl.DataFrame | pl.LazyFrame:
    """
    Generate purchase order dates based on supplier rating, delivery days, and seasonality factors.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Input Polars DataFrame or LazyFrame containing at least the following columns:
        - supplier_rating (numeric)
        - in_season (boolean)
        - delivery_days (numeric)
        - received_date (date/datetime)

    seed : int, optional
        Random seed for reproducibility of generated factors.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        DataFrame with a new column:
        - order_purchase : calculated purchase order date

    Intermediate columns (`rating_penalty`, `season_factor`, `beta_penalty`, `order_days`) are dropped at the end.
    """

    # Get the number of rows in the DataFrame
    length = df.collect().height

    # Create a random number generator with optional seed
    rng = np.random.default_rng(seed=seed)

    # Generate supplier rating penalty values ~ U(0, 2.1)
    rating_penalty = rng.uniform(low=0, high=2.1, size=length)

    # Generate seasonality factor values ~ U(0.8, 0.9)
    season_factor = rng.uniform(low=0.8, high=0.9, size=length)

    return (
        df.with_columns(
            pl.Series("rating_penalty", rating_penalty),
            pl.Series("season_factor", season_factor)
        )
        # Apply beta penalty based on supplier rating
        .with_columns(
            pl.when(pl.col("supplier_rating") < 2)
            .then(pl.col("rating_penalty") / 2)
            .when(pl.col("supplier_rating") < 4)
            .then(pl.col("rating_penalty") / 1.5)
            .otherwise(pl.col("rating_penalty"))
            .alias("beta_penalty")
        )
        # Calculate order days considering seasonality
        .with_columns(
            pl.when(pl.col("in_season"))
            .then(((pl.col("delivery_days") + pl.col("beta_penalty")) * pl.col("season_factor") + 0.5).round())
            .otherwise((pl.col("delivery_days") + pl.col("beta_penalty") + 0.5).round())
            .alias("order_days")
            .cast(pl.UInt8)
        )
        # Compute purchase order date
        .with_columns(
            (pl.col("received_date") - pl.duration(days=pl.col("order_days")))
            .alias("order_purchase_date")
        )
    ).drop(["rating_penalty", "season_factor", "beta_penalty", "order_days"])



def estimate_delivery_polars(
        df: pl.DataFrame | pl.LazyFrame,
        col_transit: str = "transit_time_decimal",
        col_weather: str = "weather_severity",
        col_day_class: str = "day_classification",
        weather_factor: dict = None,
        day_adjustment: dict = None,
        daily_drive_limit: float = 8.0,
        daily_constant_drive: float = 5.5,
        pause_time: float = 0.5,
        sleep_hours: float = 11.0,
        week_max_hours: float = 144.0,
        weekly_rest: float = 35.0
) -> pl.DataFrame | pl.LazyFrame:
    """
    Estimate delivery duration by simulating realistic driving and rest constraints.

    The calculation accounts for:
    - Daily driving limits and mandatory sleep cycles
    - Short pauses during long driving days
    - Weekly rest periods after exceeding weekly driving hours
    - Weather severity multipliers (e.g., delays in bad weather)
    - Day classification adjustments (weekdays, weekends, holidays)
    - Randomized processing overhead (picking, packing, dispatch)

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Input Polars DataFrame or LazyFrame containing transit data.
    col_transit : str, default="transit_time_decimal"
        Column name representing raw transit time in hours.
    col_weather : str, default="weather_severity"
        Column name representing weather severity classification.
    col_day_class : str, default="day_classification"
        Column name representing day classification (weekday, weekend, holiday).
    weather_factor : dict, optional
        Mapping of weather severity to time multipliers. Defaults to:
        {"Normal": 1.0, "Moderate": 1.15, "Severe": 1.3}.
    day_adjustment : dict, optional
        Mapping of day classification to additional delay (hours). Defaults to:
        {"Weekdays": 0.0, "Saturday": 0.5, "Sunday": 1.0, "Holiday": 1.5}.
    daily_drive_limit : float, default=8.0
        Maximum driving hours allowed per day before mandatory sleep.
    daily_constant_drive : float, default=5.5
        Driving hours before a short pause is required.
    pause_time : float, default=0.5
        Duration of each short pause in hours.
    sleep_hours : float, default=11.0
        Duration of mandatory sleep after exceeding daily drive limit.
    week_max_hours : float, default=144.0
        Maximum weekly driving hours before mandatory weekly rest.
    weekly_rest : float, default=35.0
        Duration of weekly rest in hours.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        DataFrame with additional column:
        - delivery_days : Final estimated delivery time in days.
    """

    # Determine number of rows depending on DataFrame type
    if isinstance(df, pl.LazyFrame):
        length = df.collect().height
    else:
        length = df.height

    # Default weather multipliers
    weather_factor = weather_factor or {"Normal": 1.0, "Moderate": 1.15, "Severe": 1.3}

    # Default day adjustments
    day_adjustment = day_adjustment or {"Weekdays": 0.0, "Saturday": 0.5, "Sunday": 1.0, "Holiday": 1.5}

    # Random processing overhead (between 4 and 36 hours)
    processing_hours = np.random.uniform(4.0, 36.0, length)

    return (
        df.with_columns([
            # Number of full sleep cycles (11h each) based on daily driving limit
            (pl.col(col_transit) // daily_drive_limit).alias("num_sleeps"),

            # Remaining driving time after last full day
            (pl.col(col_transit) % daily_drive_limit).alias("remainder_drive"),

            # Number of weekly rests required
            (pl.col(col_transit) // week_max_hours).alias("num_weekly_rests")
        ])
        .with_columns([
            # Short pauses:
            # - Each full 8h driving day guarantees one pause
            # - Remaining driving time adds a pause if >= 5.5h
            (
                (pl.col("num_sleeps") * (daily_drive_limit // daily_constant_drive)) +
                (pl.col("remainder_drive") // daily_constant_drive)
            ).alias("num_pauses")
        ])
        .with_columns([
            # Adjusted transit time including pauses, sleep, and weekly rests
            (
                pl.col(col_transit) +
                (pl.col("num_pauses") * pl.lit(pause_time)) +
                (pl.col("num_sleeps") * pl.lit(sleep_hours)) +
                (pl.col("num_weekly_rests") * pl.lit(weekly_rest))
            ).alias("transit_time")
        ])
        .with_columns([
            # Final delivery time in days:
            # - Apply weather multiplier
            # - Add day classification adjustment
            # - Add processing overhead
            (
                (pl.col("transit_time") * pl.col(col_weather).replace_strict(weather_factor, default=1.0))
                + pl.col(col_day_class).replace_strict(day_adjustment, default=0.0)
                + pl.lit(processing_hours)
            ).alias("delivery_hours")
        ])
        .with_columns([
            # Convert total hours to days
            (pl.col("delivery_hours") / pl.lit(24)).alias("delivery_days")
        ])
        .drop("transit_time_decimal", "num_sleeps", "remainder_drive", "num_weekly_rests", "num_pauses", "delivery_hours")
    )



def simulate_distribution_speeds(
    X: pl.DataFrame | pl.LazyFrame, 
    seed=456,
    # Urban Parameters
    min_speed_urban=5,
    max_speed_urban=60,
    dp_urban=8,
    # Off-Road Parameters
    min_speed_off_road=10,
    max_speed_off_road=80,
    dp_off_road=12,
    # Highway Parameters
    min_speed_road=20,
    max_speed_road=110,
    dp_speed_road=18
) -> pl.DataFrame | pl.LazyFrame:
    """
    Simulates speed distributions for different road types (urban, off-road, highway)
    and calculates the total transit time in minutes.

    Parameters
    ----------
    X : pl.DataFrame | pl.LazyFrame
        Input dataset containing distances (off_road_km, highway_km, urban_km).
    seed : int, optional
        Random seed for reproducibility (default=456).
    min_speed_urban, max_speed_urban, dp_urban : float
        Parameters for truncated normal distribution of urban speeds.
    min_speed_off_road, max_speed_off_road, dp_off_road : float
        Parameters for lognormal distribution of off-road speeds.
    min_speed_road, max_speed_road, dp_speed_road : float
        Parameters for lognormal distribution of highway speeds.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        DataFrame with a new column `transit_time_minute` representing
        the total travel time in minutes.
    """

    is_lazy = isinstance(X, pl.LazyFrame)
    df = X.collect() if is_lazy else X
    length = df.height
    
    rng = np.random.default_rng(seed=seed)
        
    # Automatic calculation of mean speeds based on provided parameters
    mean_speed_urban = (max_speed_urban + min_speed_urban) / 2
    mean_speed_off_road = (max_speed_off_road + min_speed_off_road) / 2
    mean_speed_road = (max_speed_road + min_speed_road) / 2

    # 1. Calculate sigma and mu for the lognormal distribution
    sigma_off_road = np.sqrt(np.log(1 + (dp_off_road**2 / mean_speed_off_road**2)))
    mu_off_road = np.log(mean_speed_off_road) - (sigma_off_road**2 / 2)

    sigma_road = np.sqrt(np.log(1 + (dp_speed_road**2 / mean_speed_road**2)))
    mu_road = np.log(mean_speed_road) - (sigma_road**2 / 2)

    # 2. Generate arrays with simulated speeds
    speed_off_road_arr = rng.lognormal(mean=mu_off_road, sigma=sigma_off_road, size=length)
    speed_highway_arr = rng.lognormal(mean=mu_road, sigma=sigma_road, size=length)
    speed_urban_arr = truncnorm.rvs(
        (min_speed_urban - mean_speed_urban) / dp_urban, 
        (max_speed_urban - mean_speed_urban) / dp_urban, 
        loc=mean_speed_urban, 
        scale=dp_urban, 
        size=length, 
        random_state=seed
    )

    # 3. Add new columns with simulated speeds
    result_df = df.with_columns((
       pl.Series("mean_speed_off_road", speed_off_road_arr),
       pl.Series("mean_speed_highway", speed_highway_arr),
       pl.Series("mean_speed_urban", speed_urban_arr))
    )

    # 4. Calculate total transit time in minutes
    result_df = result_df.with_columns([
        (
            (pl.col("off_road_km") / pl.col("mean_speed_off_road"))
            + (pl.col("highway_km") / pl.col("mean_speed_highway"))
            + (pl.col("urban_km") / pl.col("mean_speed_urban"))
        )
        .alias("transit_time_decimal")  
    ]).drop(["off_road_km", "highway_km", "urban_km", "mean_speed_off_road", "mean_speed_highway", "mean_speed_urban"])

    return result_df.lazy() if is_lazy else result_df




def road_simulation_polars(
        df: pl.DataFrame | pl.LazyFrame,
        distance: str = "distance_km",
        subcategory: str = "sub_category",
        category_off_road: tuple = ("Meat", "Seafood", "Vegetables", "Fruits", "Dairy", "Eggs", "Grains & Rice")
) -> pl.DataFrame | pl.LazyFrame:
    """
    Simulate road usage distribution (urban, highway, off-road) based on product categories and travel distance.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Input Polars DataFrame containing distance and category information.
    distance : str, default="distance_km"
        Column name representing the total travel distance in kilometers.
    subcategory : str, default="sub_category"
        Column name representing the product subcategory.
    category_off_road : tuple, default=("Meat", "Seafood", "Vegetables", "Fruits", "Dairy", "Eggs", "Grains & Rice")
        Tuple of product subcategories that require off-road travel.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        DataFrame with three new columns:
        - "off_road_km": estimated kilometers traveled off-road
        - "highway_km": estimated kilometers traveled on highways
        - "urban_km": estimated kilometers traveled in urban areas
    """

    # Generate a random factor between 0.1 and 0.4
    random_values = np.random.uniform(0.1, 0.4, size=df.collect().height)


    # Adjusted distance (subtracting 50 km to account for urban travel)
    _distance = pl.col(distance) - 50

    return (
        df.with_columns(
            pl.Series(random_values).alias("random_values")
        )
        .with_columns(
            # Off-road travel applies only if the product is in the off-road category
            # and total distance exceeds 50 km. Otherwise, set to 0.
            pl.when((pl.col(subcategory).is_in(category_off_road)) & (pl.col(distance) > 50))
            .then((_distance * pl.col("random_values")).cast(pl.Int32))
            .otherwise(0)
            .alias("off_road_km")
        )
        .with_columns(
            # Highway travel is the remaining distance after subtracting off-road km,
            # only if total distance exceeds 50 km.
            pl.when(pl.col(distance) > 50)
            .then(_distance - pl.col("off_road_km"))
            .otherwise(0)
            .alias("highway_km")
        )
        .with_columns(
            # Urban travel is capped at 50 km if distance exceeds 50,
            # otherwise it equals the total distance.
            pl.when(pl.col(distance) > 50)
            .then(50)
            .otherwise(pl.col(distance))
            .alias("urban_km")
        )
    ).drop("random_values").lazy()




class LagFeatureCreator(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that creates lagged (shifted) features
    for a target column within each group.

    This is typically used to add lag-based features (e.g., previous week/month values)
    to help models capture temporal dependencies.

    Attributes:
        group_column (str): Name of the column used for grouping (e.g., product_id).
        shift_column (str): Name of the column to apply the lag on (e.g., sales, demand).
        lags (list[int]): List of lag periods to create (default: [7, 14, 28]).

    Example Usage
        df = pd.DataFrame({
        'product_id': ['A', 'A', 'A', 'B', 'B', 'B'],
        'date': pd.date_range('2025-01-01', periods=3).tolist() * 2,
        'sales': [10, 12, 15, 20, 25, 23]})

        lagger = LagFeatureCreator(group_column='product_id', shift_column='sales', lags=[1, 2])
        df_lagged = lagger.transform(df)
        print(df_lagged)
    """

    def __init__(self, group_column: str, shift_column: str, lags: list[int] = [7, 14, 28]):
        """
        Initialize the LagFeatureCreator transformer.

        Args:
            group_column (str): Column used to group the data (e.g., by product or store).
            shift_column (str): Column to apply the lag on.
            lags (list[int], optional): List of lag intervals. Defaults to [7, 14, 28].
        """
        self.group_column = group_column
        self.shift_column = shift_column
        self.lags = lags

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit method (no fitting required).

        Args:
            X (pd.DataFrame): Input data.
            y (ignored): Not used, present for API compatibility.

        Returns:
            self: Returns the transformer itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by creating lagged versions of the shift column.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with additional lag columns.
        """
        X_copy = X.copy()

        # Create one new column per lag value
        for lag in self.lags:
            lag_col_name = f"{self.shift_column}_lag_{lag}"
            X_copy[lag_col_name] = (
                X_copy.groupby(self.group_column)[self.shift_column].shift(lag)
            )

        return X_copy





class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer that extracts date-related features from a specified date column.

    This transformer uses country-specific calendars from the 'workalendar' library
    to identify holidays and business days. It generates the following features:
    - year, month, day, day_of_week, is_weekend, is_holiday, is_business_day.

    Attributes:
        date_column (str): Name of the column containing date values.
        country (str): Name of the country for which the calendar is used.
        calendar (object): Instance of the country's calendar.
    
    Example usage:
        # Brazil calendar
        extractor_br = DateFeatureExtractor(country='Brazil')
        df_features_br = extractor_br.transform(df)

        # United States calendar
        extractor_us = DateFeatureExtractor(country='UnitedStates')
        df_features_us = extractor_us.transform(df)
    """

    def __init__(self, date_column='received_date', country='Brazil'):
        """
        Initialize the DateFeatureExtractor with the target date column and country-specific calendar.

        Args:
            date_column (str, optional): Name of the column containing the dates. Defaults to 'received_date'.
            country (str, optional): Country name used to select the calendar. Defaults to 'Brazil'.

        Raises:
            ValueError: If the specified country is not supported by workalendar.
        """
        self.date_column = date_column
        self.country = country
        self.calendar = self._get_calendar(country)

    def _get_calendar(self, country):
        """
        Dynamically load a country-specific calendar from the workalendar library.

        Args:
            country (str): Country name, e.g., 'Brazil', 'UnitedStates', 'France'.

        Returns:
            object: Instance of the country’s calendar.

        Raises:
            ValueError: If the country calendar cannot be found.
        """
        # Common workalendar submodules organized by region
        region_modules = [
            'workalendar.america',
            'workalendar.europe',
            'workalendar.asia',
            'workalendar.africa',
            'workalendar.oceania'
        ]

        for module_name in region_modules:
            try:
                # Dynamically import the module
                module = importlib.import_module(module_name)
                # Attempt to get the class that matches the country name
                calendar_class = getattr(module, country, None)
                if calendar_class:
                    return calendar_class()
            except ModuleNotFoundError:
                continue

        raise ValueError(f"Unsupported or unavailable country calendar: {country}")

    def fit(self, X, y=None):
        """
        Fit method (no fitting necessary for this transformer).

        Args:
            X (pd.DataFrame): Input data.
            y (ignored): Not used, present for API consistency.

        Returns:
            self: Returns the transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by extracting date-based features e as juntando.

        Args:
            X (pd.DataFrame): Input data containing the date column.

        Returns:
            pd.DataFrame: DataFrame com novas colunas de feature de data e as colunas originais.
        """
        # 1. Copy the input DataFrame to avoid modifying the original
        X_out = X.copy() 

        # 2. Convert the date column to datetime format
        dates = pd.to_datetime(X_out[self.date_column])

        features_data = []
        for date in dates:
            pd_date = pd.Timestamp(date)
            features_data.append({
                'ds': pd_date,
                'year': pd_date.year,
                'month': pd_date.month,
                'day': pd_date.day,
                'day_of_week': pd_date.dayofweek,
                'is_weekend': pd_date.dayofweek >= 5,
                'is_holiday': self.calendar.is_holiday(pd_date) if self.calendar else False, 
                'is_business_day': self.calendar.is_working_day(pd_date) if self.calendar else (pd_date.dayofweek < 5)
            })

        # 3. Create a DataFrame from the extracted features
        features_df = pd.DataFrame(features_data, index=X.index)
        
        # 4. Concatenate the new features with the original DataFrame
        for col in features_df.columns:
            X_out[col] = features_df[col]
        
        # 5. Optionally drop the original date column
        if self.date_column in X_out.columns:
            X_out = X_out.drop(columns=[self.date_column])

        return X_out



def estimate_delivery_days(row):
    """
    Estimates a realistic delivery time (in days) for a given order,
    incorporating processing time, transit duration based on distance,
    weather impact, and calendar-based delays.

    Parameters:
    -----------
    row : pandas.Series
        A row from a DataFrame containing:
        - 'distance_km': float, delivery distance in kilometers
        - 'weather_severity': str, one of ['Normal', 'Moderate', 'Severe']
        - 'day_classification': str, one of ['Weekdays', 'Saturday', 'Sunday', 'Holiday']

    Returns:
    --------
    float
        Estimated delivery time in days, rounded to two decimal places.

    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'distance_km': [45, 180, 800, 1200],
    ...     'weather_severity': ['Normal', 'Moderate', 'Severe', 'Normal'],
    ...     'day_classification': ['Weekdays', 'Saturday', 'Sunday', 'Holiday']
    ... })
    >>> df['delivery_days'] = df.apply(estimate_delivery_days_realistic, axis=1)
    >>> print(df)
       distance_km weather_severity day_classification  delivery_days
    0         45.0           Normal           Weekdays           2.13
    1        180.0         Moderate           Saturday           4.67
    2        800.0           Severe             Sunday           9.85
    3       1200.0           Normal             Holiday          14.32
    """

    # 1. Simulate processing time (e.g., picking, packing, dispatch)
    processing_days = np.random.uniform(1.0, 2.0)

    # 2. Transit time based on distance (simulating business days)
    if row['distance_km'] <= 50:
        transit_days = np.random.uniform(0.5, 1.5)
    elif row['distance_km'] <= 150:
        transit_days = np.random.uniform(1.0, 2.5)
    elif row['distance_km'] <= 400:
        transit_days = np.random.uniform(2.0, 4.0)
    elif row['distance_km'] <= 1000:
        transit_days = np.random.uniform(4.0, 8.0)
    else:
        transit_days = np.random.uniform(7.0, 15.0)

    base_days = processing_days + transit_days

    # 3. Weather impact multiplier
    weather_factor = {
        "Normal": 1.0,
        "Moderate": 1.15,
        "Severe": 1.3
    }.get(row['weather_severity'], 1.0)

    # 4. Additional delay based on day classification
    day_adjustment = {
        "Weekdays": 0.0,
        "Saturday": 0.5,
        "Sunday": 1.0,
        "Holiday": 1.5
    }.get(row['day_classification'], 0.0)

    # 5. Final delivery time calculation
    delivery_days = base_days * weather_factor + day_adjustment
    return math.ceil(delivery_days)



def create_min_max_stock(
    df: pd.DataFrame,
    base_min: pd.Series,
    base_max: pd.Series,
    base_reorder: pd.Series
) -> pd.DataFrame:
    """
    Generates randomized minimum stock, maximum stock, and reorder point values for inventory items,
    and appends them as new columns to the input DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        The original DataFrame containing inventory items.
    base_min : pd.Series
        Base minimum stock levels for each item.
    base_max : pd.Series
        Base maximum stock levels for each item.
    base_reorder : pd.Series
        Base reorder point levels for each item.

    Returns:
    -------
    pd.DataFrame
        The input DataFrame with three new columns:
        - 'min_stock': randomized minimum stock (clipped to at least 1)
        - 'max_stock': randomized maximum stock (at least one unit above min_stock)
        - 'reorder_point': randomized reorder point (between min_stock + 1 and max_stock - 1)

    Notes:
    -----
    - Randomness is seeded with 42 for reproducibility.
    - Random variation is applied in the range [-2, 1] to each base value.

    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({'item': ['A', 'B', 'C']})
    >>> base_min = pd.Series([5, 10, 15])
    >>> base_max = pd.Series([20, 25, 30])
    >>> base_reorder = pd.Series([10, 15, 20])
    >>> create_min_max_stock(data, base_min, base_max, base_reorder)
       item  min_stock  max_stock  reorder_point
    0     A          4         20             10
    1     B          9         25             15
    2     C         14         30             20
    """

    rng = np.random.default_rng(seed=42)

    # Apply random variation to base values
    min_stock = base_min + rng.integers(-2, 2, size=len(base_min))
    max_stock = base_max + rng.integers(-2, 2, size=len(base_max))
    reorder_point = base_reorder + rng.integers(-2, 2, size=len(base_reorder))

    # Ensure logical constraints
    df['min_stock'] = min_stock.clip(lower=1)
    df['max_stock'] = np.maximum(min_stock + 1, max_stock)
    df['reorder_point'] = np.maximum(min_stock + 1, np.minimum(reorder_point, max_stock - 1))

    return df



def min_max_stock_polars(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Calculate minimum and maximum stock levels for products based on supplier performance 
    and sales volatility.

    This function derives inventory thresholds by grouping data by supplier and product.
    It calculates the minimum stock using maximum lead times and determines the maximum 
    stock by adding the Minimum Order Quantity (MOQ) to the calculated minimum.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Input data containing the following columns:
        - supplier_id: Unique identifier for the supplier.
        - product_id: Unique identifier for the product.
        - sales_volume: Historical units sold.
        - delivery_days: Lead time in days.
        - supplier_rating: Numerical rating (1-5) used for safety stock adjustment.
        - moq: Minimum Order Quantity defined by the supplier.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        The original dataframe augmented with two new columns:
        - min_stock (Int32): The reorder point (Mean sales * Maximum lead time).
        - max_stock (Int32): The maximum target inventory level (min_stock + moq).
    """
    
    # Grouping keys used for window functions to ensure consistency across the same product/supplier
    partition = ["supplier_id", "product_id"]

    return (
        df.with_columns([
            # Calculate Min Stock: Product of maximum demand and maximum lead time.
            # Cast to Int32 to prevent overflow for large inventory counts.
            (
                pl.col("sales_volume").mean().over(partition) * pl.col("delivery_days").max().over(partition)
            ).cast(pl.Int32).alias("min_stock"),

            # Calculate Safety Stock: Uses a service factor (1.65 for ~95% service level) 
            # and applies a multiplier based on supplier reliability/rating.
            pl.when(pl.col("supplier_rating") >= 4)
            .then(
                pl.col("sales_volume").std().over(partition)
                * pl.col("delivery_days").max().over(partition).sqrt()
                * 1.65 * 1.2
            )
            .when(pl.col("supplier_rating") >= 2)
            .then(
                pl.col("sales_volume").std().over(partition)
                * pl.col("delivery_days").max().over(partition).sqrt()
                * 1.65 * 1.1
            )
            .otherwise(
                pl.col("sales_volume").std().over(partition)
                * pl.col("delivery_days").max().over(partition).sqrt()
                * 1.65
            ).alias("stock_safety")
        ])
        .with_columns([
            # Calculate Max Stock: Defined as the reorder point plus the Minimum Order Quantity.
            (pl.col("min_stock") + pl.col("moq")).cast(pl.Int32).alias("max_stock")
        ])
    ).drop("stock_safety") # Remove intermediate calculation column before returning


def simulate_purchase_order_columns(df, random_state=None):
    """
    Simulates purchase order-related columns based on product attributes, logistics, and calendar effects.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the following columns:
        - 'received_date': date when goods were received
        - 'sales_volume': recent sales volume
        - 'min_stock': minimum stock threshold
        - 'shelf_life_days': shelf life of the product in days
        - 'sales_demand': demand level (e.g., 'High', 'Low', 'Normal')
        - 'distance_km': delivery distance in kilometers
        - 'weather_severity': weather impact ('Severe', 'Moderate', 'Mild')

    random_state : int, optional
        Seed for random number generation to ensure reproducibility.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with simulated purchase order columns:
        - 'order_date': date the order was placed

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'received_date': ['2025-09-20', '2025-09-21'],
    ...     'sales_volume': [10, 5],
    ...     'min_stock': [20, 15],
    ...     'shelf_life_days': [5, 60],
    ...     'sales_demand': ['High', 'Low'],
    ...     'distance_km': [800, 300],
    ...     'weather_severity': ['Moderate', 'Severe']
    ... })
    >>> simulate_purchase_order_columns(df, random_state=42)
           order_date
    0 2025-09-13
    1 2025-09-14
    """
    if random_state is not None:
        np.random.seed(random_state)

    result_df = pd.DataFrame(index=df.index)

    def process_order(row):
        received_date = pd.to_datetime(row['received_date'])

        # Order date: 2 to 10 days before received date
        days_before = np.random.randint(2, 11)
        order_date = received_date - pd.Timedelta(days=days_before)

        # Base quantity based on sales and minimum stock
        base_qty = max(row['sales_volume'] * 5, row['min_stock'] * 2)

        # Adjust for perishability
        if row['shelf_life_days'] <= 7:
            order_qty = base_qty * 0.7  # Smaller orders for highly perishable items
        elif row['shelf_life_days'] <= 30:
            order_qty = base_qty * 1.0
        else:
            order_qty = base_qty * 1.5  # Larger orders for non-perishables

        # Adjust for demand
        if 'High' in str(row['sales_demand']):
            order_qty *= 1.3
        elif 'Low' in str(row['sales_demand']):
            order_qty *= 0.7

        order_qty = int(round(max(row['min_stock'], order_qty)))

        # Base delay based on distance
        base_delay = max(1, row['distance_km'] // 400)

        # Weather impact on delay
        if row['weather_severity'] == 'Severe':
            base_delay += 3
        elif row['weather_severity'] == 'Moderate':
            base_delay += 1

        # Actual delay with variability
        real_delay = base_delay + np.random.randint(0, 3)

        estimated_delivery = order_date + pd.Timedelta(days=days_before)
        actual_delivery = order_date + pd.Timedelta(days=days_before + real_delay)
        delay_days = max(0, (actual_delivery - received_date).days)

        return pd.Series({
            'order_date': order_date,
            # Uncomment below to include more columns:
            # 'order_quantity': order_qty,
            # 'estimated_delivery_date': estimated_delivery,
            # 'actual_delivery_date': actual_delivery,
            # 'delivery_delay_days': delay_days
        })

    order_data = df.apply(process_order, axis=1)
    return order_data



def simulate_sales_volume(df, random_state=None):
    """
    Simulates expected sales volume for each product row in a DataFrame based on multiple influencing factors.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing product-level data with the following required columns:
        - 'sub_category': str, product sub-category name
        - 'shelf_life_days': int, shelf life in days
        - 'sales_demand': str, one of ['Very High', 'High', 'Normal', 'Low']
        - 'in_season': bool, whether the product is currently in season
        - 'is_holiday': bool, whether the day is a holiday
        - 'is_weekend': bool, whether the day is a weekend
        - 'weather_severity': str, one of ['Catastrophic', 'Extreme', 'Severe', 'Moderate', 'Normal']

    random_state : int, optional
        Seed for reproducibility of random noise in simulation.

    Returns:
    --------
    pandas.Series
        A Series of simulated sales volumes (integers) for each row in the input DataFrame.

    Example:
    --------
    >>> data = pd.DataFrame({
    ...     'sub_category': ['Bread', 'Fruits'],
    ...     'shelf_life_days': [2, 5],
    ...     'sales_demand': ['High', 'Very High'],
    ...     'in_season': [True, False],
    ...     'is_holiday': [False, True],
    ...     'is_weekend': [True, False],
    ...     'weather_severity': ['Moderate', 'Normal']
    ... })
    >>> simulate_sales_volume(data, random_state=42)
    0    1262
    1    1462
    dtype: int64
    """

    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    def calculate_sales_per_row(row):
        # Base sales volume per sub-category
        base_sales_volume = {
            'Baking Supplies': 10, 'Bread': 117, 'Breakfast Foods': 40, 'Canned Fish': 13, 'Canned Goods': 50,
            'Coffee': 33, 'Condiments': 30, 'Dairy': 107, 'Desserts': 23, 'Dried Fruits': 10, 'Eggs': 93,
            'Fruits': 167, 'Grains & Rice': 60, 'Juices': 53, 'Meat': 133, 'Nuts & Seeds': 13, 'Oils & Vinegars': 27,
            'Pastries': 33, 'Plant-Based Milk': 17, 'Plant-Based Proteins': 8, 'Seafood': 10, 'Snacks': 83,
            'Spices': 20, 'Spreads': 17, 'Sweeteners': 23, 'Tea': 20, 'Vegetables': 160
        }

        # Default base volume if sub-category not found
        base_volume = base_sales_volume.get(row['sub_category'], 50)

        # Turnover multiplier per sub-category
        turnover_multiplier = {
            'Baking Supplies': 0.9, 'Bread': 2.5, 'Breakfast Foods': 1.7, 'Canned Fish': 0.9, 'Canned Goods': 1.5,
            'Coffee': 1.7, 'Condiments': 1.3, 'Dairy': 2.4, 'Desserts': 1.1, 'Dried Fruits': 1.0, 'Eggs': 2.3,
            'Fruits': 2.2, 'Grains & Rice': 1.5, 'Juices': 1.8, 'Meat': 2.1, 'Nuts & Seeds': 1.0, 'Oils & Vinegars': 1.2,
            'Pastries': 1.6, 'Plant-Based Milk': 1.4, 'Plant-Based Proteins': 0.8, 'Seafood': 0.8, 'Snacks': 2.0,
            'Spices': 1.2, 'Spreads': 1.1, 'Sweeteners': 1.2, 'Tea': 1.1, 'Vegetables': 2.2
        }

        # Default turnover factor if sub-category not found
        turnover_factor = turnover_multiplier.get(row['sub_category'], 1.0)

        # Adjust for shelf life urgency
        if row['shelf_life_days'] <= 2:
            turnover_factor *= 3.0
        elif row['shelf_life_days'] <= 3:
            turnover_factor *= 2.5
        elif row['shelf_life_days'] <= 7:
            turnover_factor *= 2.0

        # Adjust for demand level
        demand_boost = {
            'Very High': 1.8,
            'High': 1.5,
            'Normal': 1.0,
            'Low': 0.6
        }
        turnover_factor *= demand_boost.get(str(row['sales_demand']), 1.0)

        # Seasonal and calendar effects
        if row['in_season']:
            turnover_factor *= 1.5
        if row['is_holiday']:
            turnover_factor *= 1.6
        elif row['is_weekend']:
            turnover_factor *= 1.3

        # Weather impact
        weather_multiplier = {
            'Catastrophic': 0.1,
            'Extreme': 0.6,
            'Severe': 0.8,
            'Moderate': 0.95,
            'Normal': 1.0
        }
        turnover_factor *= weather_multiplier.get(row['weather_severity'], 1.0)

        # Final sales potential
        sales_potential = base_volume * turnover_factor

        # Add randomness to simulate real-world variability
        noise = sales_potential * 0.25
        simulated_sales = np.random.normal(sales_potential, noise)

        # Ensure minimum of 1 unit sold
        return int(round(max(1, simulated_sales)))

    # Apply simulation to each row
    return df.apply(calculate_sales_per_row, axis=1)



def simulate_sales_volume_polars(
        df: pl.DataFrame | pl.LazyFrame, 
        base_sales_volume: dict = None,
        turnover_multiplier: dict = None,
        demand_boost: dict = None,
        weather_multiplier: dict = None
    ) -> pl.DataFrame | pl.LazyFrame:
    """
    Simulate sales volume for product subcategories using multiple factors.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Input Polars DataFrame containing product information. 
        Must include columns: 'sub_category', 'shelf_life_days', 'sales_demand',
        'in_season', 'is_holiday', 'is_weekend', 'weather_severity'.
    base_sales_volume : dict, optional
        Dictionary mapping subcategories to their baseline sales volume.
        Defaults to predefined values if not provided.
    turnover_multiplier : dict, optional
        Dictionary mapping subcategories to turnover multipliers.
        Defaults to predefined values if not provided.
    demand_boost : dict, optional
        Dictionary mapping demand levels (e.g., 'High', 'Low') to multipliers.
        Defaults to predefined values if not provided.
    weather_multiplier : dict, optional
        Dictionary mapping weather severity levels to multipliers.
        Defaults to predefined values if not provided.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        A Polars DataFrame with additional column:
        - 'sales_volume': final simulated sales volume (rounded, min=1)
    """

    # Default baseline sales volume per subcategory
    if base_sales_volume is None:
        base_sales_volume = {
            'Baking Supplies': 10, 'Bread': 117, 'Breakfast Foods': 40, 'Canned Fish': 13, 'Canned Goods': 50,
            'Coffee': 33, 'Condiments': 30, 'Dairy': 107, 'Desserts': 23, 'Dried Fruits': 10, 'Eggs': 93,
            'Fruits': 167, 'Grains & Rice': 60, 'Juices': 53, 'Meat': 133, 'Nuts & Seeds': 13, 'Oils & Vinegars': 27,
            'Pastries': 33, 'Plant-Based Milk': 17, 'Plant-Based Proteins': 8, 'Seafood': 10, 'Snacks': 83,
            'Spices': 20, 'Spreads': 17, 'Sweeteners': 23, 'Tea': 20, 'Vegetables': 160
        }

    # Default turnover multipliers per subcategory
    if turnover_multiplier is None:
        turnover_multiplier = {
            'Baking Supplies': 0.9, 'Bread': 2.5, 'Breakfast Foods': 1.7, 'Canned Fish': 0.9, 'Canned Goods': 1.5,
            'Coffee': 1.7, 'Condiments': 1.3, 'Dairy': 2.4, 'Desserts': 1.1, 'Dried Fruits': 1.0, 'Eggs': 2.3,
            'Fruits': 2.2, 'Grains & Rice': 1.5, 'Juices': 1.8, 'Meat': 2.1, 'Nuts & Seeds': 1.0, 'Oils & Vinegars': 1.2,
            'Pastries': 1.6, 'Plant-Based Milk': 1.4, 'Plant-Based Proteins': 0.8, 'Seafood': 0.8, 'Snacks': 2.0,
            'Spices': 1.2, 'Spreads': 1.1, 'Sweeteners': 1.2, 'Tea': 1.1, 'Vegetables': 2.2
        }

    # Default demand boost multipliers
    if demand_boost is None:
        demand_boost = {
            'Very High': 1.8,
            'High': 1.5,
            'Normal': 1.0,
            'Low': 0.6
        }

    # Default weather severity multipliers
    if weather_multiplier is None:
        weather_multiplier = {
            'Catastrophic': 0.1,
            'Extreme': 0.6,
            'Severe': 0.8,
            'Moderate': 0.95,
            'Normal': 1.0
        }

    # Apply transformations step by step
    return (
        df.with_columns(
            # Step 1: Assign baseline volume and initial turnover factor
            base_volume = pl.col("sub_category").replace_strict(base_sales_volume, default=50),
            turnover_factor = pl.col("sub_category").replace_strict(turnover_multiplier, default=1.0)
        ).with_columns(
            # Step 2: Apply multipliers (shelf life, demand, seasonality, holiday/weekend, weather)
            turnover_factor = (
                pl.col("turnover_factor")
                # Shelf life effect
                * pl.when(pl.col("shelf_life_days") <= 2).then(3.0)
                 .when(pl.col("shelf_life_days") <= 3).then(2.5)
                 .when(pl.col("shelf_life_days") <= 7).then(2.0)
                 .otherwise(1.0)
                # Demand effect
                * pl.col("sales_demand").replace_strict(demand_boost, default=1.0)
                # Seasonality effect
                * pl.when(pl.col("in_season")).then(1.5).otherwise(1.0)
                # Holiday/Weekend effect
                * pl.when(pl.col("is_holiday")).then(1.6)
                 .when(pl.col("is_weekend")).then(1.3)
                 .otherwise(1.0)
                # Weather effect
                * pl.col("weather_severity").replace_strict(weather_multiplier, default=1.0)
            )
        ).with_columns(
            # Step 3: Add random noise to simulate variability
            sales_volume = (
                pl.col("base_volume") 
                + (pl.Series(np.random.normal(loc=0.0, scale=2.5, size=df.collect().height)) 
                   * pl.col("turnover_factor")) 
            )
        ).with_columns(
            # Step 4: Round values and enforce minimum of 0
            sales_volume = (
                pl.when(pl.col("sales_volume") < 1)
                 .then(0)
                 .otherwise(pl.col("sales_volume"))
                 .round(0)
                 .cast(pl.Int64)
            )
        )
    ).drop(["base_volume", "turnover_factor"])



def simulate_stock_quantity(row: dict):
    """
    Simulates the stock quantity based on demand and seasonality factors.
    Incorporates randomness and scenarios of stock shortage or surplus.

    Parameters:
        row (dict): A dictionary containing product-related data, including:
            - 'in_season' (bool): Indicates if the product is in season.
            - 'max_stock' (float): Maximum stock level.
            - 'min_stock' (float): Minimum stock level.
            - 'weather_severity' (str): Weather impact level ('High', 'Low', etc.).
            - 'adjusted_demand' (float): Demand adjusted for current conditions.

    Returns:
        int: Simulated stock quantity, ensuring non-negative values.

    Example:
        >>> import numpy as np
        >>> row = {
        ...     'in_season': True,
        ...     'max_stock': 100,
        ...     'min_stock': 30,
        ...     'weather_severity': 'Low',
        ...     'adjusted_demand': 60
        ... }
        >>> simulate_stock_quantity(row)
        45  # (actual output may vary due to randomness)
    """
    # Set base stock depending on seasonality
    if row['in_season']:
        base_stock = row['max_stock'] * 0.8
    else:
        base_stock = row['min_stock'] * 1.5

    # Adjust base stock according to weather severity
    if row['weather_severity'] == 'High':
        base_stock *= 0.7  # Severe weather may hinder restocking
    elif row['weather_severity'] == 'Low':
        base_stock *= 1.1  # Favorable weather may allow higher stock

    # Calculate initial stock and add random noise
    simulated_stock = base_stock - row['adjusted_demand'] + np.random.normal(0, 5)

    # Introduce intentional scenarios of stock shortage or surplus
    rand_val = np.random.rand()
    if rand_val < 0.1:  # 10% chance of stock shortage
        return np.random.randint(0, int(row['min_stock'] * 0.8))
    elif rand_val > 0.95:  # 5% chance of stock surplus
        return np.random.randint(int(row['max_stock'] * 1.2), int(row['max_stock'] * 1.5))
    else:
        # Ensure the final value is non-negative
        return max(0, int(simulated_stock))



def classify_grocery_demand(dates: pd.Series, country: str = 'BR') -> pd.Series:
    """
    Classifies grocery demand levels based on date characteristics and national holidays.

    Parameters:
    ----------
    dates : pd.Series
        A pandas Series of datetime objects representing transaction or delivery dates.
    country : str, optional
        A two-letter country code (ISO 3166-1 alpha-2) used to determine holidays.
        Defaults to 'BR' (Brazil).

    Returns:
    -------
    pd.Series
        A Series of strings indicating demand classification for each date:
        - 'High (Holiday)' if the date is a national holiday
        - 'High (Beginning of Month)' if the day is between the 1st and 5th
        - 'High (Weekend)' if the date falls on Saturday or Sunday
        - 'Normal' otherwise

    Example:
    -------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> dates = pd.Series([datetime(2025, 1, 1), datetime(2025, 1, 3), datetime(2025, 1, 4), datetime(2025, 1, 6)])
    >>> classify_grocery_demand(dates, country='BR')
    0         High (Holiday)
    1    High (Beginning of Month)
    2         High (Weekend)
    3                Normal
    dtype: object
    """
    country_holidays = holidays.country_holidays(country)

    def classify_single(date):
        if date in country_holidays:
            return 'Very High'
        elif 1 <= date.day <= 5:
            return 'High'
        elif date.weekday() >= 5:  # Saturday or Sunday
            return 'High'
        else:
            return 'Normal'
    
    return dates.apply(classify_single)


def classify_grocery_demand_polars(
    df: pl.DataFrame | pl.LazyFrame, 
    columns_date: str, 
    country: str = 'BR'
) -> pl.DataFrame | pl.LazyFrame:
    """
    Classify grocery sales demand levels based on dates and holidays.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input Polars DataFrame or LazyFrame containing grocery sales data.
    columns_date : str
        Name of the column containing date values.
    country : str, optional
        Country code (default is 'BR') used to determine national holidays.

    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Same type as input, with an additional column 'sales_demand'
        categorizing demand as "Very High", "High", or "Normal".
    """

    # Detect if input is LazyFrame
    is_lazy = isinstance(df, pl.LazyFrame)

    # Get unique years (precisa coletar apenas esse resultado)
    years = df.select(pl.col(columns_date).dt.year().unique()).collect()[columns_date]
    
    country_holidays = holidays.country_holidays(country.upper(), years=years)

    # Define classification expression (Expr)
    demand_expr = (
        pl.when(pl.col(columns_date).is_in(country_holidays))
        .then(pl.lit("Very High"))
        .when(pl.col(columns_date).dt.day().is_between(1, 10))
        .then(pl.lit("High"))
        .when(pl.col(columns_date).dt.weekday() >= 6)  # Saturday = 6, Sunday = 7
        .then(pl.lit("High"))
        .otherwise(pl.lit("Normal"))
        .alias("sales_demand")
    )

    # Apply expression
    df = df.with_columns(demand_expr)

    # Return in same format as input
    return df.lazy() if is_lazy else df





def day_classification(dates: pd.Series, country: str = 'BR') -> pd.Series:
    """
    Classifies each date in a pandas Series as 'Holiday', 'Saturday', 'Sunday', or 'Weekdays' 
    based on the specified country's holiday calendar.

    Parameters:
    ----------
    dates : pd.Series
        A pandas Series of datetime objects to classify.
    country : str, optional
        A two-letter country code (ISO 3166-1 alpha-2) used to determine holidays. 
        Defaults to 'BR' (Brazil).

    Returns:
    -------
    pd.Series
        A Series of strings indicating the classification of each date.

    Example:
    -------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> dates = pd.Series([datetime(2025, 1, 1), datetime(2025, 1, 4), datetime(2025, 1, 5), datetime(2025, 1, 6)])
    >>> day_classification(dates, country='BR')
    0     Holiday
    1    Saturday
    2      Sunday
    3    Weekdays
    dtype: object
    """
    country_holidays = holidays.country_holidays(country)

    def classify_single(date):
        if date in country_holidays:
            return 'Holiday'
        elif date.dayofweek == 5:
            return 'Saturday'
        elif date.dayofweek == 6:
            return 'Sunday'
        else:
            return 'Weekdays'
        
    return dates.apply(classify_single)


def day_classification_lazy(df: pl.DataFrame | pl.LazyFrame, col: str, country: str = "BR") -> pl.DataFrame | pl.LazyFrame:
    """
    Classify days in a datetime column as holidays, weekends, or weekdays.
    """
    
    # Extract all unique years present in the datetime column
    years = df.select(pl.col(col).dt.year().unique()).collect().to_series().to_list()
    
    # Retrieve holidays for the specified country and years
    country_holidays = holidays.country_holidays(country.upper(), years=years)
    
    # Convert holiday dates into a list and then to Polars Series
    holiday_dates = list(country_holidays.keys())
    holiday_dates_pl = pl.Series(holiday_dates).cast(pl.Date)
    
    # Add classification columns
    return (
        df.with_columns([
            # Flag if the date is a holiday
            pl.col(col).is_in(holiday_dates_pl).alias("is_holiday"),
            
            # Classify the day of the week
            pl.when(pl.col(col).dt.weekday() == 6)
            .then(pl.lit("Saturday"))
            .when(pl.col(col).dt.weekday() == 7)
            .then(pl.lit("Sunday"))
            .otherwise(pl.lit("Weekday"))
            .alias("day_classification"),
            
            # Flag if the day is a weekend
            pl.col(col).dt.weekday().is_in([6, 7]).alias("is_weekend")
        ])
    )




def create_stock_distribution_vectorized(stock_min, stock_max, seed: int=None, 
                                         prob_stock: list=[0.12, 0.28, 0.60],
                                         prob_extreme: list=[0.68, 0.27, 0.05]):
    """
    Generates a vector of stock quantities based on probabilistic conditions: 'out of stock', 'overstocked', or 'normal',
    applied element-wise to input vectors.

    Parameters:
    ----------
    stock_min : pandas.Series
        Series of minimum stock quantities for each item.
    stock_max : pandas.Series
        Series of maximum stock quantities for each item.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
    prob_stock : list of float, optional pl.DataFrame | pl.LazyFrame
        Probabilities for each stock condition: ['out', 'over', 'normal'] respectively.
        Default is [0.12, 0.28, 0.60].
    prob_extreme : list of float, optional
        Probabilities for selecting intervals within 'out' and 'over' conditions.
        Default is [0.68, 0.27, 0.05].

    Returns:
    -------
    numpy.ndarray
        Array of generated stock quantities for each item.

    Stock Conditions:
    -----------------
    - 'normal': Random integer between stock_min[i] and stock_max[i].
    - 'over': Value above stock_max[i] using a multiplier from a probabilistic interval.
    - 'out': Value below stock_min[i] using a divider from a probabilistic interval.

    Example:
    --------
    >>> import pandas as pd
    >>> stock_min = pd.Series([50, 30, 20])
    >>> stock_max = pd.Series([100, 80, 60])
    >>> create_stock_distribution_vectorized(stock_min, stock_max, seed=42)
    array([  0,  94,  44])  # (actual values may vary depending on condition and seed)
    """

    # Initialize random number generator with optional seed
    rng = np.random.default_rng(seed=seed)
    n = len(stock_min)

    # Define possible stock conditions
    stock_condition = ['out', 'over', 'normal']

    # Randomly assign a condition to each item
    conditions = rng.choice(stock_condition, size=n, p=prob_stock)

    # Initialize result array
    results = np.zeros(n, dtype=int)

    # Process each item based on its assigned condition
    for i, condition in enumerate(conditions):
        if condition == 'normal':
            # Generate stock within normal range
            results[i] = rng.integers(stock_min.iloc[i], stock_max.iloc[i] + 1)

        elif condition == 'over':
            # Define intervals for overstock multipliers
            multipliers_intervals = [(1.05, 1.15), (1.16, 1.30), (1.31, 1.70)]
            # Select an interval based on extreme probabilities
            chosen_interval = rng.choice(multipliers_intervals, p=prob_extreme)
            # Generate multiplier and apply to stock_max
            multiplier = rng.uniform(chosen_interval[0], chosen_interval[1])
            results[i] = int(np.ceil(stock_max.iloc[i] * multiplier))

        elif condition == 'out':
            # Define intervals for out-of-stock dividers
            dividers_intervals = [(0.05, 0.15), (0.16, 0.30), (0.31, 0.70)]
            # Select an interval based on extreme probabilities
            chosen_interval = rng.choice(dividers_intervals, p=prob_extreme)
            # Generate divider and apply to stock_min
            divider = rng.uniform(chosen_interval[0], chosen_interval[1])
            results[i] = int(np.floor(stock_min.iloc[i] * divider))

    return results


def create_stock_distribution_polars(df: pl.DataFrame | pl.LazyFrame,
                                     min_stock: str = "min_stock",
                                     max_stock: str = "max_stock",
                                     prob_stock: list = [0.11, 0.25, 0.60, 0.04],
                                     stock_conditions: list = ['out', 'over', 'normal', 'extreme'],
                                     seed: int = None
                                     ) -> pl.DataFrame | pl.LazyFrame:
    """
    Generate a stock distribution simulation using Polars DataFrame or LazyFrame.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Input Polars DataFrame or LazyFrame containing stock columns.
    min_stock : str, default="min_stock"
        Column name representing the minimum stock value.
    max_stock : str, default="max_stock"
        Column name representing the maximum stock value.
    prob_stock : list, default=[0.11, 0.25, 0.60, 0.04]
        Probability distribution for each stock condition.
        Must align with `stock_conditions` order.
    stock_conditions : list, default=['out', 'over', 'normal', 'extreme']
        Possible stock condition categories.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Original DataFrame with additional columns:
        - "condition": chosen stock condition
        - "extreme_condition": random extreme multiplier
        - "randon_factor": random float between 0 and 1
        - "stock_quantity": simulated stock quantity based on condition
    """

    # Initialize random number generator with optional seed
    rng = np.random.default_rng(seed=seed)

    # Get the number of rows in the DataFrame
    length = df.collect().height

    # Randomly assign conditions based on given probabilities
    conditions = rng.choice(stock_conditions, size=length, p=prob_stock)

    # Generate extreme condition multipliers (always >= 1.0)
    extreme_conditions = abs(rng.normal(scale=1.08, size=length)) + 1.0

    # Generate random factors between 0 and 1
    randon_factor = rng.random(length)

    # Add new columns and compute stock_quantity based on condition rules
    return (
        df.with_columns(
            pl.Series("condition", conditions),
            pl.Series("extreme_condition", extreme_conditions),
            pl.Series("randon_factor", randon_factor)
        )
        .with_columns(
            pl.when(pl.col("condition") == "out")
            .then(0)  # No stock available
            .when(pl.col("condition") == "over")
            .then(pl.col(max_stock) * (pl.col("extreme_condition")))  # Overstock scenario
            .when(pl.col("condition") == "extreme")
            .then(pl.col(max_stock) * (pl.col("extreme_condition") + 0.8))  # Extreme overstock
            .when(pl.col("condition") == "normal")
            .then(pl.col(min_stock)
                  + (pl.col(max_stock) - pl.col(min_stock))
                  * pl.col("randon_factor"))  # Normal distribution between min and max
            .otherwise(pl.col(min_stock))  # Fallback to minimum stock
            .cast(pl.Int32)        
            .alias("stock_quantity")
        )
    ).drop(["condition", "extreme_condition", "randon_factor"])






def create_suppliers(storage_df:pd.DateOffset, cost_price_df:pd.DataFrame, products_df:pd.DataFrame, suppliers_df:pd.DataFrame):
    """
        Generates a supplier-product relationship table with realistic logistics and commercial attributes.

        This function links suppliers to products based on category, assigning one primary supplier and
        multiple secondary suppliers per product. It uses lead time and shelf life intervals to simulate
        delivery dynamics and applies randomized commercial parameters such as price variation, reliability,
        payment terms, and quality ratings.

        Parameters:
            storage_df (pd.DataFrame): Contains category-level lead time and shelf life intervals.
            cost_price_df (pd.DataFrame): Contains product names and their base supply prices.
            products_df (pd.DataFrame): Contains product metadata including product ID and category.
            suppliers_df (pd.DataFrame): Contains supplier metadata including supplier ID and category.

        Returns:
            pd.DataFrame: A DataFrame with supplier-product relationships, including:
                - supplier_id
                - product_id
                - supply_price
                - is_primary_supplier
                - lead_time_days
                - min_order_quantity
                - reliability_score
                - payment_terms_days
                - quality_rating
    """
    # Function to convert a string interval like '[min, max]' into a tuple (min, max)
    def parse_interval(interval_str):
        """Converts string '[min, max]' to tuple (min, max)"""
        try:
            if isinstance(interval_str, str):
                return ast.literal_eval(interval_str)  # Safely evaluate string to tuple
            return interval_str  # If already a tuple, return as-is
        except:
            return (1, 30)  # Default fallback range

    # Apply interval parsing to lead time and shelf life columns
    storage_df['lead_time_range'] = storage_df['lead_time_days'].apply(parse_interval)
    storage_df['shelf_life_range'] = storage_df['shelf_life_days'].apply(parse_interval)

    # Create dictionaries mapping each category to its lead time and shelf life ranges
    category_lead_times = {}
    category_shelf_lives = {}
    for _, row in storage_df.iterrows():
        category_lead_times[row['category']] = row['lead_time_range']
        category_shelf_lives[row['category']] = row['shelf_life_range']

    # Initialize list to store supplier-product relationship records
    supplier_product_relationships = []

    # Iterate through each product in the pricing dataset
    for _, price_row in cost_price_df.iterrows():
        product_name = price_row['product']
        main_supply_price = price_row['price']
        
        # Get product details from products_df
        product_info = products_df[products_df['product'] == product_name]
        if len(product_info) == 0:
            continue  # Skip if product not found
        
        product_id = product_info['product_id'].values[0]
        product_category = product_info['category'].values[0]
        
        # Get lead time range for the product's category
        lead_time_range = category_lead_times.get(product_category, (1, 10))
        
        # Filter suppliers that match the product category
        category_suppliers = suppliers_df[suppliers_df['category'] == product_category]

        # Initialize random number generator with fixed seed for reproducibility
        rng = np.random.default_rng(seed=42)
        
        if len(category_suppliers) > 0:
            # Sort suppliers to consistently select the first as the main supplier
            category_suppliers = category_suppliers.sort_values('supplier_id')
            
            # Assign main supplier
            main_supplier = category_suppliers.iloc[0]
            main_supplier_id = main_supplier['supplier_id']
            
            # Generate lead time for main supplier within a narrow range
            main_lead_time = rng.integers(lead_time_range[0], lead_time_range[0] + 3)
            
            # Add main supplier relationship entry
            supplier_product_relationships.append({
                'supplier_id': main_supplier_id,
                'product_id': product_id,
                'supply_price': main_supply_price,
                'is_primary_supplier': True,
                'lead_time_days': main_lead_time,
                'min_order_quantity': rng.choice([3, 5, 10]),
                'reliability_score': round(rng.uniform(0.85, 0.95), 2),
                'payment_terms_days': 30,
                'quality_rating': rng.choice([4, 5], p=[0.3, 0.7])
            })
            
            # Process secondary suppliers
            secondary_suppliers = category_suppliers.iloc[1:]
            
            for _, supplier in secondary_suppliers.iterrows():
                supplier_id = supplier['supplier_id']
                supplier_name = supplier['supplier']
                
                # Define valid lead time range for secondary suppliers
                valid_range = range(max(1, lead_time_range[0] + 1), lead_time_range[1])
                
                # Adjust price and lead time based on supplier type
                if 'Distributor' in supplier_name:
                    variation_factor = rng.uniform(0.95, 1.15)
                    lead_time = rng.choice(valid_range)
                else:
                    variation_factor = rng.uniform(0.9, 1.2)
                    lead_time = rng.integers(lead_time_range[0], lead_time_range[1])
                
                # Calculate secondary supplier price with bounds
                secondary_price = round(main_supply_price * variation_factor, 2)
                secondary_price = max(secondary_price, main_supply_price * 0.5)
                secondary_price = min(secondary_price, main_supply_price * 2.0)
                
                # Add secondary supplier relationship entry
                supplier_product_relationships.append({
                    'supplier_id': supplier_id,
                    'product_id': product_id,
                    'supply_price': secondary_price,
                    'is_primary_supplier': False,
                    'lead_time_days': lead_time,
                    'min_order_quantity': rng.choice([5, 10, 25, 50]),
                    'reliability_score': round(rng.uniform(0.7, 0.9), 2),
                    'payment_terms_days': rng.choice([15, 30, 45, 60]),
                    'quality_rating': rng.choice([2, 3, 4], p=[0.2, 0.5, 0.3])
                })

    # Convert the list of relationships into a DataFrame
    supplier_products_df = pd.DataFrame(supplier_product_relationships)

    # Return the final DataFrame
    return supplier_products_df




def random_shelf_life(range_list):
    """
        Generate a random integer shelf life value within a specified range.

        Parameters:
            range_list (list or tuple): A two-element list or tuple containing the minimum and maximum 
                                        shelf life values (e.g., [10, 30] or (10, 30)).

        Returns:
            int: A randomly selected integer between the lower bound (inclusive) and upper bound (exclusive).

        Example:
            random_shelf_life([10, 30]) → 17
    """
    
    # Create a random number generator with a fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)
    
    return rng.integers(range_list[0], range_list[1])





def create_supplier_cat(categories: list, categories_prob: list, supplier_pool: list) -> dict:
    """
    Assigns a random category to each supplier based on a given probability distribution.

    Parameters:
    - categories (list): A list of supply chain categories (e.g., 'Produce', 'Dairy and Cold Cuts').
    - categories_prob (list): A list of probabilities corresponding to each category. Must sum to 1.
    - supplier_pool (list): A list of supplier names.

    Returns:
    - dict: A dictionary mapping each supplier to a randomly assigned category.
    """

    # Remove duplicates, if any
    supplier_pool = list(set(supplier_pool))

    # Create dictionary with randomly assigned categories based on probabilities
    supplier_dict = {
        supplier: np.random.choice(categories, p=categories_prob)
        for supplier in supplier_pool
    }

    return supplier_dict




def create_IDs(length: int, suffix: str):
    """
    Generates a list of unique identifiers in the format 'number|suffix'.

    The IDs are created by adding a base value (1e6) to a random sample of unique
    integers in the range [0, 1e6). The suffix is appended to each number using the '|' separator.

    Parameters:
    ----------
    length : int
        Number of unique IDs to generate.
    suffix : str
        Suffix to be appended to each ID.

    Returns:
    -------
    np.ndarray
        Array of strings containing the unique IDs in the format 'number|suffix'.
    """

    # Base value to ensure large and consistent numbers
    base_id = int(1e6)

    # Generate 'length' unique random integers in the range [0, 1e6)
    timestamps = np.random.choice(int(1e6), size=length, replace=False)

    # Add the base value to the generated numbers to create numeric IDs
    number = base_id + timestamps

    # Format each number as a string with the suffix, separated by '|'
    return np.array([f'{num}|{suffix}' for num in number])
