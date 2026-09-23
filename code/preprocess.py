
"""PRT565 - Climate and population preprocessing.

Source data:
1. Berkeley Earth country-level temperature data
2. World Population dataset
"""

from pathlib import Path
import json
import re
import unicodedata

import numpy as np
import pandas as pd



# Project configuration


REFERENCE_YEARS = (1970, 1980, 1990, 2000)

ANNUAL_START = 1960
ANNUAL_END = 2009

MIN_MONTHS_PER_YEAR = 12
YEARS_PER_DECADE = 10

TEST_REFERENCE_YEAR = 2000

CLIMATE_COLUMNS = {
    "dt",
    "AverageTemperature",
    "AverageTemperatureUncertainty",
    "Country",
}

POPULATION_COLUMNS = {
    "CCA3",
    "Country/Territory",
    "Continent",
    "Area (km²)",
} | {
    f"{year} Population"
    for year in (1970, 1980, 1990, 2000, 2010)
}



# Load and audit raw datasets


def load_and_audit(raw_dir: Path):

    climate_path = (
        raw_dir / "GlobalLandTemperaturesByCountry.csv"
    )

    population_path = (
        raw_dir / "world_population.csv"
    )

    for path in (climate_path, population_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {path}"
            )

    climate = pd.read_csv(
        climate_path,
        low_memory=False
    )

    population = pd.read_csv(
        population_path,
        low_memory=False
    )

    # Check required columns before processing.

    for name, frame, required in (
        ("Climate", climate, CLIMATE_COLUMNS),
        ("Population", population, POPULATION_COLUMNS),
    ):
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} dataset is missing: {sorted(missing)}"
            )

    # Check duplicate identifiers.

    if climate.duplicated(["Country", "dt"]).any():
        raise ValueError(
            "Duplicate climate country-date records found."
        )

    if population["CCA3"].duplicated().any():
        raise ValueError(
            "Duplicate population country codes found."
        )

    # Convert dates and numerical columns.

    climate["dt"] = pd.to_datetime(
        climate["dt"],
        errors="coerce"
    )

    if climate["dt"].isna().any():
        raise ValueError("Invalid climate dates found.")

    for column in (
        "AverageTemperature",
        "AverageTemperatureUncertainty",
    ):
        climate[column] = pd.to_numeric(
            climate[column],
            errors="coerce"
        )

    numeric_population_columns = [
        "Area (km²)",
        *[
            f"{year} Population"
            for year in (1970, 1980, 1990, 2000, 2010)
        ],
    ]

    for column in numeric_population_columns:
        population[column] = pd.to_numeric(
            population[column],
            errors="coerce"
        )

    if (
        population["Area (km²)"].isna().any()
        or (population["Area (km²)"] <= 0).any()
    ):
        raise ValueError(
            "Population data contains invalid land areas."
        )

    if population["CCA3"].isna().any():
        raise ValueError("Missing country codes found.")

    # Preserve original missingness information.

    audit = {
        "climate_rows": int(len(climate)),
        "population_rows": int(len(population)),
        "climate_countries": int(
            climate["Country"].nunique()
        ),
        "population_countries": int(
            population["CCA3"].nunique()
        ),
        "missing_climate_values": (
            climate.isna().sum().to_dict()
        ),
        "missing_population_values": (
            population.isna().sum().to_dict()
        ),
        "climate_start_date": str(
            climate["dt"].min().date()
        ),
        "climate_end_date": str(
            climate["dt"].max().date()
        ),
    }

    return climate, population, audit