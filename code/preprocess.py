
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


# Country-name standardization

# Manually reviewed naming differences.
# Do not automatically assume similar names
# describe equivalent geographic populations.

COUNTRY_ALIASES = {
    "burma": "myanmar",
    "congo": "republic of the congo",
    "congo democratic republic of the": "dr congo",
    "cote d ivoire": "ivory coast",
    "falkland islands islas malvinas": "falkland islands",
    "federated states of micronesia": "micronesia",
    "macedonia": "north macedonia",
    "palestina": "palestine",
    "swaziland": "eswatini",
    "turks and caicas islands": "turks and caicos islands",
    "virgin islands": "united states virgin islands",
}


def normalize_country(name):
    """Normalize formatting without fuzzy matching."""

    name = unicodedata.normalize(
        "NFKD", str(name)
    )

    name = "".join(
        char for char in name
        if not unicodedata.combining(char)
    )

    name = name.casefold()
    name = name.replace("&", " and ")

    name = re.sub(
        r"[^a-z0-9]+",
        " ",
        name
    )

    return name.strip()


def make_country_crosswalk(
    climate,
    population
):
    """Create an auditable climate-to-population crosswalk."""

    pop_reference = population[
        ["CCA3", "Country/Territory"]
    ].copy()

    pop_reference["join_key"] = (
        pop_reference["Country/Territory"]
        .map(normalize_country)
    )

    if pop_reference["join_key"].duplicated().any():
        raise ValueError(
            "Population country names have duplicate normalized keys."
        )

    climate_reference = (
        climate[["Country"]]
        .drop_duplicates()
        .rename(
            columns={"Country": "climate_country"}
        )
    )

    climate_reference["normalized_name"] = (
        climate_reference["climate_country"]
        .map(normalize_country)
    )

    climate_reference["join_key"] = (
        climate_reference["normalized_name"]
        .replace(COUNTRY_ALIASES)
    )

    crosswalk = climate_reference.merge(
        pop_reference,
        on="join_key",
        how="left",
        validate="many_to_one",
        indicator=True
    )

    crosswalk = crosswalk.rename(
        columns={
            "CCA3": "cca3",
            "Country/Territory": "population_country",
        }
    )

    crosswalk["matching_method"] = np.select(
        [
            crosswalk["_merge"].eq("left_only"),
            crosswalk["normalized_name"].isin(
                COUNTRY_ALIASES
            ),
            crosswalk["climate_country"].eq(
                crosswalk["population_country"]
            ),
        ],
        [
            "unmatched",
            "manual_alias",
            "exact",
        ],
        default="normalized"
    )

    matched = crosswalk[
        crosswalk["_merge"].eq("both")
    ].copy()

    if matched["cca3"].duplicated().any():
        raise ValueError(
            "Multiple climate locations map to one country. "
            "Review the country crosswalk."
        )

    return (
        matched.drop(columns="_merge"),
        crosswalk.drop(columns="_merge")
    )



# Annual temperature aggregation


def annual_climate_table(
    climate,
    matched_countries
):
    """Calculate annual means from complete monthly records."""

    climate_matched = climate.merge(
        matched_countries[
            ["climate_country", "cca3"]
        ],
        left_on="Country",
        right_on="climate_country",
        how="inner",
        validate="many_to_one"
    )

    climate_matched["year"] = (
        climate_matched["dt"].dt.year
    )

    # Restrict the period to the historical
    # decades needed for the project.

    climate_matched = climate_matched[
        climate_matched["year"].between(
            ANNUAL_START,
            ANNUAL_END
        )
    ].copy()

    annual = climate_matched.groupby(
        ["cca3", "year"],
        as_index=False
    ).agg(
        annual_mean_temp_c=(
            "AverageTemperature",
            "mean"
        ),
        valid_months=(
            "AverageTemperature",
            "count"
        ),
        annual_mean_uncertainty_c=(
            "AverageTemperatureUncertainty",
            "mean"
        )
    )

    # Audit incomplete country-years.

    incomplete = annual[
        annual["valid_months"] < MIN_MONTHS_PER_YEAR
    ].copy()

    # Keep complete annual observations only.

    annual = annual[
        annual["valid_months"] == MIN_MONTHS_PER_YEAR
    ].copy()

    if annual.duplicated(
        ["cca3", "year"]
    ).any():
        raise AssertionError(
            "Duplicate annual climate observations."
        )

    return (
        annual.sort_values(
            ["cca3", "year"]
        ).reset_index(drop=True),
        incomplete
    )



# Decadal climate feature engineering


def compute_decadal_climate(annual):
    """Calculate warming trends and temperature volatility."""

    records = []

    for cca3, group in annual.groupby("cca3"):

        country = group.set_index("year")

        for start in (1960, 1970, 1980, 1990, 2000):

            years = np.arange(
                start,
                start + YEARS_PER_DECADE
            )

            # Require all ten complete annual values.

            if not set(years).issubset(country.index):
                continue

            decade = country.loc[years]

            temperatures = (
                decade["annual_mean_temp_c"]
                .to_numpy(dtype=float)
            )

            centered_years = (
                years.astype(float) - years.mean()
            )

            # Ordinary least-squares trend.

            slope_per_year = (
                np.dot(
                    centered_years,
                    temperatures - temperatures.mean()
                )
                / np.dot(
                    centered_years,
                    centered_years
                )
            )

            fitted = (
                temperatures.mean()
                + slope_per_year * centered_years
            )

            residuals = temperatures - fitted

            records.append({
                "cca3": cca3,
                "decade_start": start,
                "decade_end": start + 9,
                "mean_temp_c": float(
                    temperatures.mean()
                ),
                "warming_c_per_decade": float(
                    slope_per_year * 10
                ),
                "detrended_volatility_c": float(
                    residuals.std(ddof=1)
                ),
                "mean_uncertainty_c": float(
                    decade[
                        "annual_mean_uncertainty_c"
                    ].mean()
                ),
            })

    decades = pd.DataFrame(records)

    if decades.empty:
        raise ValueError(
            "No complete climate decades were generated."
        )

    if decades.duplicated(
        ["cca3", "decade_start"]
    ).any():
        raise AssertionError(
            "Duplicate country-decade records."
        )

    return decades.sort_values(
        ["cca3", "decade_start"]
    ).reset_index(drop=True)



# Historical population feature engineering


def historical_population_panel(population):
    """Convert population snapshots into country-year rows."""

    population_columns = [
        f"{year} Population"
        for year in (1970, 1980, 1990, 2000, 2010)
    ]

    panel = population.melt(
        id_vars=[
            "CCA3",
            "Country/Territory",
            "Continent",
            "Area (km²)",
        ],
        value_vars=population_columns,
        var_name="population_year",
        value_name="population"
    )

    panel = panel.rename(
        columns={
            "CCA3": "cca3",
            "Country/Territory": "country",
            "Continent": "continent",
            "Area (km²)": "area_km2",
        }
    )

    panel["population_year"] = (
        panel["population_year"]
        .str.extract(r"(\d{4})")
        .astype(int)
    )

    panel = panel.sort_values(
        ["cca3", "population_year"]
    ).reset_index(drop=True)

    # Historical population density.

    panel["density_per_km2"] = (
        panel["population"] / panel["area_km2"]
    )

    # Previous population snapshot for each country.

    previous_population = (
        panel.groupby("cca3")["population"]
        .shift(1)
    )

    # Percentage growth over the previous decade.

    panel["growth_prior_decade_pct"] = (
        (
            panel["population"]
            / previous_population
        ) - 1
    ) * 100

    # No 1960 population exists in this dataset,
    # so prior growth for 1970 remains unknown.

    panel.loc[
        panel["population_year"] == 1970,
        "growth_prior_decade_pct"
    ] = np.nan

    if panel[
        ["population", "density_per_km2"]
    ].isna().any().any():
        raise ValueError(
            "Missing historical population values."
        )

    if (
        panel["population"] <= 0
    ).any():
        raise ValueError(
            "Nonpositive population values found."
        )

    if panel.duplicated(
        ["cca3", "population_year"]
    ).any():
        raise AssertionError(
            "Duplicate country-year population records."
        )

    return panel