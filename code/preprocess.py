
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

    # European climate labels matched to the population-country names.
    "united kingdom europe": "united kingdom",
    "france europe": "france",
    "denmark europe": "denmark",
    "netherlands europe": "netherlands",
}


# The Berkeley Earth file contains both a normal label and a Europe-only
# label for these countries. The population dataset represents the European
# country geography, so we keep the (Europe) climate record and exclude the
# broader alternative climate label from modelling.
EXCLUDED_ALTERNATIVE_CLIMATE_LABELS = {
    "United Kingdom",
    "France",
    "Denmark",
    "Netherlands",
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
            crosswalk["climate_country"].isin(
                EXCLUDED_ALTERNATIVE_CLIMATE_LABELS
            ),
            crosswalk["_merge"].eq("left_only"),
            crosswalk["normalized_name"].isin(
                COUNTRY_ALIASES
            ),
            crosswalk["climate_country"].eq(
                crosswalk["population_country"]
            ),
        ],
        [
            "excluded_alternative",
            "unmatched",
            "manual_alias",
            "exact",
        ],
        default="normalized"
    )

    # Use only confirmed matches that are not the excluded alternative
    # climate labels. This leaves exactly one climate source per country.
    matched = crosswalk[
        crosswalk["_merge"].eq("both")
        & ~crosswalk["climate_country"].isin(
            EXCLUDED_ALTERNATIVE_CLIMATE_LABELS
        )
    ].copy()

    if matched["cca3"].duplicated().any():
        duplicates = matched.loc[
            matched["cca3"].duplicated(keep=False),
            ["climate_country", "population_country", "cca3"]
        ].sort_values("cca3")

        raise ValueError(
            "Multiple climate locations still map to one country after "
            "applying the preferred-source rules. Review the crosswalk.\n"
            + duplicates.to_string(index=False)
        )

    return (
        matched.drop(columns="_merge"),
        crosswalk.drop(columns="_merge")
    )


def find_population_only_countries(matched_countries, population):
    """Find population countries/territories that have no matched climate record.

    The crosswalk keeps all climate country names using a left join so that
    unmatched climate names remain visible. This complementary audit checks
    the opposite direction as well, so population entries without a confirmed
    climate match are also documented.
    """

    matched_codes = set(
        matched_countries["cca3"].dropna()
    )

    population_only = population.loc[
        ~population["CCA3"].isin(matched_codes),
        ["CCA3", "Country/Territory", "Continent"]
    ].copy()

    return population_only.sort_values(
        "Country/Territory"
    ).reset_index(drop=True)



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

    # One preferred climate source is retained per country, so no averaging
    # between normal and (Europe) labels is needed here.
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


# Climate and population integration


def build_country_periods(
    decadal_climate,
    population_panel
):
    """Construct historical predictor and outcome periods."""

    # Select the population at each reference year.

    current_population = population_panel[
        population_panel["population_year"].isin(
            REFERENCE_YEARS
        )
    ].copy()

    current_population = current_population.rename(
        columns={
            "population_year": "reference_year",
            "population": "population_at_reference",
            "density_per_km2": (
                "density_at_reference_per_km2"
            ),
            "growth_prior_decade_pct": (
                "population_growth_prior_decade_pct"
            ),
        }
    )

    # Historical climate predictors end before
    # the reference year.

    current_population["prior_decade_start"] = (
        current_population["reference_year"] - 10
    )

    previous_climate = decadal_climate.rename(
        columns={
            "decade_start": "prior_decade_start",
            "mean_temp_c": "prior_decade_mean_temp_c",
            "warming_c_per_decade": (
                "prior_decade_warming_c_per_decade"
            ),
            "detrended_volatility_c": (
                "prior_decade_detrended_volatility_c"
            ),
            "mean_uncertainty_c": (
                "prior_decade_mean_temp_uncertainty_c"
            ),
        }
    )

    climate_predictor_columns = [
        "cca3",
        "prior_decade_start",
        "prior_decade_mean_temp_c",
        "prior_decade_warming_c_per_decade",
        "prior_decade_detrended_volatility_c",
        "prior_decade_mean_temp_uncertainty_c",
    ]

    periods = current_population.merge(
        previous_climate[climate_predictor_columns],
        on=["cca3", "prior_decade_start"],
        how="inner",
        validate="many_to_one"
    )

    # Following-decade warming is an outcome,
    # never a predictor.

    future_climate = decadal_climate[
        [
            "cca3",
            "decade_start",
            "warming_c_per_decade",
        ]
    ].rename(
        columns={
            "decade_start": "reference_year",
            "warming_c_per_decade": (
                "future_warming_c_per_decade"
            ),
        }
    )

    periods = periods.merge(
        future_climate,
        on=["cca3", "reference_year"],
        how="inner",
        validate="many_to_one"
    )

    # Population at the end of the outcome decade.

    future_population = population_panel[
        [
            "cca3",
            "population_year",
            "population",
            "density_per_km2",
        ]
    ].rename(
        columns={
            "population_year": "outcome_population_year",
            "population": "outcome_population",
            "density_per_km2": (
                "future_density_per_km2"
            ),
        }
    )

    periods["outcome_population_year"] = (
        periods["reference_year"] + 10
    )

    periods = periods.merge(
        future_population,
        on=["cca3", "outcome_population_year"],
        how="inner",
        validate="many_to_one"
    )

    if periods.empty:
        raise ValueError(
            "No matching country-period observations."
        )

    if periods.duplicated(
        ["cca3", "reference_year"]
    ).any():
        raise AssertionError(
            "Duplicate country-period observations."
        )

    return periods.sort_values(
        ["cca3", "reference_year"]
    ).reset_index(drop=True)


# -------------------------------------------------
# Target construction and temporal split
# -------------------------------------------------

CLASS_NAMES = ("Low", "Medium", "High")

TARGET_COLUMN = "target_exposure_class"

NUMERIC_PREDICTORS = [
    "population_at_reference",
    "density_at_reference_per_km2",
    "population_growth_prior_decade_pct",
    "prior_decade_mean_temp_c",
    "prior_decade_warming_c_per_decade",
    "prior_decade_detrended_volatility_c",
    "prior_decade_mean_temp_uncertainty_c",
    "area_km2",
]

CATEGORICAL_PREDICTORS = ["continent"]

PREDICTORS = (
    NUMERIC_PREDICTORS
    + CATEGORICAL_PREDICTORS
)


def percentile_against_training(
    values,
    sorted_reference
):
    """Calculate percentiles using a fixed training reference."""

    if len(sorted_reference) == 0:
        raise ValueError(
            "Empty training reference distribution."
        )

    return (
        np.searchsorted(
            sorted_reference,
            values,
            side="right"
        )
        / len(sorted_reference)
    )


def create_exposure_labels(periods):
    """Construct labels without fitting cutoffs to test data."""

    train = periods[
        periods["reference_year"] < TEST_REFERENCE_YEAR
    ].copy()

    test = periods[
        periods["reference_year"] == TEST_REFERENCE_YEAR
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Training or testing period is empty."
        )

    # Check the required outcome variables.

    outcome_columns = [
        "future_warming_c_per_decade",
        "future_density_per_km2",
        "outcome_population",
    ]

    for frame in (train, test):

        values = frame[
            outcome_columns
        ].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "Missing or nonfinite outcome data."
            )

        if (
            frame["future_density_per_km2"] <= 0
        ).any():
            raise ValueError(
                "Future density must be positive."
            )

    # Reference distributions from training data only.

    warming_reference = np.sort(
        train[
            "future_warming_c_per_decade"
        ].to_numpy(dtype=float)
    )

    density_reference = np.sort(
        np.log1p(
            train["future_density_per_km2"]
        ).to_numpy(dtype=float)
    )

    # Calculate the index for both periods using
    # the same frozen reference distributions.

    for frame in (train, test):

        frame["warming_percentile"] = (
            percentile_against_training(
                frame[
                    "future_warming_c_per_decade"
                ].to_numpy(dtype=float),
                warming_reference
            )
        )

        frame["density_percentile"] = (
            percentile_against_training(
                np.log1p(
                    frame["future_density_per_km2"]
                ).to_numpy(dtype=float),
                density_reference
            )
        )

        # Geometric mean: both components matter.

        frame["future_exposure_index"] = np.sqrt(
            frame["warming_percentile"]
            * frame["density_percentile"]
        )

    # Freeze class boundaries using training data.

    cutoffs = np.quantile(
        train["future_exposure_index"],
        [1 / 3, 2 / 3]
    )

    if np.isclose(cutoffs[0], cutoffs[1]):
        raise ValueError(
            "Exposure class boundaries overlap."
        )

    for frame in (train, test):

        frame[TARGET_COLUMN] = pd.Categorical(
            np.select(
                [
                    frame["future_exposure_index"]
                    < cutoffs[0],

                    frame["future_exposure_index"]
                    < cutoffs[1],
                ],
                ["Low", "Medium"],
                default="High"
            ),
            categories=CLASS_NAMES,
            ordered=True
        )

    label_metadata = {
        "target": (
            "Constructed subsequent warming-density "
            "exposure proxy"
        ),
        "train_years": [1970, 1980, 1990],
        "test_year": 2000,
        "low_medium_cutoff": float(cutoffs[0]),
        "medium_high_cutoff": float(cutoffs[1]),
        "predictors": PREDICTORS,
        "class_order": list(CLASS_NAMES),
    }

    return train, test, label_metadata


# -------------------------------------------------
# Reproducible pipeline and exports
# -------------------------------------------------

def run_pipeline(project_root):
    """Run the complete preprocessing pipeline."""

    project_root = Path(project_root).resolve()

    raw_dir = project_root / "data" / "raw"

    output_dir = (
        project_root / "data" / "processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    climate, population, audit = load_and_audit(
        raw_dir
    )

    matched, crosswalk = make_country_crosswalk(
        climate,
        population
    )

    # Check the opposite direction as well:
    # population countries/territories without a confirmed climate match.
    population_only = find_population_only_countries(
        matched,
        population
    )

    annual, incomplete = annual_climate_table(
        climate,
        matched
    )

    decades = compute_decadal_climate(
        annual
    )

    panel = historical_population_panel(
        population
    )

    periods = build_country_periods(
        decades,
        panel
    )

    train, test, metadata = create_exposure_labels(
        periods
    )

    # Explicit feature whitelist.
    # Future outcomes and audit-only scores are excluded.

    model_columns = [
        "country",
        "cca3",
        "reference_year",
        *PREDICTORS,
        TARGET_COLUMN,
    ]

    train_model = train[
        model_columns
    ].copy()

    test_model = test[
        model_columns
    ].copy()

    forbidden = {
        "future_warming_c_per_decade",
        "future_density_per_km2",
        "outcome_population",
        "future_exposure_index",
    }

    if forbidden.intersection(model_columns):
        raise AssertionError(
            "Future outcome leakage in predictors."
        )

    # Save intermediate datasets for reproducibility.

    output_files = {
        "country_matching_audit.csv": crosswalk,
        "population_only_countries.csv": population_only,
        "annual_climate.csv": annual,
        "incomplete_country_years.csv": incomplete,
        "decadal_climate_features.csv": decades,
        "historical_population_panel.csv": panel,
        "all_country_periods_AUDIT_ONLY.csv": (
            pd.concat([train, test])
        ),
        "ml_train.csv": train_model,
        "ml_test.csv": test_model,
    }

    for filename, dataframe in output_files.items():

        dataframe.to_csv(
            output_dir / filename,
            index=False
        )

    # Extend the data-quality report.

    audit.update({
        "matched_countries": int(
            matched["cca3"].nunique()
        ),
        "unmatched_climate_locations": int(
            crosswalk["population_country"].isna().sum()
        ),
        "unmatched_climate_names": (
            crosswalk.loc[
                crosswalk["population_country"].isna(),
                "climate_country"
            ].sort_values().tolist()
        ),
        "manual_alias_matches": int(
            crosswalk["matching_method"].eq("manual_alias").sum()
        ),
        "excluded_alternative_climate_labels": int(
            crosswalk["matching_method"].eq(
                "excluded_alternative"
            ).sum()
        ),
        "excluded_alternative_climate_names": (
            crosswalk.loc[
                crosswalk["matching_method"].eq(
                    "excluded_alternative"
                ),
                "climate_country"
            ].sort_values().tolist()
        ),
        "population_without_climate_match": int(
            len(population_only)
        ),
        "population_without_climate_match_names": (
            population_only["Country/Territory"].tolist()
        ),
        "complete_country_years": int(
            len(annual)
        ),
        "incomplete_country_years": int(
            len(incomplete)
        ),
        "complete_country_decades": int(
            len(decades)
        ),
        "country_periods": int(
            len(periods)
        ),
        "training_observations": int(
            len(train_model)
        ),
        "testing_observations": int(
            len(test_model)
        ),
        "training_class_distribution": (
            train_model[TARGET_COLUMN]
            .value_counts()
            .to_dict()
        ),
        "testing_class_distribution": (
            test_model[TARGET_COLUMN]
            .value_counts()
            .to_dict()
        ),
        "missing_training_growth": int(
            train_model[
                "population_growth_prior_decade_pct"
            ].isna().sum()
        ),
    })

    # Export methodological documentation.

    with open(
        output_dir / "data_quality_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            audit,
            file,
            indent=4,
            ensure_ascii=False
        )

    with open(
        output_dir / "label_definition.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("Preprocessing completed successfully.")
    print("Model-ready files saved to:", output_dir)

    return {
        "train": train_model,
        "test": test_model,
        "audit": audit,
        "metadata": metadata,
        "country_crosswalk": crosswalk,
        "population_only": population_only,
    }


if __name__ == "__main__":

    root = Path(__file__).resolve().parent.parent

    run_pipeline(root)