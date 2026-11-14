
import pandas as pd
import numpy as np
import pvlib

# Load PVGIS TMY dataset
file_path = "data/trichy_tmy.csv"
df = pd.read_csv(file_path, skiprows=17)

# Convert timestamps
df["time(UTC)"] = pd.to_datetime(
    df["time(UTC)"],
    format="%Y%m%d:%H%M",
    errors="coerce"
)

df = df.dropna(subset=["time(UTC)"])
df = df.set_index("time(UTC)")
df.index = df.index.tz_localize("UTC")

# Location
latitude = 10.783
longitude = 78.683

# Solar position
solar_position = pvlib.solarposition.get_solarposition(
    df.index, latitude, longitude
)

ghi = df["G(h)"]
dni = df["Gb(n)"]
dhi = df["Gd(h)"]

# Annual optimization
angles = np.arange(0, 91, 1)
energy = []

for tilt in angles:
    irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=180,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=dni,
        ghi=ghi,
        dhi=dhi
    )

    poa = irradiance["poa_global"].clip(lower=0)
    daily_energy = poa.sum() / 1000 * 0.86 / 365
    energy.append(daily_energy)

energy = np.array(energy)

# Seasonal optimization
df["month"] = df.index.month

seasons = {
    "Summer": [3, 4, 5],
    "Monsoon": [6, 7, 8, 9],
    "Winter": [10, 11, 12, 1, 2]
}

seasonal_results = {}

for season, months in seasons.items():

    season_df = df[df["month"].isin(months)]

    season_position = solar_position.loc[season_df.index]

    season_energy = []

    for tilt in angles:

        irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=180,
            solar_zenith=season_position["apparent_zenith"],
            solar_azimuth=season_position["azimuth"],
            dni=season_df["Gb(n)"],
            ghi=season_df["G(h)"],
            dhi=season_df["Gd(h)"]
        )

        poa = irradiance["poa_global"].clip(lower=0)

        daily_energy = (
            poa.sum() / 1000 * 0.86
            / len(season_df.index.normalize().unique())
        )

        season_energy.append(daily_energy)

    season_energy = np.array(season_energy)

    best_index = np.argmax(season_energy)

    seasonal_results[season] = {
        "angle": int(angles[best_index]),
        "energy": float(season_energy[best_index]),
        "angles": angles,
        "energy_curve": season_energy
    }
