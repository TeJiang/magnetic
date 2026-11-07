import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def calibrate(susc_samp, susc_vial, susc_stand, mass_samp, volume):
    susc_real = (susc_samp - susc_vial) * 1E-3 * (volume *1E-9) / (mass_samp *1E-3) * 92.38 / susc_stand
    return susc_real

if __name__ == "__main__":
    work_path = Path("C:/Users/tjiang/OneDrive - MNHN/MNHN/2025-04-14-magnetic/2026-05-04 analog")
    output_path = work_path

    mass_file = work_path / "mass.xlsx"
    mass = pd.read_excel(mass_file)

    susc_file = work_path / "2026-05-04-05 analog_2_clean.xlsx"
    susc = pd.read_excel(susc_file)
    susc = susc.drop(index=susc.index[0])

    # prepare vial data
    vial_small_data = susc[susc["Name"]=="empty vial small"].copy()
    # add the pot susc back
    vial_small_data["Vol. susc."] = vial_small_data["Vol. susc."] + vial_small_data["Pot.susc."]

    vial_small_data_63 = vial_small_data[vial_small_data["Freq."].between(50, 200)]
    vial_small_data_63_average = vial_small_data_63["Vol. susc."].mean()

    vial_small_data_1k = vial_small_data[vial_small_data["Freq."].between(900, 1200)]
    vial_small_data_1k_average = vial_small_data_1k["Vol. susc."].mean()

    vial_small_data_16k = vial_small_data[vial_small_data["Freq."].between(15000, 17000)]
    vial_small_data_16k_average = vial_small_data_16k["Vol. susc."].mean()

    # prepare standard data
    standard_data = susc[susc["Name"].str.contains("standard92.38", na=False)].copy()
    # add the pot susc back
    standard_data["Vol. susc."] = standard_data["Vol. susc."] + standard_data["Pot.susc."]

    standard_data_63 = standard_data[standard_data["Freq."].between(50, 200)]
    standard_data_63_average = standard_data_63["Vol. susc."].mean()

    standard_data_1k = standard_data[standard_data["Freq."].between(900, 1200)]
    standard_data_1k_average = standard_data_1k["Vol. susc."].mean()

    standard_data_16k = standard_data[standard_data["Freq."].between(15000, 17000)]
    standard_data_16k_average = standard_data_16k["Vol. susc."].mean()

    # Prepare sample ID list
    sample_ID_list = mass["ID"]
    set_list = mass["set"]
    label_list = mass["label"]
    volume = 10000
    for index, row in mass.iterrows():
        sample_ID = row["ID"]
        set_ID = row["set"]
        label = row["label"]
        mass_value = row["mass of the powder (g)"]

        sample_data = susc[susc["Name"] == sample_ID]
        sample_data["Vol. susc."] = sample_data["Vol. susc."] + sample_data["Pot.susc."]

        sample_data_63 = sample_data[sample_data["Freq."].between(50, 200)]
        sample_data_63_average = sample_data_63["Vol. susc."].mean()
        if not np.isnan(sample_data_63_average):
            sample_susc_63 = calibrate(
                susc_samp=sample_data_63_average,
                susc_vial=vial_small_data_63_average,
                susc_stand=standard_data_63_average,
                mass_samp=mass_value,
                volume=volume,
            )
        else:
            sample_susc_63 = np.nan
        mass.at[index, "63_value"] = sample_susc_63

        sample_data_1k = sample_data[sample_data["Freq."].between(900, 1200)]
        sample_data_1k_average = sample_data_1k["Vol. susc."].mean()
        if not np.isnan(sample_data_1k_average):
            sample_susc_1k = calibrate(
                susc_samp=sample_data_1k_average,
                susc_vial=vial_small_data_1k_average,
                susc_stand=standard_data_1k_average,
                mass_samp=mass_value,
                volume=volume,
            )
        else:
            sample_susc_1k = np.nan
        mass.at[index, "1k_value"] = sample_susc_1k

        sample_data_16k = sample_data[sample_data["Freq."].between(15000, 17000)]
        sample_data_16k_average = sample_data_16k["Vol. susc."].mean()
        if not np.isnan(sample_data_16k_average):
            sample_susc_16k = calibrate(
                susc_samp=sample_data_16k_average,
                susc_vial=vial_small_data_16k_average,
                susc_stand=standard_data_16k_average,
                mass_samp=mass_value,
                volume=volume,
            )
        else:
            sample_susc_16k = np.nan
        mass.at[index, "16k_value"] = sample_susc_16k
    mass.to_excel(output_path / "mass_updated.xlsx", index=False)