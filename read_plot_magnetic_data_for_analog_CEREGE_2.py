import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def calibrate(susc_samp, susc_vial, susc_stand, mass_samp, volume):
    susc_real = (susc_samp - susc_vial) * 1E-3 * (volume * 1E-9) / (mass_samp * 1E-3) * 92.38 / susc_stand
    return susc_real


if __name__ == "__main__":
    work_path = Path("C:/Users/tjiang/OneDrive - MNHN/MNHN/2025-04-14-magnetic/2026-05-04 analog")
    output_path = work_path

    mass_file = work_path / "mass_Freq.xlsx"
    mass = pd.read_excel(mass_file)

    susc_file = work_path / "2026-05-07 analog_renamed.xlsx"
    susc = pd.read_excel(susc_file)
    susc = susc.drop(index=susc.index[0])

    vial_file = work_path / "2026-05-04-05 analog_2_clean.xlsx"
    vial_data = pd.read_excel(vial_file)
    vial_data = vial_data.drop(index=vial_data.index[0])

    # prepare vial data - added .copy() to avoid warning
    vial_small_data = vial_data[vial_data["Name"].str.contains("empty vial small")].copy()
    vial_small_data["Vol. susc."] = vial_small_data["Vol. susc."] + vial_small_data["Pot.susc."]
    vial_small_data_average = vial_small_data["Vol. susc."].mean()

    # prepare standard data
    standard_data = susc[susc["Name"].str.contains("standard92.38", na=False)].copy()
    # add the pot susc back
    standard_data["Vol. susc."] = standard_data["Vol. susc."] + standard_data["Pot.susc."]

    # Store standard data as dictionary for faster lookup
    standard_dict = {
        "F_16k": standard_data[standard_data["Freq."].between(15500, 16500)]["Vol. susc."].mean(),
        "F_8k": standard_data[standard_data["Freq."].between(7500, 8500)]["Vol. susc."].mean(),
        "F_4k": standard_data[standard_data["Freq."].between(3500, 4500)]["Vol. susc."].mean(),
        "F_2k": standard_data[standard_data["Freq."].between(1500, 2500)]["Vol. susc."].mean(),
        "F_1k": standard_data[standard_data["Freq."].between(900, 1500)]["Vol. susc."].mean(),
        "F_500": standard_data[standard_data["Freq."].between(400, 800)]["Vol. susc."].mean(),
        "F_250": standard_data[standard_data["Freq."].between(200, 300)]["Vol. susc."].mean(),
        "F_125": standard_data[standard_data["Freq."].between(100, 180)]["Vol. susc."].mean(),
        "F_63": standard_data[standard_data["Freq."].between(50, 80)]["Vol. susc."].mean(),
    }

    # Prepare sample ID list
    volume = 10000
    sample_list = [
        "ZH-JT-180907 ILM <45 um Original",
        "ZH-JT-180908 ILM <45 um 50mJ*2",
        "ZH-JT-180910 ILM <45 um 50mJ*4",
        "ZH-JT-180909 ILM <45 um 50mJ*6",
        "ZH-JT-180803 OPX <45 um Original",
        "ZH-JT-180804 OPX <45 um 50mJ*8",
        "ZH-JT-180911 OLB <45 um Original",
        "ZH-JT-180914 OLB <45 um 50mJ*2",
        "ZH-JT-180913 OLB <45 um 50mJ*4",
        "ZH-JT-180912 OLB <45 um 50mJ*6",
    ]

    Freq_list = [
        "F_16k", "F_8k", "F_4k", "F_2k", "F_1k", "F_500", "F_250", "F_125", "F_63"
    ]

    # Initialize columns if they don't exist
    for freq in Freq_list:
        if freq not in mass.columns:
            mass[freq] = np.nan

    for index, row in mass.iterrows():
        sample_ID = row["ID"]
        set_ID = row["set"]
        label = row["label"]
        mass_value = row["mass of the powder for frequency (g)"]

        if sample_ID in sample_list:
            sample_data = susc[susc["Name"] == sample_ID].copy()
            sample_data["Vol. susc."] = sample_data["Vol. susc."] + sample_data["Pot.susc."]

            # Store sample data as dictionary for faster lookup
            sample_dict = {
                "F_16k": sample_data[sample_data["Freq."].between(15500, 16500)]["Vol. susc."].mean(),
                "F_8k": sample_data[sample_data["Freq."].between(7500, 8500)]["Vol. susc."].mean(),
                "F_4k": sample_data[sample_data["Freq."].between(3500, 4500)]["Vol. susc."].mean(),
                "F_2k": sample_data[sample_data["Freq."].between(1500, 2500)]["Vol. susc."].mean(),
                "F_1k": sample_data[sample_data["Freq."].between(900, 1500)]["Vol. susc."].mean(),
                "F_500": sample_data[sample_data["Freq."].between(400, 800)]["Vol. susc."].mean(),
                "F_250": sample_data[sample_data["Freq."].between(200, 300)]["Vol. susc."].mean(),
                "F_125": sample_data[sample_data["Freq."].between(100, 180)]["Vol. susc."].mean(),
                "F_63": sample_data[sample_data["Freq."].between(50, 80)]["Vol. susc."].mean(),
            }

            # Calculate real susceptibility for each frequency
            for freq in Freq_list:
                susc_sample = sample_dict.get(freq)
                susc_stand = standard_dict.get(freq)

                # Skip if any required value is NaN
                if pd.notna(susc_sample) and pd.notna(susc_stand) and pd.notna(vial_small_data_average):
                    susc_sample_real_Freq = calibrate(
                        susc_samp=susc_sample,
                        susc_vial=vial_small_data_average,
                        susc_stand=susc_stand,
                        mass_samp=mass_value,
                        volume=volume,
                    )
                    mass.at[index, freq] = susc_sample_real_Freq
                else:
                    mass.at[index, freq] = np.nan
                    print(f"Warning: Missing data for {sample_ID} at {freq}")

    mass.to_excel(output_path / "mass_Freq_updated.xlsx", index=False)
    print("Processing complete!")