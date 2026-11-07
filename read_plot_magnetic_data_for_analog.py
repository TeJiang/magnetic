import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from matplotlib.lines import lineStyles

if __name__=="__main__":
    work_path = Path("C:/Users/tjiang/OneDrive - MNHN/MNHN/2025-04-14-magnetic/2026-02-25 SW sample test")
    data = pd.read_excel(work_path / "2026-02-25 SW sample test PM_corrected.xlsx")
    subfigure_list = [
        "JSC_1A_lt45um",
        "OLB_lt45um",
        "ILM_lt45um",
        "AN_lt45um",
    ]
    plt.figure(figsize = (12, 5))
    for i, subfigure in enumerate(subfigure_list):
        plt.subplot(1, len(subfigure_list),i+1)
        plt.title(subfigure)
        original_data_16k, irradiated_data_16k = [], []
        original_data_250, irradiated_data_250 = [], []
        x = [1, 2, 3, 4]
        for index, row in data.iterrows():
            Name = str(row["Name"])
            Vol_susc = row["Vol. susc."]
            Freq = row["Freq."]
            if subfigure in Name:
                print(Name, Vol_susc, Freq)
                if "original" in Name:
                    if abs(16000 - Freq) < 200:
                        original_data_16k.append(Vol_susc)
                    else:
                        original_data_250.append(Vol_susc)
                else:
                    if abs(16000 - Freq) < 200:
                        irradiated_data_16k.append(Vol_susc)
                    else:
                        irradiated_data_250.append(Vol_susc)
        plt.plot(x, original_data_16k, "-o", label="original 16kHz", color="blue")
        plt.plot(x, original_data_250, "-o", label="original 250Hz", color="orange")
        plt.plot(x, irradiated_data_16k, "-D", label="50mJ*6 16kHz", color="blue")
        plt.plot(x, irradiated_data_250, "-D", label="50mJ*6 250Hz", color="orange")
        # plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(work_path / "2026-02-25_test.png", dpi=300)
    plt.close()
