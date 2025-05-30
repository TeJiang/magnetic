import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    work_path = Path("C:/Users/tjiang/OneDrive - MNHN/MNHN/2025-04-14-magnetic/2025-05-28 2nd test with new container")
    table_list = list(work_path.glob("*.xls"))
    tables = []
    for table in table_list:
        tables.append(pd.read_excel(table))

    data = pd.concat(tables, ignore_index=True)
    data = data[~pd.isna(data["Name"])]
    freq_list = [
        16000,
        4000,
        1000,
        250,
        63,
    ]
    material_list = [
        "empty",
        "standard",
        "rock",
        # "fake vial"
    ]
    position_list = [
        "outside on table",
        "inside high",
    ]
    combinations = []
    for i, row in data.iterrows():
        Name = row["Name"]
        freq = row["Freq."]
        for material_candidate in material_list:
            if material_candidate in Name:
                material = material_candidate
        for position_candidate in position_list:
            if position_candidate in Name:
                position = position_candidate
        combination = (material, position, min(freq_list, key=lambda x: abs(x-freq)))
        if combination not in combinations:
            combinations.append(combination)

    colors = ["orange", "green"]
    # one material get one column
    plt.figure(figsize=(20, 8), dpi=300)
    for i in range(len(material_list)):
        # add one subplot
        plt.subplot(1, 4, i+1)
        plt.title(material_list[i])
        # one position get one color
        for j in range(len(position_list)):
            values_list =[]
            mean_list = []
            std_list = []
            y_labels = []  # Now y-axis will show labels
            # one frequency get one row
            for k in range(len(freq_list)):
                # creat empty list to store data
                values = []

                # start to search each row
                for l, row in data.iterrows():
                    Name = row["Name"]
                    freq = min(freq_list, key=lambda x: abs(x-row["Freq."]))
                    # check candidate
                    if material_list[i] in Name and position_list[j] in Name and freq_list[k] == freq:
                        if row["Comment"] != "outlier":
                            values.append(row["Vol. susc."])
                values_list.append(values)
                mean_list.append(np.mean(values))
                std_list.append(np.std(values))
                y_labels.append(f"{freq_list[k]}")

            # Create y-positions (since we swapped axes)
            y_positions = np.arange(len(y_labels))

            # ERRORBAR (now horizontal)
            plt.errorbar(
                x=mean_list,  # Values on x-axis
                y=y_positions,  # Categories on y-axis
                xerr=std_list,  # Error bars now horizontal
                fmt='o',
                capsize=5,
                # color=colors[j],
                # ecolor=colors[j],
                markersize=5,
                label=f"{position_list[j]}",
            )

            # Set y-ticks (categories)
            plt.yticks(
                y_positions,
                y_labels,
                fontsize=9,  # Adjust font size if needed
                va='center'  # Center-align labels vertically
            )

        plt.legend(loc='lower left')
        plt.xlabel("Vol. susc. [1E-3SI]")
        plt.ylabel("Freq. [Hz]")
        plt.grid(True, linestyle='--', alpha=0.7)  # Grid only for x-axis
        plt.tight_layout()

    plt.savefig(work_path / "plot_swapped_axes.png", dpi=300, bbox_inches='tight')
    plt.close()