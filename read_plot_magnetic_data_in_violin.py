import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    work_path = Path("C:/Users/tjiang/OneDrive - MNHN/MNHN/2025-04-14-magnetic")
    table = pd.read_excel(work_path / "results_cleaned.xlsx")
    table = table.sort_values(
        by=["standard", "position", "frenquency"],
        ascending=True,
    )

    combinations = []
    for i, row in table.iterrows():
        combination = (row["position_glovebox"], row["position"], row["standard"], row["frenquency"])
        if combination not in combinations:
            combinations.append(combination)

    scatter_list = []
    std_list = []
    y_labels = []  # Now y-axis will show labels

    for combination in combinations:
        mask = (
                (table["position_glovebox"] == combination[0]) &
                (table["position"] == combination[1]) &
                (table["standard"] == combination[2]) &
                (table["frenquency"] == combination[3])
        )
        values = table[mask]["value"].values
        scatter_list.append(np.mean(values))
        std_list.append(np.std(values))
        y_labels.append(f"{combination[0]}, {combination[1]}, {combination[2]}, {combination[3]}")

    # Create y-positions (since we swapped axes)
    y_positions = np.arange(len(y_labels))

    plt.figure(figsize=(18, 10))  # Taller figure for vertical labels

    # ERRORBAR (now horizontal)
    plt.errorbar(
        x=scatter_list,  # Values on x-axis
        y=y_positions,  # Categories on y-axis
        xerr=std_list,  # Error bars now horizontal
        fmt='o',
        capsize=5,
        color='blue',
        ecolor='red',
        markersize=5,
        label='Value ± Std Dev'
    )

    # Set y-ticks (categories)
    plt.yticks(
        y_positions,
        y_labels,
        fontsize=9,  # Adjust font size if needed
        va='center'  # Center-align labels vertically
    )

    plt.ylabel('Measurement Conditions')  # Now on y-axis
    plt.xlabel('Values')  # Now on x-axis
    plt.title('Values with Horizontal Error Bars')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # Grid only for x-axis
    plt.tight_layout()

    plt.savefig(work_path / "plot_swapped_axes.png", dpi=300, bbox_inches='tight')
    plt.close()