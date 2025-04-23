import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path



if __name__=="__main__":
    work_path = Path("C:/Users/tjiang/OneDrive - MNHN/MNHN/2025-04-14-magnetic")
    table = pd.read_excel(work_path / "results_cleaned.xlsx")
    keywords = list(set(table["position"].tolist()))
    standards = [1, 2, 3]
    print(keywords)

    plt.figure(figsize=(16, 6))
    for i, standard in enumerate(standards):
        data_points = []
        plt.subplot(1, 3, i+1)
        for keyword in keywords:
            selected_row = table[(table["position"] == keyword) & (table["standard"] == standard)]
            size_index = selected_row["frenquency"].tolist()
            size_list = []
            for size in size_index:
                if size > 300:
                    size_list.append(35)
                else:
                    size_list.append(10)
            selected_row_value = selected_row["value"].tolist()
            for value in selected_row_value:
                data_points.append(value)
            plt.scatter(range(len(selected_row)), selected_row_value, s=size_list, alpha=0.7,
                        label=f"{keyword} - {standard}")
        data_points = np.array(data_points)
        data_points_mean = np.mean(data_points, axis=0)
        data_points_std = np.std(data_points, axis=0)
        plt.hlines(y=[data_points_mean-data_points_std,
                      data_points_mean,
                      data_points_mean+data_points_std],xmin=0, xmax=6,
                   color = ['g', 'r', 'g'], linestyles="--", alpha=0.5)
        ymin, ymax = plt.gca().get_ylim()
        plt.ylim(-10, 100)
        plt.title(f"standard {standard}: mean {data_points_mean:.2f}, std {data_points_std:.2f}")
        plt.legend()
    plt.tight_layout()
    plt.savefig(work_path / "magnetic_data_zoom.png", dpi=300)
    plt.close()