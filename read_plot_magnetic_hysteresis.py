import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


# --------------------------------------------------------------------------
# 1. File-reading helpers (robust header detection + parameter extraction)
# --------------------------------------------------------------------------

def find_header_line(filepath):
    """0-based index of the line 'Step,...,Field,...' (the data table header)."""
    with open(filepath, "r", encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f):
            s = line.lstrip("\ufeff").strip()
            if s.startswith("Step,") and "Field" in s:
                return i
    return 0


def read_data(filepath):
    df = pd.read_csv(filepath,
                     skiprows=find_header_line(filepath),
                     engine="python",
                     on_bad_lines="skip",
                     encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def read_extracted(filepath):
    """Parse the '#EXTRACTED PARAMETERS' block into a dict."""
    out = {}
    with open(filepath, "r", encoding="utf-8-sig", errors="ignore") as f:
        in_block = False
        for line in f:
            s = line.strip()
            if s.startswith("#EXTRACTED PARAMETERS"):
                in_block = True
                continue
            if in_block and s.startswith("START TIME"):
                break
            if in_block and ":" in s:
                k, v = s.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def num(s):
    """First number found in the string."""
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s or "")
    return float(m.group()) if m else np.nan


# --------------------------------------------------------------------------
# 2. Filename parsing
# --------------------------------------------------------------------------

def parse_sample_name(stem):
    """Return (sample, treatment). Example:
       ZH-JT-180501_JSC_LRS_Original-103mg   -> ('JSC_LRS', 'Original')
       ZH-JT-180905_JSC_LRS_st45um_50mJ2-... -> ('JSC_LRS_st45um', '50mJ2')
       Anorthosite_334mg                     -> ('Anorthosite', 'Original')
    """
    base = re.sub(r'\s+Step\s+\d+\s+.*$', '', stem)         # strip " Step N ..."
    base = re.sub(r'^ZH-JT-\d+_', '', base)                 # strip run ID
    m = re.search(r'_(Original|50mJ\d+)(?:[-_]|$)', base)
    if m:
        return base[:m.start()], m.group(1)
    return re.sub(r'[-_]\d+mg$', '', base), 'Original'      # no treatment label


def parse_mass(stem):
    """Mass in mg from the filename, or NaN."""
    m = re.search(r'[-_](\d+)mg', stem)
    return float(m.group(1)) if m else np.nan


def classify(stem):
    if "Hysteresis" in stem:
        return "hysteresis"
    if "Remanence" in stem:
        return "remanence"
    return None


# --------------------------------------------------------------------------
# 3. Main pipeline
# --------------------------------------------------------------------------

if __name__ == "__main__":
    work_path    = Path("C:/Users/tjiang/OneDrive - MNHN/MNHN/CE5-magnetic/2026-05-08 analog")
    data_path    = work_path / "CSV data"
    output_path  = work_path / "output"
    indiv_dir    = output_path / "individual"
    compare_dir  = output_path / "comparison"
    for d in (output_path, indiv_dir, compare_dir):
        d.mkdir(parents=True, exist_ok=True)

    treat_order  = ["Original", "50mJ2", "50mJ4", "50mJ6", "50mJ8"]
    treat_colors = dict(zip(treat_order,
                            ["k", "tab:blue", "tab:green", "tab:orange", "tab:red"]))

    rows   = []
    groups = defaultdict(list)         # (sample, curve_type) -> [(treatment, stem, x, y)]

    # ---------------- loop over files ----------------
    for file in sorted(data_path.glob("*.csv")):
        stem  = file.stem
        ctype = classify(stem)
        if ctype is None:
            continue

        sample, treatment = parse_sample_name(stem)
        mass              = parse_mass(stem)
        data              = read_data(file)
        params            = read_extracted(file)

        x = pd.to_numeric(data.iloc[:, 3], errors="coerce")
        y = pd.to_numeric(data.iloc[:, 4], errors="coerce")

        # ---------- (a) individual plot ----------
        fig, ax = plt.subplots(figsize=(6, 5))
        if ctype == "hysteresis" and "Segment" in data.columns:
            for seg in sorted(data["Segment"].dropna().unique()):
                mask = data["Segment"] == seg
                ax.plot(x[mask], y[mask], lw=0.8, label=f"segment {int(seg)}")
            ax.legend(fontsize=8)
        else:                                  # remanence curve = single branch
            ax.plot(x, y, "o-", ms=3, lw=0.8)
        ax.set_xlabel(data.columns[3])
        ax.set_ylabel(data.columns[4])
        ax.set_title(f"{stem}\n[{ctype}, {mass:.0f} mg]", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(indiv_dir / f"{stem}.png", dpi=200)
        plt.close(fig)

        # ---------- (b) collect parameters ----------
        row = {"file": stem, "sample": sample, "treatment": treatment,
               "curve_type": ctype, "mass_mg": mass}
        for k, v in params.items():
            row[k]          = v
            row[f"{k}_num"] = num(v)
        rows.append(row)

        groups[(sample, ctype)].append((treatment, stem, x, y))

    # ---------------- parameter table ----------------
    params_df = pd.DataFrame(rows)
    params_df.to_csv(output_path / "parameters.csv", index=False)
    print(f"Parsed {len(params_df)} files | {params_df['sample'].nunique()} samples")

    # ---------------- (c) overlay comparison per sample ----------------
    for (sample, ctype), items in groups.items():
        if len(items) < 2:
            continue

        items.sort(key=lambda t: treat_order.index(t[0])
                   if t[0] in treat_order else 99)

        fig, ax = plt.subplots(figsize=(7, 6))
        for treatment, stem, x, y in items:
            color = treat_colors.get(treatment, "gray")
            label = treatment
            # mark duplicate runs (files ending in _b or _c)
            if stem.endswith("_b") or stem.endswith("_c"):
                label += f" ({stem[-1]})"
            if ctype == "hysteresis":
                ax.plot(x, y, lw=1.0, color=color, label=label)
            else:
                ax.plot(x, y, "o-", ms=3, lw=1.0, color=color, label=label)

        ax.set_xlabel("Field (µ₀H) [T]")
        ax.set_ylabel("Moment (m) [A·m²]")
        ax.set_title(f"{sample} — {ctype}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="treatment")
        fig.tight_layout()
        safe = re.sub(r'[^A-Za-z0-9_-]', '_', f"{sample}_{ctype}")
        fig.savefig(compare_dir / f"overlay_{safe}.png", dpi=200)
        plt.close(fig)

    # ---------------- (d) numeric parameter comparison ----------------
    # Pick the important ones we discussed
    interesting = [
        "Intrinsic coercivity_num",   # hysteresis  Hc   [T]
        "Remanence_num",              # hysteresis  Mr   [A·m²]
        "Saturation moment_num",      # hysteresis  Ms   [A·m²]
        "Hr [T]_num",                 # remanence   Hr   [T]
        "mrMax [A·m²]_num",           # remanence   Mr_sat [A·m²]
        "mr(100mT)/mrMax [%]_num",    # remanence   hardness
        "mr(200mT)/mrMax [%]_num",
    ]
    for pname in interesting:
        if pname not in params_df.columns:
            continue
        sub = params_df.dropna(subset=[pname])
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for sample, grp in sub.groupby("sample"):
            grp = grp.copy()
            grp["ord"] = grp["treatment"].map(
                lambda t: treat_order.index(t) if t in treat_order else 99)
            grp = grp.sort_values("ord")
            ax.plot(grp["treatment"], grp[pname], "o-",
                    lw=1.0, label=sample)
        ax.set_ylabel(pname.replace("_num", ""))
        ax.set_xlabel("treatment")
        ax.set_title(pname.replace("_num", ""))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe = re.sub(r'[^A-Za-z0-9_-]', '_', pname)
        fig.savefig(compare_dir / f"param_{safe}.png", dpi=200)
        plt.close(fig)

    print("Done.")