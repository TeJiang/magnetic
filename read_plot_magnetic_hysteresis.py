import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


# --------------------------------------------------------------------------
# 1. File-reading helpers (robust header detection + parameter extraction)
# --------------------------------------------------------------------------

def read_lines(filepath):
    """Whole file as a list of lines (UTF-8 first, cp1252 as fallback)."""
    raw = Path(filepath).read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").splitlines()


def find_header_line(lines):
    """0-based index of the line 'Step,...,Field,...' (the data table header)."""
    for i, line in enumerate(lines):
        s = line.lstrip("\ufeff").strip()
        if s.startswith("Step,") and "Field" in s:
            return i
    raise ValueError("data table header not found")


def read_data(lines):
    """Data table as a DataFrame. Column 'bad' is True when a Status flag is not GOOD."""
    h = find_header_line(lines)
    table = [lines[h]]
    for line in lines[h + 1:]:  # keep data rows only (they start with the step number)
        if line.startswith("Step,") or line.startswith("##DATA TABLE"):
            break  # a second table starts here
        if re.match(r"\s*\d", line):  # blank / '#' lines inside the table are skipped
            table.append(line)
    df = pd.read_csv(io.StringIO("\n".join(table)), index_col=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    bad = pd.Series(False, index=df.index)
    for c in df.columns:
        if c.endswith("Status"):
            bad |= df[c].astype(str).str.strip().str.upper().ne("GOOD")
    df["bad"] = bad
    return df


def get_columns(data):
    """Names of the field and moment columns (fallback: columns 3 and 4)."""
    fcol = next((c for c in data.columns if c.startswith("Field") and "Status" not in c), None)
    mcol = next((c for c in data.columns if c.startswith("Moment") and "Status" not in c), None)
    if fcol is None or mcol is None:
        fcol, mcol = data.columns[3], data.columns[4]
    return fcol, mcol


def read_extracted(lines):
    """Parse the '#EXTRACTED PARAMETERS' block into a dict."""
    out = {}
    in_block = False
    for line in lines:
        s = line.strip()
        if s.startswith("#EXTRACTED PARAMETERS"):
            in_block = True
            continue
        if in_block and (s.startswith("START TIME") or s.startswith("#DATA")):
            break
        if in_block and ":" in s:
            k, v = s.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def num(s):
    """First number found in the string (converted to T if the unit is mT / uT)."""
    m = re.search(r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-zµ]*)', s or "")
    if not m:
        return np.nan
    factor = {"mT": 1e-3, "µT": 1e-6, "uT": 1e-6}.get(m.group(2), 1.0)
    return float(m.group(1)) * factor


# --------------------------------------------------------------------------
# 2. Filename parsing
# --------------------------------------------------------------------------

TREAT_RE = re.compile(r'_(Original|\d+mJ\d*)(?=[-_ ]|$)')
MASS_RE = re.compile(r'[-_](\d+(?:\.\d+)?)\s*mg', re.I)


def parse_sample_name(stem):
    """Return (sample, treatment). Example:
       ZH-JT-180501_JSC_LRS_Original-103mg Hysteresis   -> ('JSC_LRS', 'Original')
       ZH-JT-180905_JSC_LRS_st45um_50mJ2-98mg Hysteresis -> ('JSC_LRS_st45um', '50mJ2')
       Anorthosite_334mg Hysteresis                      -> ('Anorthosite', 'Original')
    Mass, the 'Hysteresis'/'Remanence' word and the replicate suffix (_b, _c) are removed.
    """
    base = re.sub(r'\s+Step\s+\d+\s+.*$', '', stem)  # strip " Step N ..."
    base = re.sub(r'^ZH-JT-\d+_', '', base)  # strip run ID
    m = TREAT_RE.search(base)
    sample = base[:m.start()] if m else base
    sample = MASS_RE.sub('', sample)
    sample = re.sub(r'[-_ ]*(?:Hysteresis|Remanence).*$', '', sample, flags=re.I)
    sample = re.sub(r'_[bc]$', '', sample).strip('-_ ')
    return sample, (m.group(1) if m else 'Original')


def parse_mass(stem):
    """Mass in mg from the filename, or NaN."""
    m = MASS_RE.search(stem)
    return float(m.group(1)) if m else np.nan


def parse_replicate(stem):
    """'a' for the first run, 'b' / 'c' for files ending in _b / _c."""
    m = re.search(r'_([bc])$', stem)
    return m.group(1) if m else 'a'


def classify(stem):
    low = stem.lower()
    if "hysteresis" in low:
        return "hysteresis"
    if "remanence" in low:
        return "remanence"
    return None


def treat_key(t):
    """Sort key: Original first, then laser energy, then number of shots (50mJ2 < 50mJ4 ...)."""
    if t == "Original":
        return (0, 0)
    m = re.match(r'(\d+)mJ(\d*)', t)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (10 ** 9, 0)


# --------------------------------------------------------------------------
# 3. Plot helper: mark Ms, Mr, Hci (values from the instrument file)
# --------------------------------------------------------------------------

def mark_parameters(ax, hc, mr, ms, scale, unit):
    """Draw Ms (dashed lines), Mr (squares at H=0) and Hci (triangles at M=0)."""
    if np.isfinite(ms):
        ax.axhline(ms * scale, color="tab:green", ls="--", lw=0.8,
                   label=f"Ms = {ms * scale:.3g} {unit}")
        ax.axhline(-ms * scale, color="tab:green", ls="--", lw=0.8)
    if np.isfinite(mr):
        ax.plot([0, 0], [mr * scale, -mr * scale], "s", color="tab:blue", ms=3,
                label=f"Mr = {mr * scale:.3g} {unit}")
    if np.isfinite(mr) and np.isfinite(ms):
        ax.plot([], [], " ", label=f"Mr/Ms = {mr / ms:.3f}")
    if np.isfinite(hc):
        ax.plot([hc, -hc], [0, 0], "^", color="tab:orange", ms=3,
                label=f"Hci = {hc * 1e3:.2f} mT")


# --------------------------------------------------------------------------
# 4. Main pipeline
# --------------------------------------------------------------------------

if __name__ == "__main__":
    work_path = Path("C:/Users/tjiang/OneDrive - MNHN/MNHN/CE5-magnetic/2026-05-08 analog")
    data_path = work_path / "CSV data"
    output_path = work_path / "output"
    indiv_dir = output_path / "individual"
    compare_dir = output_path / "comparison"
    for d in (output_path, indiv_dir, compare_dir):
        d.mkdir(parents=True, exist_ok=True)
    for d in (indiv_dir, compare_dir):  # separate subfolders per curve type
        for sub in ("hysteresis", "remanence"):
            (d / sub).mkdir(exist_ok=True)

    # ---------------- settings ----------------
    NORMALIZE = True  # divide moments by the mass in the file name -> A·m²/kg
    ZOOM_T = 0.05  # T, half-width of the low-field zoom in the overlays

    treat_order = ["Original", "50mJ2", "50mJ4", "50mJ6", "50mJ8"]
    treat_colors = dict(zip(treat_order,
                            ["k", "tab:blue", "tab:green", "tab:orange", "tab:red"]))

    rows = []
    groups = defaultdict(list)  # (sample, curve_type) -> [(treatment, replicate, x, y, bad)]

    # ---------------- load susceptibility file ----------------
    sus_file = output_path / "suceptibility.xlsx"

    sus_df = pd.DataFrame()
    if sus_file.exists():
        sus_raw = pd.read_excel(sus_file)
        # Clean & standardize sample names and treatments to match VSM parser
        set_map = {
            'CI ARS': 'CI_ARS',
            'JSC-1A LRS <45 um': 'JSC_LRS_st45um',
            'JSC-1A LRS': 'JSC_LRS',
            'ILM <45 um': 'ILM_st45um',
            'OPX <45 um': 'OPX_st45um',
            'OLB <45 um': 'OLB_st45um',
            'AN <45 um': 'Anorthosite'
        }
        sus_raw['sample'] = sus_raw['set'].map(set_map).fillna(sus_raw['set'])
        sus_raw['treatment'] = sus_raw['label'].str.replace('*', '', regex=False).str.replace('50mJ', '50mJ',
                                                                                              regex=False)
        # Normalize 50mJ2 etc naming
        sus_raw['treatment'] = sus_raw['treatment'].apply(lambda t: t if t == 'Original' else t.replace(' ', ''))
        sus_df = sus_raw

    # ---------------- loop over files ----------------
    for file in sorted(data_path.glob("*.csv")):
        stem = file.stem
        ctype = classify(stem)
        if ctype is None:
            continue

        sample, treatment = parse_sample_name(stem)
        replicate = parse_replicate(stem)
        mass = parse_mass(stem)  # mg, from the file name
        try:
            lines = read_lines(file)
            data = read_data(lines)
            params = read_extracted(lines)
            fcol, mcol = get_columns(data)
        except Exception as e:  # one bad file should not stop the loop
            print(f"SKIPPED {file.name}: {e}")
            continue

        x = pd.to_numeric(data[fcol], errors="coerce")
        y = pd.to_numeric(data[mcol], errors="coerce")
        ok = x.notna() & y.notna()
        data = data[ok].reset_index(drop=True)
        x, y, bad = x[ok].to_numpy(), y[ok].to_numpy(), data["bad"].to_numpy()
        print(f"{file.name}: {len(x)} points, {int(bad.sum())} with Status not GOOD")

        # mass normalization (the mass in the VSM header is 1 g, so we use the file name)
        mass_kg = mass * 1e-6
        if NORMALIZE and np.isfinite(mass_kg):
            scale, unit = 1.0 / mass_kg, "A·m²/kg"
        else:
            scale, unit = 1.0, "A·m²"
            if NORMALIZE:
                print(f"   WARNING {file.name}: no mass in the file name, NOT normalized")

        # parameters from the instrument file (T and A·m²)
        hc = num(params.get("Intrinsic coercivity"))
        mr = num(params.get("Remanence"))
        ms = num(params.get("Saturation moment"))

        # ---------- (a) individual plot ----------
        fig, ax = plt.subplots(figsize=(6, 5))
        if ctype == "hysteresis":  # whole loop (all segments) as one line
            ax.plot(x, y * scale, lw=0.8, color="k")
        else:  # remanence curve = single branch
            ax.plot(x, y * scale, "o-", ms=3, lw=0.8)
        if bad.any():  # points with a Status different from GOOD
            ax.plot(x[bad], y[bad] * scale, "o", color="red", ms=4, label="Status not GOOD")
        if ctype == "hysteresis":
            mark_parameters(ax, hc, mr, ms, scale, unit)
        ax.legend(fontsize=8)
        ax.set_xlabel(fcol)
        ax.set_ylabel(f"Moment (m) [{unit}]")
        ax.set_title(f"{stem}\n[{ctype}, {mass:.0f} mg]", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(indiv_dir / ctype / f"{stem}.png", dpi=200)
        plt.close(fig)

        # ---------- (b) collect parameters ----------
        row = {"file": stem, "sample": sample, "treatment": treatment,
               "replicate": replicate, "curve_type": ctype, "mass_mg": mass,
               "n_bad_points": int(bad.sum())}
        for k, v in params.items():
            row[k] = v
            row[f"{k}_num"] = num(v)
        row["Ms_Am2kg"] = ms / mass_kg
        row["Mr_Am2kg"] = mr / mass_kg
        row["Mr_Ms"] = mr / ms
        rows.append(row)

        groups[(sample, ctype)].append((treatment, replicate, x, y * scale, bad, ms * scale))

    # ---------------- parameter table ----------------
    params_df = pd.DataFrame(rows)
    params_df.to_excel(output_path / "parameters.xlsx", index=False)
    print(f"Parsed {len(params_df)} files | {params_df['sample'].nunique()} samples")

    # Consistent color assignment across all samples based on sorted global sample names
    all_unique_samples = sorted(params_df["sample"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    sample_color_map = {s: cmap(i % cmap.N) for i, s in enumerate(all_unique_samples)}

    # ---------------- (c) overlay comparison per sample ----------------
    ylabel = "Moment (m) [A·m²/kg]" if NORMALIZE else "Moment (m) [A·m²]"

    for (sample, ctype), items in groups.items():
        if len(items) < 2:
            continue

        items.sort(key=lambda t: (treat_key(t[0]), t[1]))

        if ctype == "hysteresis":
            fig, axs = plt.subplots(2, 2, figsize=(11, 9))  # 4 subfigures: a, b, c, d
            axs = axs.flatten()
        else:
            fig, ax0 = plt.subplots(figsize=(7.5, 6))
            axs = [ax0]

        for treatment, replicate, x, y, bad, sample_ms in items:
            color = treat_colors.get(treatment, "gray")

            # Base treatment label
            label = treatment
            if replicate != "a":  # duplicate runs (files ending in _b or _c)
                label += f" ({replicate})"

            # If remanence curve, attach Hr information to the legend label
            if ctype == "remanence":
                sub_p = params_df[
                    (params_df["sample"] == sample) &
                    (params_df["treatment"] == treatment) &
                    (params_df["replicate"] == replicate) &
                    (params_df["curve_type"] == "remanence")
                    ]
                if not sub_p.empty and "Hr [T]_num" in sub_p.columns:
                    hr_val = sub_p["Hr [T]_num"].values[0]
                    if np.isfinite(hr_val):
                        label += f" (Hr = {abs(hr_val) * 1e3:.1f} mT)"

            ls = "--" if replicate != "a" else "-"
            norm_factor = sample_ms if (np.isfinite(sample_ms) and sample_ms != 0) else np.max(np.abs(y))
            y_norm = (y / norm_factor) if norm_factor else y

            for i, ax in enumerate(axs):
                if ctype == "hysteresis":
                    curr_y = y_norm if i >= 2 else y
                    ax.plot(x, curr_y, lw=1.0, color=color, ls=ls, label=label)
                    ax.plot(x[bad], curr_y[bad], "o", color="red", ms=3)  # Status not GOOD

                    if i in (1, 3):  # Zoom subplots
                        ax.set_xlim(-ZOOM_T, ZOOM_T)
                    if i >= 2:  # Force strict bounds for normalized plots (c and d)
                        ax.set_ylim(-1.05, 1.05)
                else:
                    ax.plot(x, y, "o-", ms=3, lw=1.0, color=color, ls=ls, label=label)
                    ax.plot(x[bad], y[bad], "o", color="red", ms=3)

                label = "_nolegend_"  # only one legend entry per curve

        if ctype == "hysteresis":
            axs[0].set_title("a) Original Hysteresis", fontsize=9)
            axs[1].set_title(f"b) Zoom (±{ZOOM_T * 1e3:.0f} mT)", fontsize=9)
            axs[2].set_title("c) Normalized ($M / M_s$)", fontsize=9)
            axs[3].set_title(f"d) Normalized Zoom (±{ZOOM_T * 1e3:.0f} mT)", fontsize=9)

            axs[0].set_ylabel(ylabel)
            axs[1].set_ylabel(ylabel)
            axs[2].set_ylabel("Normalized Moment ($M / M_s$)")
            axs[3].set_ylabel("Normalized Moment ($M / M_s$)")

            # Parameter text box on subplot (c)
            sample_params = params_df[params_df["sample"] == sample]
            if not sample_params.empty and "Mr_Ms" in sample_params.columns:
                box_text = f"Sample: {sample}\n" + "─" * 18
                for _, row_p in sample_params.iterrows():
                    t_name = row_p.get("treatment", "")
                    mrms = row_p.get("Mr_Ms", np.nan)
                    hci = row_p.get("Intrinsic coercivity_num", np.nan)
                    if np.isfinite(mrms):
                        box_text += f"\n{t_name}: Mr/Ms={mrms:.2f}"
                        if np.isfinite(hci):
                            box_text += f", Bc={hci * 1e3:.1f}mT"

                axs[2].text(0.03, 0.97, box_text, transform=axs[2].transAxes, fontsize=7.5,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))
        else:
            axs[0].set_ylabel("Moment (m) [A·m²]")

        for ax in axs:
            ax.set_xlabel("Field (µ₀H) [T]")
            ax.grid(alpha=0.3)

        axs[0].legend(fontsize=8, title="treatment", loc="upper left", framealpha=0.7)
        fig.suptitle(f"{sample} — {ctype}", fontsize=11, fontweight="bold")
        fig.tight_layout()
        safe = re.sub(r'[^A-Za-z0-9_-]', '_', f"{sample}_{ctype}")
        fig.savefig(compare_dir / ctype / f"overlay_{safe}.png", dpi=200)
        plt.close(fig)

    # ---------------- (d) numeric parameter comparison ----------------
    interesting = [
        ("Intrinsic coercivity_num", "Hci [T]"),
        ("Mr_Ms", "Mr/Ms"),
        ("Ms_Am2kg", "Ms [A·m²/kg]"),
        ("Mr_Am2kg", "Mr [A·m²/kg]"),
        ("Hr [T]_num", "Hr [T]  (remanence)"),
    ]
    selected_samples = [
        "CI_ARS",
        "JSC_LRS",
        "JSC_LRS_st45um",
        "OLB_st45um",
        "OPX_st45um",
        "Anorthosite",
        "ILM_st45um",
    ]
    all_treat = sorted(params_df["treatment"].unique(), key=treat_key)

    for pname, ylab in interesting:
        if pname not in params_df.columns:
            continue
        sub = params_df.dropna(subset=[pname])
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 8))
        for sample, grp in sub.groupby("sample"):
            if sample in selected_samples:
                s = grp.groupby("treatment")[pname].agg(["mean", "std"]).reset_index()
                s["pos"] = s["treatment"].map(all_treat.index)
                s = s.sort_values("pos")
                ax.errorbar(s["pos"], s["mean"], yerr=s["std"].fillna(0), fmt="o-",
                            lw=1.0, capsize=3, label=sample, color=sample_color_map.get(sample))
        ax.set_xticks(range(len(all_treat)))
        ax.set_xticklabels(all_treat)
        ax.set_ylabel(ylab)
        ax.set_xlabel("treatment")
        ax.set_title(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe = re.sub(r'[^A-Za-z0-9_-]', '_', pname)
        fig.savefig(compare_dir / f"param_{safe}_noCI.png", dpi=200)
        plt.close(fig)

    # ---------------- (e) Final plot: Mr/Ms vs Bcr/Bc for all samples & treatments (Day-Plot) ----------------
    if all(col in params_df.columns for col in ["Mr_Ms", "Intrinsic coercivity_num", "Hr [T]_num"]):
        sub_all = params_df.groupby(["sample", "treatment"])[
            ["Mr_Ms", "Intrinsic coercivity_num", "Hr [T]_num"]].mean().reset_index()
        sub_all = sub_all.dropna(subset=["Mr_Ms", "Intrinsic coercivity_num", "Hr [T]_num"])

        sub_all["treat_sort"] = sub_all["treatment"].map(treat_key)
        sub_all = sub_all.sort_values(["sample", "treat_sort"])

        treat_label_map = {"Original": "0", "50mJ2": "2", "50mJ4": "4", "50mJ6": "6", "50mJ8": "8"}

        if not sub_all.empty:
            fig, ax = plt.subplots(figsize=(7, 7))

            for sample, grp in sub_all.groupby("sample"):
                bcr_bc = np.abs(grp["Hr [T]_num"] / grp["Intrinsic coercivity_num"])
                mr_ms = grp["Mr_Ms"]
                color = sample_color_map.get(sample)

                ax.plot(bcr_bc, mr_ms, "o-", color=color, label=sample, alpha=0.8, lw=1.2, ms=5)

                for _, row in grp.iterrows():
                    x_val = np.abs(row["Hr [T]_num"] / row["Intrinsic coercivity_num"])
                    y_val = row["Mr_Ms"]
                    t_name = str(row["treatment"])
                    t_label = treat_label_map.get(t_name, t_name)
                    ax.text(x_val * 1.05, y_val * 1.05, t_label, fontsize=7, color=color, alpha=0.9, fontweight="bold")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.axvline(2, color="gray", ls="--", lw=0.8)
            ax.axvline(5, color="gray", ls="--", lw=0.8)
            ax.axhline(0.02, color="gray", ls="--", lw=0.8)
            ax.axhline(0.5, color="gray", ls="--", lw=0.8)

            ax.text(1.2, 0.6, "SD", fontsize=11, fontweight="bold", color="dimgray")
            ax.text(3.0, 0.3, "PSD", fontsize=11, fontweight="bold", color="dimgray")
            ax.text(2.2, 0.05, "SD-MD", fontsize=10, fontweight="bold", color="dimgray")
            ax.text(10, 0.008, "MD", fontsize=11, fontweight="bold", color="dimgray")

            ax.set_xlabel(r"$B_{cr} / B_c$ (Coercivity Ratio)")
            ax.set_ylabel(r"$M_r / M_s$ (Remanence Ratio)")
            ax.set_title("Day-Plot ($M_r/M_s$ vs $B_{cr}/B_c$) across all treatments")
            ax.set_xlim(1, 100)
            ax.set_ylim(0.005, 1)

            ax.grid(True, which="both", ls=":", alpha=0.4)
            ax.legend(fontsize=8, bbox_to_anchor=(0.7, 0.8), loc="upper left")
            fig.tight_layout()
            fig.savefig(compare_dir / "Mr_Ms_vs_Bcr_Bc_all_samples.png", dpi=200)
            plt.close(fig)

    # ---------------- (f) Susceptibility vs Treatment (3 subfigures for 63, 1k, 16k) ----------------
    if not sus_df.empty:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        freqs = [("63_value", "63 Hz"), ("1k_value", "1 kHz"), ("16k_value", "16 kHz")]

        selected_sus_samples = [
            # "CI_ARS",
            "JSC_LRS",
            "JSC_LRS_st45um",
            "OLB_st45um",
            "OPX_st45um",
            "Anorthosite",
            "ILM_st45um",
        ]

        for idx, (col, f_label) in enumerate(freqs):
            ax = axs[idx]
            if col in sus_df.columns:
                sub_sus = sus_df.dropna(subset=[col])
                for sample, grp in sub_sus.groupby("sample"):
                    if sample in selected_sus_samples:
                        s = grp.sort_values("treatment", key=lambda x: x.map(treat_key))
                        s["pos"] = s["treatment"].map(treat_key).map(
                            lambda k: all_treat.index(k) if k in all_treat else 0)  # align with treat order
                        # Alternatively map directly by treat_order
                        s_mapped = []
                        for _, r in s.iterrows():
                            t = r["treatment"]
                            if t in all_treat:
                                s_mapped.append({"pos": all_treat.index(t), "val": r[col], "treatment": t})
                        if s_mapped:
                            df_s = pd.DataFrame(s_mapped).sort_values("pos")
                            ax.errorbar(df_s["pos"], df_s["val"], fmt="o-", lw=1.0, capsize=3, label=sample,
                                        color=sample_color_map.get(sample))

                ax.set_xticks(range(len(all_treat)))
                ax.set_xticklabels(all_treat)
                ax.set_ylabel(f"Susceptibility ({f_label})")
                ax.set_xlabel("treatment")
                ax.set_title(f"Frequency: {f_label}")
                ax.grid(alpha=0.3)
                if idx == 0:
                    ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(compare_dir / "param_susceptibility_mixture_noCI.png", dpi=200)
        plt.close(fig)

    # ---------------- (g) Ms vs Susceptibility Plot (Selected samples with text labels) ----------------
    if not sus_df.empty and "Ms_Am2kg" in params_df.columns:
        # Merge susceptibility with params_df on sample and treatment
        merged_sus = pd.merge(params_df, sus_df, on=["sample", "treatment"], how="inner")

        selected_sus_samples = [
            "CI_ARS",
            "JSC_LRS",
            "JSC_LRS_st45um",
            "OLB_st45um",
            "OPX_st45um",
            "Anorthosite",
            "ILM_st45um",
        ]

        treat_label_map = {"Original": "0", "50mJ2": "2", "50mJ4": "4", "50mJ6": "6", "50mJ8": "8"}

        for sus_col, sus_name in [("63_value", "63 Hz"), ("16k_value", "16 kHz"), ("1k_value", "1k Hz")]:
            if sus_col not in merged_sus.columns:
                continue
            sub_m = merged_sus.dropna(subset=["Ms_Am2kg", sus_col])
            sub_m["treat_sort"] = sub_m["treatment"].map(treat_key)
            sub_m = sub_m.sort_values(["sample", "treat_sort"])

            if not sub_m.empty:
                fig, ax = plt.subplots(figsize=(6, 6))

                for sample, grp in sub_m.groupby("sample"):
                    if sample in selected_sus_samples:
                        x_sus = grp[sus_col]
                        y_ms = grp["Ms_Am2kg"]
                        color = sample_color_map.get(sample)

                        ax.plot(x_sus, y_ms, "o-", color=color, label=sample, alpha=0.8, lw=1.2, ms=5)

                        for _, row in grp.iterrows():
                            x_val = row[sus_col]
                            y_val = row["Ms_Am2kg"]
                            t_name = str(row["treatment"])
                            t_label = treat_label_map.get(t_name, t_name)
                            ax.text(x_val * 1.05, y_val * 1.05, t_label, fontsize=7, color=color, alpha=0.9,
                                    fontweight="bold")

                ax.set_xlabel(f"Susceptibility ({sus_name})")
                ax.set_ylabel(r"Saturation Magnetization $M_s$ [A·m²/kg]")
                ax.set_title(f"$M_s$ vs Susceptibility ({sus_name}) across treatments")
                ax.grid(True, which="both", ls=":", alpha=0.4)
                ax.legend(fontsize=8, bbox_to_anchor=(0.90, 0.05), loc="lower right")
                fig.tight_layout()
                safe_name = sus_col.split("_")[0]
                fig.savefig(compare_dir / f"Ms_vs_susceptibility_{safe_name}.png", dpi=200)
                plt.close(fig)

    print("Done.")