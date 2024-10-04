import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys
from scipy.stats import pearsonr, kendalltau
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['font.size'] = 10

def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

def weighted_mean(num_per_set, value_per_set):
    return np.sum(num_per_set*value_per_set) / np.sum(num_per_set)

def get_bootstrap_weighted_value(num_set, value_set, nboots=10000):
    value_samples = np.zeros(nboots)
    for b in range(nboots):
        inds = np.random.choice(len(value_set), len(value_set))
        value_samples[b] = weighted_mean(num_set[inds], value_set[inds])

    return np.percentile(value_samples, 2.5), np.percentile(value_samples, 97.5)


def get_weighted_stats(results, preds_col, truth_col, groups):
    results["pred"] = results[preds_col]
    results["truth"] = results[truth_col]

    logs = []
    for target in groups:
        subset = results[results["group_id"] == target]
        logs.append(
            [
                target,
                pearsonr(subset["pred"], subset["truth"])[0],
                kendalltau(subset["pred"], subset["truth"])[0],
                rmse(subset["pred"], subset["truth"]),
                len(subset),
            ]
        )

    performance = pd.DataFrame(
        logs, columns=["Target", "Pearson's R", "Kendall's Tau", "RMSE", "Sample size"]
    )

    wpcc = weighted_mean(np.array(performance["Sample size"]), np.array(performance["Pearson's R"]))
    wpcc_low, wpcc_high = get_bootstrap_weighted_value(np.array(performance["Sample size"]), np.array(performance["Pearson's R"]))
    wktau = weighted_mean(np.array(performance["Sample size"]), np.array(performance["Kendall's Tau"]))
    wktau_low, wktau_high = get_bootstrap_weighted_value(np.array(performance["Sample size"]), np.array(performance["Kendall's Tau"]))

    return wpcc, wpcc_low, wpcc_high, wktau, wktau_low, wktau_high


def get_performance_df(results, preds_col, truth_col, groups):
    results["pred"] = results[preds_col]
    results["truth"] = results[truth_col]

    logs = []
    for target in groups:
        subset = results[results["group_id"] == target]
        logs.append(
            [
                target,
                pearsonr(subset["pred"], subset["truth"])[0],
                kendalltau(subset["pred"], subset["truth"])[0],
                rmse(subset["pred"], subset["truth"]),
                len(subset),
            ]
        )

    performance = pd.DataFrame(
        logs, columns=["Target", "Pearson's R", "Kendall's Tau", "RMSE", "Sample size"]
    )
    return performance


def plot_lineplot_with_error_bars(
    fep_pcc_data,
    fep_ktau_data,
    df,
    df_baseline,
    label_fep,
    label,
    label_baseline,
    pcc_lims,
    ktau_lims,
    pcc_ticks,
    ktau_ticks,
    xlabel,
    invert_xaxis=False,
    outpath=None,
):
    color = "steelblue"
    color_baseline = "green"
    color_fep = "firebrick"

    fig = plt.figure(figsize=(3.8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    # plot PCC data
    ax1.axhline(y=fep_pcc_data[0], color=color_fep, linestyle="--", label=label_fep, lw=2)
    ax1.axhspan(ymin=fep_pcc_data[1], ymax=fep_pcc_data[2], color=color_fep, alpha=0.4)

    sns.lineplot(
        data = df,
        x="Tanimoto",
        y="WPCC",
        label=label,
        color=color,
        marker="o",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax1
    )
    errors = np.array([df["WPCC"] - df["WPCC_low"], df["WPCC_high"] - df["WPCC"]])
    ax1.errorbar(df["Tanimoto"], df["WPCC"], yerr=errors, fmt='o', capsize=4, elinewidth=1.5, ecolor=color, capthick=1.5, zorder=0)

    sns.lineplot(
        data = df_baseline,
        x="Tanimoto",
        y="WPCC",
        label=label_baseline,
        color=color_baseline,
        marker="o",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax1
    )
    errors = np.array([df_baseline["WPCC"] - df_baseline["WPCC_low"], df_baseline["WPCC_high"] - df_baseline["WPCC"]])
    ax1.errorbar(df_baseline["Tanimoto"], df_baseline["WPCC"], yerr=errors, fmt='o', capsize=4, elinewidth=1.5, ecolor=color_baseline, capthick=1.5, zorder=0)


    # plot Ktau data
    ax2.axhline(y=fep_ktau_data[0], color=color_fep, linestyle="--", lw=2)
    ax2.axhspan(ymin=fep_ktau_data[1], ymax=fep_ktau_data[2], color=color_fep, alpha=0.4)

    sns.lineplot(
        data = df,
        x="Tanimoto",
        y="WKtau",
        color=color,
        marker="o",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax2,
        legend=False
    )
    errors = np.array([df["WKtau"] - df["WKtau_low"], df["WKtau_high"] - df["WKtau"]])
    ax2.errorbar(df["Tanimoto"], df["WKtau"], yerr=errors, fmt='o', capsize=4, elinewidth=1.5, ecolor=color, capthick=1.5, zorder=0)

    sns.lineplot(
        data = df_baseline,
        x="Tanimoto",
        y="WKtau",
        color=color_baseline,
        marker="o",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax2,
        legend=False
    )

    errors = np.array([df_baseline["WKtau"] - df_baseline["WKtau_low"], df_baseline["WKtau_high"] - df_baseline["WKtau"]])
    ax2.errorbar(df_baseline["Tanimoto"], df_baseline["WKtau"], yerr=errors, fmt='o', capsize=4, elinewidth=1.5, ecolor=color_baseline, capthick=1.5, zorder=0)


    ax1.set_ylabel("Weighted mean PCC")
    ax1.set_xlabel("")
    ax1.set_ylim(pcc_lims)
    if invert_xaxis:
        ax1.invert_xaxis()
    ax1.set_yticks(pcc_ticks)
    ax1.set_xticks([0.9, 0.8, 0.7, 0.6])
    ax1.set_title("a)", loc="left", fontsize=14, position=(-0.25, 1.0))

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(r"Weighted mean K$\tau$")
    ax2.set_ylim(ktau_lims)
    ax2.set_yticks(ktau_ticks)
    ax2.set_xticks([0.9, 0.8, 0.7, 0.6])
    ax2.set_title("b)", loc="left", fontsize=14, position=(-0.25, 1.0))

    ax1.legend(bbox_to_anchor=(0.5, 1.25), loc="center", frameon=False)

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath)


"""
Print ligsim90 results
"""
R = 1.987e-3
T = 297

fep_index = pd.read_csv('new_data/fep/fep_benchmark_fep+_predictions.csv', index_col=0)
counts = pd.DataFrame(fep_index["group_id"].value_counts())
groups = list(counts[counts["count"] >= 10].index)

performance_fep = get_performance_df(fep_index, 'Pred. dG (kcal/mol)', 'Exp. dG (kcal/mol)', groups)

fep_index = fep_index[["graph_id", "Exp. dG (kcal/mol)"]]
fep_index = fep_index.rename(columns={"graph_id":"unique_id"})
results = pd.read_csv('new_data/bindingnet_bindingdb/AEV-PLIG_fep_benchmark_pdbbind_U_bindingnet_U_bindingdb_ligsim90_predictions.csv')
results["preds"] = -R*T*np.log(10)*results['preds']
results = results.merge(fep_index, how="left", on="unique_id")
performance_aevplig = get_performance_df(results, 'preds', 'Exp. dG (kcal/mol)', groups)

print("FEP+")
print("WPCC:", weighted_mean(np.array(performance_fep["Sample size"]), np.array(performance_fep["Pearson's R"])))
print("CI:", get_bootstrap_weighted_value(np.array(performance_fep["Sample size"]), np.array(performance_fep["Pearson's R"])))
print("WKtau:", weighted_mean(np.array(performance_fep["Sample size"]), np.array(performance_fep["Kendall's Tau"])))
print("CI:", get_bootstrap_weighted_value(np.array(performance_fep["Sample size"]), np.array(performance_fep["Kendall's Tau"])))
print("\n")
print("AEV-PLIG")
print("WPCC:", weighted_mean(np.array(performance_aevplig["Sample size"]), np.array(performance_aevplig["Pearson's R"])))
print("CI:", get_bootstrap_weighted_value(np.array(performance_aevplig["Sample size"]), np.array(performance_aevplig["Pearson's R"])))
print("WKtau:", weighted_mean(np.array(performance_aevplig["Sample size"]), np.array(performance_aevplig["Kendall's Tau"])))
print("CI:", get_bootstrap_weighted_value(np.array(performance_aevplig["Sample size"]), np.array(performance_aevplig["Kendall's Tau"])))

# baseline AEV-PLIG
fep_index = pd.read_csv('new_data/fep/fep_benchmark_fep+_predictions.csv', index_col=0)
fep_index = fep_index[["graph_id", "Exp. dG (kcal/mol)"]]
fep_index = fep_index.rename(columns={"graph_id":"unique_id"})
results = pd.read_csv('new_data/baseline/AEV-PLIG_fep_benchmark_pdbbind_ligsim90_predictions.csv')
results["preds"] = -R*T*np.log(10)*results['preds']
results = results.merge(fep_index, how="left", on="unique_id")
performance_aevplig = get_performance_df(results, 'preds', 'Exp. dG (kcal/mol)', groups)
print("\n")
print("AEV-PLIG baseline")
print("WPCC:", weighted_mean(np.array(performance_aevplig["Sample size"]), np.array(performance_aevplig["Pearson's R"])))
print("CI:", get_bootstrap_weighted_value(np.array(performance_aevplig["Sample size"]), np.array(performance_aevplig["Pearson's R"])))
print("WKtau:", weighted_mean(np.array(performance_aevplig["Sample size"]), np.array(performance_aevplig["Kendall's Tau"])))
print("CI:", get_bootstrap_weighted_value(np.array(performance_aevplig["Sample size"]), np.array(performance_aevplig["Kendall's Tau"])))



"""
Load all predictions
"""
fep_index = pd.read_csv('new_data/fep/fep_benchmark_fep+_predictions.csv', index_col=0)
counts = pd.DataFrame(fep_index["group_id"].value_counts())
groups = list(counts[counts["count"] >= 10].index)

fep_wpcc, fep_wpcc_low, fep_wpcc_high, fep_wktau, fep_wktau_low, fep_wktau_high = get_weighted_stats(fep_index, 'Pred. dG (kcal/mol)', 'Exp. dG (kcal/mol)', groups)
fep_index = fep_index[["graph_id", "Exp. dG (kcal/mol)"]]
fep_index = fep_index.rename(columns={"graph_id":"unique_id"})

# baseline
results = pd.read_csv('new_data/baseline/AEV-PLIG_fep_benchmark_pdbbind_ligsim90_predictions.csv')
results["preds"] = -R*T*np.log(10)*results['preds']
results = results.merge(fep_index, how="left", on="unique_id")
baseline_ls90_wpcc, baseline_ls90_wpcc_low, baseline_ls90_wpcc_high, baseline_ls90_wktau, baseline_ls90_wktau_low, baseline_ls90_wktau_high = get_weighted_stats(results, 'preds', 'Exp. dG (kcal/mol)', groups)

results = pd.read_csv('new_data/baseline/AEV-PLIG_fep_benchmark_pdbbind_ligsim80_predictions.csv')
results["preds"] = -R*T*np.log(10)*results['preds']
results = results.merge(fep_index, how="left", on="unique_id")
baseline_ls80_wpcc, baseline_ls80_wpcc_low, baseline_ls80_wpcc_high, baseline_ls80_wktau, baseline_ls80_wktau_low, baseline_ls80_wktau_high = get_weighted_stats(results, 'preds', 'Exp. dG (kcal/mol)', groups)

results = pd.read_csv('new_data/baseline/AEV-PLIG_fep_benchmark_pdbbind_ligsim70_predictions.csv')
results["preds"] = -R*T*np.log(10)*results['preds']
results = results.merge(fep_index, how="left", on="unique_id")
baseline_ls70_wpcc, baseline_ls70_wpcc_low, baseline_ls70_wpcc_high, baseline_ls70_wktau, baseline_ls70_wktau_low, baseline_ls70_wktau_high = get_weighted_stats(results, 'preds', 'Exp. dG (kcal/mol)', groups)

results = pd.read_csv('new_data/baseline/AEV-PLIG_fep_benchmark_pdbbind_ligsim60_predictions.csv')
results["preds"] = -R*T*np.log(10)*results['preds']
results = results.merge(fep_index, how="left", on="unique_id")
baseline_ls60_wpcc, baseline_ls60_wpcc_low, baseline_ls60_wpcc_high, baseline_ls60_wktau, baseline_ls60_wktau_low, baseline_ls60_wktau_high = get_weighted_stats(results, 'preds', 'Exp. dG (kcal/mol)', groups)

# bindingnet_bindingdb
results = pd.read_csv('new_data/bindingnet_bindingdb/AEV-PLIG_fep_benchmark_pdbbind_U_bindingnet_U_bindingdb_ligsim90_predictions.csv')
results["preds"] = -R*T*np.log(10)*results['preds']
results = results.merge(fep_index, how="left", on="unique_id")
bindingnet_bindingdb_ls90_wpcc, bindingnet_bindingdb_ls90_wpcc_low, bindingnet_bindingdb_ls90_wpcc_high, bindingnet_bindingdb_ls90_wktau, bindingnet_bindingdb_ls90_wktau_low, bindingnet_bindingdb_ls90_wktau_high  = get_weighted_stats(results, 'preds', 'Exp. dG (kcal/mol)', groups)

results = pd.read_csv('new_data/bindingnet_bindingdb/AEV-PLIG_fep_benchmark_pdbbind_U_bindingnet_U_bindingdb_ligsim80_predictions.csv')
results["preds"] = -R*T*np.log(10)*results['preds']
results = results.merge(fep_index, how="left", on="unique_id")
bindingnet_bindingdb_ls80_wpcc, bindingnet_bindingdb_ls80_wpcc_low, bindingnet_bindingdb_ls80_wpcc_high, bindingnet_bindingdb_ls80_wktau, bindingnet_bindingdb_ls80_wktau_low, bindingnet_bindingdb_ls80_wktau_high  = get_weighted_stats(results, 'preds', 'Exp. dG (kcal/mol)', groups)

results = pd.read_csv('new_data/bindingnet_bindingdb/AEV-PLIG_fep_benchmark_pdbbind_U_bindingnet_U_bindingdb_ligsim70_predictions.csv')
results["preds"] = -R*T*np.log(10)*results['preds']
results = results.merge(fep_index, how="left", on="unique_id")
bindingnet_bindingdb_ls70_wpcc, bindingnet_bindingdb_ls70_wpcc_low, bindingnet_bindingdb_ls70_wpcc_high, bindingnet_bindingdb_ls70_wktau, bindingnet_bindingdb_ls70_wktau_low, bindingnet_bindingdb_ls70_wktau_high  = get_weighted_stats(results, 'preds', 'Exp. dG (kcal/mol)', groups)

results = pd.read_csv('new_data/bindingnet_bindingdb/AEV-PLIG_fep_benchmark_pdbbind_U_bindingnet_U_bindingdb_ligsim60_predictions.csv')
results["preds"] = -R*T*np.log(10)*results['preds']
results = results.merge(fep_index, how="left", on="unique_id")
bindingnet_bindingdb_ls60_wpcc, bindingnet_bindingdb_ls60_wpcc_low, bindingnet_bindingdb_ls60_wpcc_high, bindingnet_bindingdb_ls60_wktau, bindingnet_bindingdb_ls60_wktau_low, bindingnet_bindingdb_ls60_wktau_high  = get_weighted_stats(results, 'preds', 'Exp. dG (kcal/mol)', groups)

df = pd.DataFrame(
    {
        "Tanimoto": [0.9, 0.8, 0.7, 0.6],
        "WPCC": [bindingnet_bindingdb_ls90_wpcc, bindingnet_bindingdb_ls80_wpcc, bindingnet_bindingdb_ls70_wpcc, bindingnet_bindingdb_ls70_wpcc],
        "WPCC_low": [bindingnet_bindingdb_ls90_wpcc_low, bindingnet_bindingdb_ls80_wpcc_low, bindingnet_bindingdb_ls70_wpcc_low, bindingnet_bindingdb_ls70_wpcc_low],
        "WPCC_high": [bindingnet_bindingdb_ls90_wpcc_high, bindingnet_bindingdb_ls80_wpcc_high, bindingnet_bindingdb_ls70_wpcc_high, bindingnet_bindingdb_ls70_wpcc_high],
        "WKtau": [bindingnet_bindingdb_ls90_wktau, bindingnet_bindingdb_ls80_wktau, bindingnet_bindingdb_ls70_wktau, bindingnet_bindingdb_ls60_wktau],
        "WKtau_low": [bindingnet_bindingdb_ls90_wktau_low, bindingnet_bindingdb_ls80_wktau_low, bindingnet_bindingdb_ls70_wktau_low, bindingnet_bindingdb_ls60_wktau_low],
        "WKtau_high": [bindingnet_bindingdb_ls90_wktau_high, bindingnet_bindingdb_ls80_wktau_high, bindingnet_bindingdb_ls70_wktau_high, bindingnet_bindingdb_ls60_wktau_high],
    }
)

df_baseline = pd.DataFrame(
    {
        "Tanimoto": [0.9, 0.8, 0.7, 0.6],
        "WPCC": [baseline_ls90_wpcc, baseline_ls80_wpcc, baseline_ls70_wpcc, baseline_ls60_wpcc],
        "WPCC_low": [baseline_ls90_wpcc_low, baseline_ls80_wpcc_low, baseline_ls70_wpcc_low, baseline_ls60_wpcc_low],
        "WPCC_high": [baseline_ls90_wpcc_high, baseline_ls80_wpcc_high, baseline_ls70_wpcc_high, baseline_ls60_wpcc_high],
        "WKtau": [baseline_ls90_wktau, baseline_ls80_wktau, baseline_ls70_wktau, baseline_ls60_wktau],
        "WKtau_low": [baseline_ls90_wktau_low, baseline_ls80_wktau_low, baseline_ls70_wktau_low, baseline_ls60_wktau_low],
        "WKtau_high": [baseline_ls90_wktau_high, baseline_ls80_wktau_high, baseline_ls70_wktau_high, baseline_ls60_wktau_high],
    }
)


"""
Plot lineplot
"""

plot_lineplot_with_error_bars(
    [fep_wpcc, fep_wpcc_low, fep_wpcc_high],
    [fep_wktau, fep_wktau_low, fep_wktau_high],
    df,
    df_baseline,
    "FEP+",
    "AEV-PLIG (PDBbind+BindingNet+BindingDB)",
    "AEV-PLIG (PDBbind only)",
    [0.2, 0.8],
    [0.0, 0.6],
    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    xlabel="Maximum Tanimoto similarity",
    invert_xaxis=True,
    outpath="figures/new_figure4.png",
)


"""
Plot barplot for the biggest series, remove the duplicate targets like BACE
"""

def plot_bars_comparison(df1, df2, df3, column1, column2, xlabel='', y1_label='', y2_label='', y1_lim=(0,1), y2_lim=(0,2), outpath=None):
    
    labels = df1.Target
    x = [x for x in range(14)]
    
    bar_width = 0.23

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,6), sharex=True, constrained_layout=True)
    
    bars1_1 = ax1.bar(x, df1[column1], bar_width, label='AEV-PLIG (baseline)', edgecolor='black')
    bars1_2 = ax1.bar([i + bar_width for i in x], df2[column1], bar_width, label='AEV-PLIG', edgecolor='black')
    bars1_3 = ax1.bar([i + 2*bar_width for i in x], df3[column1], bar_width, label='FEP+', edgecolor='black')

    bars2_1 = ax2.bar(x, df1[column2], bar_width, label='AEV-PLIG', edgecolor='black')
    bars2_2 = ax2.bar([i + bar_width for i in x], df2[column2], bar_width, label='AEV-PLIG (enriched)', edgecolor='black')
    bars2_3 = ax2.bar([i + 2*bar_width for i in x], df3[column2], bar_width, label='FEP', edgecolor='black')

    ax1.set_ylim(y1_lim)
    ax2.set_ylim(y2_lim)

    ax2.set_xlabel(xlabel, labelpad=1)
    ax1.set_ylabel(y1_label)
    ax2.set_ylabel(y2_label)

    ax1.set_xticks([i + bar_width for i in x])
    ax2.set_xticks([i + bar_width for i in x])
    ax2.set_xticklabels(labels)
    ax1.legend(ncol=3, frameon=False, loc='upper left', bbox_to_anchor=(0.24, 1.25))
    
    plt.xticks(rotation=90)

    if outpath:
        plt.tight_layout(pad=1)
        plt.savefig(outpath, dpi=500)

# load FEP+ results
results = pd.read_csv('new_data/fep/fep_benchmark_fep+_predictions.csv', index_col=0)
counts = pd.DataFrame(results["group_id"].value_counts())
groups = list(counts[counts["count"] >= 25].index)
groups.remove('jacs_set/bace')
performance_fep = get_performance_df(results, 'Pred. dG (kcal/mol)', 'Exp. dG (kcal/mol)', groups)
print("FEP+")
print("WPCC:", weighted_mean(np.array(performance_fep["Sample size"]), np.array(performance_fep["Pearson's R"])))
print("WKtau:", weighted_mean(np.array(performance_fep["Sample size"]), np.array(performance_fep["Kendall's Tau"])))

# load AEV-PLIG results
fep_index = pd.read_csv('new_data/fep/fep_benchmark_fep+_predictions.csv', index_col=0)
fep_index = fep_index[["graph_id", "Exp. dG (kcal/mol)"]]
fep_index = fep_index.rename(columns={"graph_id":"unique_id"})
results = pd.read_csv('new_data/bindingnet_bindingdb/AEV-PLIG_fep_benchmark_pdbbind_U_bindingnet_U_bindingdb_ligsim90_predictions.csv')
results["preds"] = -R*T*np.log(10)*results['preds']
results = results.merge(fep_index, how="left", on="unique_id")
performance_aevplig = get_performance_df(results, 'preds', 'Exp. dG (kcal/mol)', groups)
print("AEV-PLIG")
print("WPCC:", weighted_mean(np.array(performance_aevplig["Sample size"]), np.array(performance_aevplig["Pearson's R"])))
print("WKtau:", weighted_mean(np.array(performance_aevplig["Sample size"]), np.array(performance_aevplig["Kendall's Tau"])))

# baseline AEV-PLIG
results = pd.read_csv('new_data/baseline/AEV-PLIG_fep_benchmark_pdbbind_ligsim90_predictions.csv')
results["preds"] = -R*T*np.log(10)*results['preds']
results = results.merge(fep_index, how="left", on="unique_id")
performance_baseline = get_performance_df(results, 'preds', 'Exp. dG (kcal/mol)', groups)
print("AEV-PLIG baseline")
print("WPCC:", weighted_mean(np.array(performance_baseline["Sample size"]), np.array(performance_baseline["Pearson's R"])))
print("WKtau:", weighted_mean(np.array(performance_baseline["Sample size"]), np.array(performance_baseline["Kendall's Tau"])))

# Change Target names
target_dict = {'gpcrs/ox2_hip_custcore':'OX2',
 'merck/syk_4puz_fullmap':'SYK',
 'jacs_set/mcl1_extra_flips':'Mcl1',
 'merck/hif2a_automap':'HIF2α',
 'merck/pfkfb3_automap':'PFKFB3',
 'jacs_set/p38':'p38',
 'merck/eg5_extraprotomers':'Eg5',
 'janssen_bace/bace_ciordia_retro':'BACE1',
 'merck/tnks2_fullmap':'TNKS2',
 'merck/cdk8_5cei_new_helix_loop_extra':'CDK8',
 'mcs_docking/renin_customcore':'Renin',
 'opls_stress/fxa_yoshikawa_set':'Factor Xa',
 'misc/galectin3_extra':'Galectin',
 'merck/shp2':'SHP-2'}

performance_fep['Target'] = performance_fep['Target'].map(target_dict)
performance_aevplig['Target'] = performance_aevplig['Target'].map(target_dict)
performance_baseline['Target'] = performance_baseline['Target'].map(target_dict)

# print table of performance
performance = performance_fep.merge(performance_aevplig, on='Target', suffixes=('_fep', '_aevplig'))
performance = performance[['Target',"Pearson's R_fep","Pearson's R_aevplig",
                        "Kendall's Tau_fep", "Kendall's Tau_aevplig",
                        'RMSE_fep', 'RMSE_aevplig']]

# plot barplot
plot_bars_comparison(performance_baseline, performance_aevplig, performance_fep, "Pearson's R", "Kendall's Tau", xlabel='Target', y1_label='PCC', y2_label=r"K$\tau$", y1_lim=(-0.2,1), y2_lim=(-0.2, 1), outpath='figures/new_figure6.png')