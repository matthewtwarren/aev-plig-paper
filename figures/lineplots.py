import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
import warnings
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['font.size'] = 10

def pearson(x, y):
    rp = np.corrcoef(x, y)[0,1]
    return rp

def rmse(x, y):
    return float(np.sqrt(mean_squared_error(x, y)))

def kendalltau(x, y):
    return float(stats.kendalltau(x, y)[0])

def weighted_pearson(preds, truth, target):
    warnings.filterwarnings("error")
    results = pd.DataFrame(data={"preds":preds, "truth":truth, "target":target})
    targets = list(range(8))
    logs = []
    for target in targets:
        try:
            subset = results[results["target"] == target]
            logs.append([target, pearson(subset["preds"], subset["truth"]), len(subset)])
        except RuntimeWarning:
            pass
    performance = pd.DataFrame(logs, columns=["Target","Pearson's R","Sample size"])
    return np.average(performance["Pearson's R"], weights=performance["Sample size"])

def weighted_kendall(preds, truth, target):
    results = pd.DataFrame(data={"preds":preds, "truth":truth, "target":target})
    targets = list(range(8))
    logs = []
    for target in targets:
        subset = results[results["target"] == target]
        kendall = kendalltau(subset["preds"], subset["truth"])
        if(np.isnan(kendall)):
            pass
        else:
            logs.append([target, kendall, len(subset)])

    performance = pd.DataFrame(logs, columns=["Target","Kendall","Sample size"])
    return np.average(performance["Kendall"], weights=performance["Sample size"])

def weighted_rmse(preds, truth, target):
    results = pd.DataFrame(data={"preds":preds, "truth":truth, "target":target})
    targets = list(range(8))
    logs = []
    for target in targets:
        subset = results[results["target"] == target]
        if(len(subset) == 0):
            pass
        else:
            logs.append([target, rmse(subset["preds"], subset["truth"]), len(subset)])

    performance = pd.DataFrame(logs, columns=["Target","RMSE","Sample size"])
    return np.average(performance["RMSE"], weights=performance["Sample size"])

def get_weighted_stats_short(results, preds_col, truth_col, targets):
    results['pred'] = results[preds_col]
    results['truth'] = results[truth_col]
    
    logs = []
    for target in targets:
        subset = results[results["Target"] == target]
        logs.append([target, pearson(subset["pred"], subset["truth"]), kendalltau(subset["pred"], subset["truth"]), rmse(subset["pred"], subset["truth"]), len(subset)])

    performance = pd.DataFrame(logs, columns=["Target","Pearson's R","Kendall's Tau","RMSE","Sample size"])

    wpcc = np.average(performance["Pearson's R"], weights=performance["Sample size"])
    wktau = np.average(performance["Kendall's Tau"], weights=performance["Sample size"])
    wrmse = np.average(performance["RMSE"], weights=performance["Sample size"])

    return wpcc, wktau, wrmse

def plot_lineplot(
    x,
    y1,
    y2,
    y1_label,
    y2_label,
    y1_baseline,
    y2_baseline,
    y1_baseline_label,
    y2_baseline_label,
    y1_lims,
    y2_lims,
    y1_ticks,
    y2_ticks,
    xlabel,
    invert_xaxis=False,
    outpath=None,
):
    y1_color = "steelblue"
    y2_color = "firebrick"

    fig = plt.figure(figsize=(3.8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    # plot bindingnet data
    sns.lineplot(
        x=x,
        y=y1[0],
        label=y1_label,
        color=y1_color,
        marker="o",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax1,
    )
    sns.lineplot(
        x=x,
        y=y2[0],
        label=y2_label,
        color=y2_color,
        marker="o",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax1,
    )

    sns.lineplot(
        x=x,
        y=y1[1],
        color=y1_color,
        marker="o",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax2,
        legend=False,
    )
    sns.lineplot(
        x=x,
        y=y2[1],
        color=y2_color,
        marker="o",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax2,
        legend=False,
    )

    # plot baseline data
    sns.lineplot(
        x=x,
        y=y1_baseline[0],
        label=y1_baseline_label,
        color=y1_color,
        marker="^",
        linestyle="--",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax1,
    )
    sns.lineplot(
        x=x,
        y=y2_baseline[0],
        label=y2_baseline_label,
        color=y2_color,
        marker="^",
        linestyle="--",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax1,
    )

    sns.lineplot(
        x=x,
        y=y1_baseline[1],
        color=y1_color,
        marker="^",
        linestyle="--",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax2,
        legend=False,
    )
    sns.lineplot(
        x=x,
        y=y2_baseline[1],
        color=y2_color,
        marker="^",
        linestyle="--",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax2,
        legend=False,
    )
    
    ax1.set_ylabel("Weighted mean PCC")
    ax1.set_ylim(y1_lims)
    if invert_xaxis:
        ax1.invert_xaxis()
    ax1.set_yticks(y1_ticks)
    ax1.set_title("a)", loc="left", fontsize=14, position=(-0.25, 1.0))

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(r'Weighted mean K$\tau$')
    ax2.set_ylim(y2_lims)
    ax2.set_xticks(x)
    ax2.set_yticks(y2_ticks)
    ax2.set_title("b)", loc="left", fontsize=14, position=(-0.25, 1.0))
    
    ax1.legend(bbox_to_anchor=(0.5, 1.25), loc="center", frameon=False)

    plt.tight_layout()
    
    if outpath:
        plt.savefig(outpath)

def plot_lineplot_with_constants(
    x,
    y1,
    y2,
    y1_label,
    y2_label,
    y1_baseline,
    y2_baseline,
    y1_baseline_label,
    y2_baseline_label,
    y1_lims,
    y2_lims,
    y1_ticks,
    y2_ticks,
    xlabel,
    invert_xaxis=False,
    outpath=None,
):
    y1_color = "steelblue"
    y2_color = "firebrick"

    fig = plt.figure(figsize=(3.8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    ax1.axhline(
        y=y1_baseline[0], color=y1_color, linestyle="--", label=y1_baseline_label, lw=2
    )
    ax1.axhline(
        y=y2_baseline[0], color=y2_color, linestyle="--", label=y2_baseline_label, lw=2
    )

    ax2.axhline(
        y=y1_baseline[1], color=y1_color, linestyle="--", lw=2
    )
    ax2.axhline(
        y=y2_baseline[1], color=y2_color, linestyle="--", lw=2
    )

    sns.lineplot(
        x=x,
        y=y1[0],
        label=y1_label,
        color=y1_color,
        marker="o",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax1,
    )
    sns.lineplot(
        x=x,
        y=y2[0],
        label=y2_label,
        color=y2_color,
        marker="o",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax1,
    )

    sns.lineplot(
        x=x,
        y=y1[1],
        color=y1_color,
        marker="o",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax2,
        legend=False,
    )
    sns.lineplot(
        x=x,
        y=y2[1],
        color=y2_color,
        marker="o",
        ms=10,
        lw=2,
        markeredgecolor="black",
        markeredgewidth=1.1,
        ax=ax2,
        legend=False,
    )
    
    ax1.set_ylabel("Weighted mean PCC")
    ax1.set_ylim(y1_lims)
    if invert_xaxis:
        ax1.invert_xaxis()
    ax1.set_yticks(y1_ticks)
    ax1.set_title("a)", loc="left", fontsize=14, position=(-0.25, 1.0))

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(r'Weighted mean K$\tau$')
    ax2.set_ylim(y2_lims)
    ax2.set_xticks(x)
    ax2.set_yticks(y2_ticks)
    ax2.set_title("b)", loc="left", fontsize=14, position=(-0.25, 1.0))
    
    ax1.legend(bbox_to_anchor=(0.5, 1.25), loc="center", frameon=False)

    plt.tight_layout()
    
    if outpath:
        plt.savefig(outpath)


'''
Load results Schrodinger
'''
results = pd.read_csv('../data/baseline/schrodinger_pdbbind_fep_predictions.csv')
schro_baseline_wpcc, schro_baseline_wktau, schro_baseline_rmse = get_weighted_stats_short(results,'preds','Exp. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Thrombin","Tyk2"])
results = pd.read_csv('../data/baseline/schrodinger_pdbbind_ligsim90_predictions.csv')
schro_baseline_ls90_wpcc, schro_baseline_ls90_wktau, schro_baseline_ls90_rmse = get_weighted_stats_short(results,'preds','Exp. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Thrombin","Tyk2"])
results = pd.read_csv('../data/baseline/schrodinger_pdbbind_ligsim90_predictions.csv')
schro_baseline_ls80_wpcc, schro_baseline_ls80_wktau, schro_baseline_ls80_rmse = get_weighted_stats_short(results,'preds','Exp. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Thrombin","Tyk2"])
results = pd.read_csv('../data/baseline/schrodinger_pdbbind_ligsim70_predictions.csv')
schro_baseline_ls70_wpcc, schro_baseline_ls70_wktau, schro_baseline_ls70_rmse = get_weighted_stats_short(results,'preds','Exp. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Thrombin","Tyk2"])
results = pd.read_csv('../data/baseline/schrodinger_pdbbind_ligsim60_predictions.csv')
schro_baseline_ls60_wpcc, schro_baseline_ls60_wktau, schro_baseline_ls60_rmse = get_weighted_stats_short(results,'preds','Exp. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Thrombin","Tyk2"])

results = pd.read_csv('../data/bindingnet/schrodinger_pdbbind_U_bindingnet_fep_predictions.csv')
schro_bindingnet_wpcc, schro_bindingnet_wktau, schro_bindingnet_rmse = get_weighted_stats_short(results,'preds','Exp. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Thrombin","Tyk2"])
results = pd.read_csv('../data/bindingnet/schrodinger_pdbbind_U_bindingnet_ligsim90_predictions.csv')
schro_bindingnet_ls90_wpcc, schro_bindingnet_ls90_wktau, schro_bindingnet_ls90_rmse = get_weighted_stats_short(results,'preds','Exp. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Thrombin","Tyk2"])
results = pd.read_csv('../data/bindingnet/schrodinger_pdbbind_U_bindingnet_ligsim80_predictions.csv')
schro_bindingnet_ls80_wpcc, schro_bindingnet_ls80_wktau, schro_bindingnet_ls80_rmse = get_weighted_stats_short(results,'preds','Exp. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Thrombin","Tyk2"])
results = pd.read_csv('../data/bindingnet/schrodinger_pdbbind_U_bindingnet_ligsim70_predictions.csv')
schro_bindingnet_ls70_wpcc, schro_bindingnet_ls70_wktau, schro_bindingnet_ls70_rmse = get_weighted_stats_short(results,'preds','Exp. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Thrombin","Tyk2"])
results = pd.read_csv('../data/bindingnet/schrodinger_pdbbind_U_bindingnet_noligsim_predictions.csv')
schro_bindingnet_ls60_wpcc, schro_bindingnet_ls60_wktau, schro_bindingnet_ls60_rmse = get_weighted_stats_short(results,'preds','Exp. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Thrombin","Tyk2"])

'''
Load results Merck
'''
results = pd.read_csv('../data/baseline/merck_pdbbind_fep_predictions.csv')
merck_baseline_wpcc, merck_baseline_wktau, merck_baseline_rmse = get_weighted_stats_short(results,'preds','Exp. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])
results = pd.read_csv('../data/baseline/merck_pdbbind_ligsim90_predictions.csv')
merck_baseline_ls90_wpcc, merck_baseline_ls90_wktau, merck_baseline_ls90_rmse = get_weighted_stats_short(results,'preds','Exp. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])
results = pd.read_csv('../data/baseline/merck_pdbbind_ligsim90_predictions.csv')
merck_baseline_ls80_wpcc, merck_baseline_ls80_wktau, merck_baseline_ls80_rmse = get_weighted_stats_short(results,'preds','Exp. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])
results = pd.read_csv('../data/baseline/merck_pdbbind_ligsim70_predictions.csv')
merck_baseline_ls70_wpcc, merck_baseline_ls70_wktau, merck_baseline_ls70_rmse = get_weighted_stats_short(results,'preds','Exp. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])
results = pd.read_csv('../data/baseline/merck_pdbbind_ligsim60_predictions.csv')
merck_baseline_ls60_wpcc, merck_baseline_ls60_wktau, merck_baseline_ls60_rmse = get_weighted_stats_short(results,'preds','Exp. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])

results = pd.read_csv('../data/bindingnet/merck_pdbbind_U_bindingnet_fep_predictions.csv')
merck_bindingnet_wpcc, merck_bindingnet_wktau, merck_bindingnet_rmse = get_weighted_stats_short(results,'preds','Exp. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])
results = pd.read_csv('../data/bindingnet/merck_pdbbind_U_bindingnet_ligsim90_predictions.csv')
merck_bindingnet_ls90_wpcc, merck_bindingnet_ls90_wktau, merck_bindingnet_ls90_rmse = get_weighted_stats_short(results,'preds','Exp. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])
results = pd.read_csv('../data/bindingnet/merck_pdbbind_U_bindingnet_ligsim80_predictions.csv')
merck_bindingnet_ls80_wpcc, merck_bindingnet_ls80_wktau, merck_bindingnet_ls80_rmse = get_weighted_stats_short(results,'preds','Exp. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])
results = pd.read_csv('../data/bindingnet/merck_pdbbind_U_bindingnet_ligsim70_predictions.csv')
merck_bindingnet_ls70_wpcc, merck_bindingnet_ls70_wktau, merck_bindingnet_ls70_rmse = get_weighted_stats_short(results,'preds','Exp. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])
results = pd.read_csv('../data/bindingnet/merck_pdbbind_U_bindingnet_noligsim_predictions.csv')
merck_bindingnet_ls60_wpcc, merck_bindingnet_ls60_wktau, merck_bindingnet_ls60_rmse = get_weighted_stats_short(results,'preds','Exp. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])



'''
Plot lineplot of baseline v bindingnet results
'''
plot_lineplot(
    [1.0, 0.9, 0.8, 0.7, 0.6],
    [[merck_bindingnet_wpcc, merck_bindingnet_ls90_wpcc, merck_bindingnet_ls80_wpcc, merck_bindingnet_ls70_wpcc, merck_bindingnet_ls60_wpcc], [merck_bindingnet_wktau, merck_bindingnet_ls90_wktau, merck_bindingnet_ls80_wktau, merck_bindingnet_ls70_wktau, merck_bindingnet_ls60_wktau]],
    [[schro_bindingnet_wpcc, schro_bindingnet_ls90_wpcc, schro_bindingnet_ls80_wpcc, schro_bindingnet_ls70_wpcc, schro_bindingnet_ls60_wpcc], [schro_bindingnet_wktau, schro_bindingnet_ls90_wktau, schro_bindingnet_ls80_wktau, schro_bindingnet_ls70_wktau, schro_bindingnet_ls60_wktau]],
     "Schindler (PDBbind + BindingNet)", "Wang (PDBbind + BindingNet)",
    [[merck_baseline_wpcc, merck_baseline_ls90_wpcc, merck_baseline_ls80_wpcc, merck_baseline_ls70_wpcc, merck_baseline_ls60_wpcc], [merck_baseline_wktau, merck_baseline_ls90_wktau, merck_baseline_ls80_wktau, merck_baseline_ls70_wktau, merck_baseline_ls60_wktau]],
    [[schro_baseline_wpcc, schro_baseline_ls90_wpcc, schro_baseline_ls80_wpcc, schro_baseline_ls70_wpcc, schro_baseline_ls60_wpcc], [schro_baseline_wktau, schro_baseline_ls90_wktau, schro_baseline_ls80_wktau, schro_baseline_ls70_wktau, schro_baseline_ls60_wktau]],
    "Schindler (PDBbind only)", "Wang (PDBbind only)",
    [0.2, 0.8],
    [0.0, 0.6],
    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    xlabel="Maximum Tanimoto similarity",
    invert_xaxis=True,
    outpath="figure4.png",
    )


'''
Load Enriched training results for Schrodinger and Merck
'''
results = pd.read_csv('../data/bindingnet/schrodinger_pdbbind_U_bindingnet_ligsim90_predictions.csv')
schro_0_wpcc, schro_0_wktau, schro_0_rmse = get_weighted_stats_short(results,'preds','Exp. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Thrombin","Tyk2"])
results = pd.read_csv('../data/bindingnet_enriched/schrodinger_pdbbind_U_bindingnet_ligsim90_enriched3_predictions.csv')
schro_3_wpcc, schro_3_wktau, schro_3_rmse = get_weighted_stats_short(results,'preds','Exp. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Thrombin","Tyk2"])
results = pd.read_csv('../data/bindingnet_enriched/schrodinger_pdbbind_U_bindingnet_ligsim90_enriched6_predictions.csv')
schro_6_wpcc, schro_6_wktau, schro_6_rmse = get_weighted_stats_short(results,'preds','Exp. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Thrombin","Tyk2"])
results = pd.read_csv('../data/bindingnet_enriched/schrodinger_pdbbind_U_bindingnet_ligsim90_enriched9_predictions.csv')
schro_9_wpcc, schro_9_wktau, schro_9_rmse = get_weighted_stats_short(results,'preds','Exp. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Tyk2"])

results = pd.read_csv('../data/bindingnet/merck_pdbbind_U_bindingnet_ligsim90_predictions.csv')
merck_0_wpcc, merck_0_wktau, merck_0_rmse = get_weighted_stats_short(results,'preds','Exp. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])
results = pd.read_csv('../data/bindingnet_enriched/merck_pdbbind_U_bindingnet_ligsim90_enriched3_predictions.csv')
merck_3_wpcc, merck_3_wktau, merck_3_rmse = get_weighted_stats_short(results,'preds','Exp. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])
results = pd.read_csv('../data/bindingnet_enriched/merck_pdbbind_U_bindingnet_ligsim90_enriched6_predictions.csv')
merck_6_wpcc, merck_6_wktau, merck_6_rmse = get_weighted_stats_short(results,'preds','Exp. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])
results = pd.read_csv('../data/bindingnet_enriched/merck_pdbbind_U_bindingnet_ligsim90_enriched9_predictions.csv')
merck_9_wpcc, merck_9_wktau, merck_9_rmse = get_weighted_stats_short(results,'preds','Exp. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])

results = pd.read_csv("../data/fep/schrodinger_graphs.csv", index_col=0)
schro_fep_wpcc, schro_fep_wktau, schro_fep_rmse = get_weighted_stats_short(results,'Exp. dG','Pred. dG',["Bace","CDK2","Jnk1","MCL1","p38","PTP1B","Thrombin","Tyk2"]) 
results = pd.read_csv("../data/fep/merck_graphs.csv", index_col=0)
merck_fep_wpcc, merck_fep_wktau, merck_fep_rmse = get_weighted_stats_short(results,'Exp. ΔG','Pred. ΔG',["cdk8","cmet","eg5","hif2a","pfkfb3","shp2","syk","tnks2"])


'''
Plot lineplot of enrichment results
'''
plot_lineplot_with_constants(
    [0, 3, 6, 9],
    [[merck_0_wpcc, merck_3_wpcc, merck_6_wpcc, merck_9_wpcc], [merck_0_wktau, merck_3_wktau, merck_6_wktau, merck_9_wktau]],
    [[schro_0_wpcc, schro_3_wpcc, schro_6_wpcc, schro_9_wpcc], [schro_0_wktau, schro_3_wktau, schro_6_wktau, schro_9_wktau]],
    "Schindler (AEV-PLIG)", "Wang (AEV-PLIG)",
    [merck_fep_wpcc, merck_fep_wktau],
    [schro_fep_wpcc, schro_fep_wktau],
    "Schindler (FEP+)", "Wang (FEP+)",
    [0.4, 0.8],
    [0.2, 0.6],
    [0.4, 0.5, 0.6, 0.7, 0.8],
    [0.2, 0.3, 0.4, 0.5, 0.6],
    xlabel="Number of enrichment ligands",
    outpath="figure5.png"
)