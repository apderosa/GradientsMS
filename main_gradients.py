import warnings
warnings.simplefilter('ignore', category=FutureWarning)

import pandas as pd
from glob import glob
from scipy.io import loadmat
from scipy.stats import ttest_ind, chi2_contingency
import numpy as np

from brainspace.datasets import load_parcellation, load_conte69
from brainspace.utils.parcellation import map_to_labels
from brainspace.plotting import plot_hemispheres

import plotly.express as px

from pingouin import partial_corr
from scipy.stats import spearmanr, norm

from brainstat.stats.terms import FixedEffect
from brainstat.stats.SLM import SLM
from scipy import stats
from statsmodels.stats import multitest
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist, squareform
from pycirclize import Circos

from utils import delete_nan_subjects
from nested_CV import regression_nested_CV

figures_path = '/home/ws/Documenti/MS_Connectomics/figures'

########################################################################################################################
# Load dataset & demographic statistics

report_MS = pd.read_excel('/home/ws/Documenti/MS_Connectomics/preprocessed_data/ms_NAPOLI_rpt_2023-01-25-121134.xlsx')
report_HC = pd.read_excel('/home/ws/Documenti/MS_Connectomics/preprocessed_data/hc_NAPOLI_rpt_2023-01-25-121218.xlsx')
path_MS = sorted(glob('/home/ws/Documenti/MS_Connectomics/preprocessed_data/ms/sub*.mat/200cortical_z.mat'))
path_HC = sorted(glob('/home/ws/Documenti/MS_Connectomics/preprocessed_data/hc/sub*.mat/200cortical_z.mat'))

print('MS mean age: %0.1f (%0.1f)' % (np.mean(report_MS.age), np.std(report_MS.age)))
print('HC mean age: %0.1f (%0.1f)' % (np.mean(report_HC.age), np.std(report_HC.age)))

t_age, p_age = ttest_ind(report_MS.age, report_HC.age)
t_edu, p_edu = ttest_ind(report_MS.education, report_HC.education)

print('p-value for age difference: %0.2f' % p_age)
print('p-value for educational years difference: %0.2f' % p_edu)

report = pd.concat((report_MS, report_HC))
crosstab = pd.crosstab(report.sex, report.group)
chi_sex, p_sex, _, _ = chi2_contingency(crosstab)

print(crosstab)
print('p-value for sex difference: %0.2f' % p_sex)

matconn_MS = list()
matconn_HC = list()

for sub in range(0, len(path_MS)):
    mat = loadmat(path_MS[sub])['zconmat']
    # mat = np.arctanh(mat)
    matconn_MS.append(mat)
    del mat
for sub in range(0, len(path_HC)):
    mat = loadmat(path_HC[sub])['zconmat']
    # mat = np.arctanh(mat)
    matconn_HC.append(mat)
    del mat

matconn_MS, report_MS = delete_nan_subjects(matconn_MS, report_MS)
matconn_HC, report_HC = delete_nan_subjects(matconn_HC, report_HC)

report = pd.concat((report_MS, report_HC))

nMS = len(report_MS)
nHC = len(report_HC)

########################################################################################################################
# Load gradients

labeling = load_parcellation('schaefer', scale=200, join=True)
mask = labeling != 0
surf_lh, surf_rh = load_conte69()

schaefer_atlas = pd.read_excel('/home/ws/Documenti/MS_Connectomics/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.xlsx')
atlas_label = loadmat('/home/ws/Documenti/MS_Connectomics/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.mat')['Nets']
ind_gradients = loadmat('/home/ws/Documenti/MS_Connectomics/gradients_matlab_200z.mat')['ind_gradients'][:, :, :2]
group_gradients = loadmat('/home/ws/Documenti/MS_Connectomics/group_gradients_matlab_200z.mat')['group_gradients'][:, :2]

nTot = ind_gradients.shape[0]
nGr = ind_gradients.shape[2]

#grad = [None] * 2
#for g in range(0, nGr):
#    grad[g] = map_to_labels(group_gradients[:, g], labeling, mask=mask, fill=np.nan)
#
#plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 400), cmap='viridis_r',
#                 color_bar=True, label_text=['Gradient 1', 'Gradient 2'], zoom=1.55)

########################################################################################################################
# Set up model

age = np.concatenate((report_MS['age'], report_HC['age']))
sex = np.concatenate((report_MS['sex'], report_HC['sex']))
FD = np.concatenate((report_MS['mean_fd'], report_HC['mean_fd']))
bpf = np.concatenate((report_MS['BPF'], report_HC['BPF']))
vol = np.concatenate((report_MS['GM_vol']+report_MS['WM_vol']+report_MS['CSF_vol'], report_HC['GM_vol']+report_HC['WM_vol']+report_HC['CSF_vol']))
dd = np.concatenate((report_MS['disease_duration'], report_HC['disease_duration']))
grp = np.concatenate((report_MS['group'], report_HC['group']))

term_age = FixedEffect(age, 'age')
term_sex = FixedEffect(sex, 'sex')
term_fd = FixedEffect(FD, 'FD')
term_grp = FixedEffect(grp, 'group')

model = term_age + term_sex + term_fd + term_grp
contrast = (grp == 'MS').astype(int) - (grp == 'HC').astype(int)

########################################################################################################################
# Network gradients

nets = np.unique(atlas_label)
nNets = nets.shape[0]
nroi = atlas_label.shape[0]

networks = ['VN', 'SMN', 'DAN', 'VAN', 'LN', 'FPN', 'DMN']

nets_gradient = np.zeros((nTot, nNets, nGr))

for net in range(0, nNets):
    idx = np.where(atlas_label == nets[net])[0]
    nets_gradient[:, net, :] = np.mean(ind_gradients[:, idx, :], axis=1)

########################################################################################################################
# Regional gradients

slm = SLM(model, contrast, two_tailed=True)
slm.fit(ind_gradients)

pval = stats.t.sf(np.abs(slm.t), slm.df) * 2
pcor = multitest.multipletests(pval.ravel(), alpha=0.05, method='bonferroni')

sig_idx = np.where(pcor[1] < 0.05)
sig_t = np.zeros((nroi, 1))
sig_t[sig_idx] = slm.t.T[sig_idx]
print(sig_idx[0])

if len(sig_idx[0]) > 0:

    data = map_to_labels(sig_t[:, 0], labeling, mask, fill=np.nan)
    plot_hemispheres(surf_lh, surf_rh, array_name=data, size=(1200, 600), cmap='Reds', color_bar=True,
                     label_text=['Significant t'], zoom=1.25, transparent_bg=True, layout_style='grid')

sd_grad1 = np.std(ind_gradients[:, :, 0], axis=1)
sd_grad2 = np.std(ind_gradients[:, :, 1], axis=1)

slm = SLM(model, contrast, two_tailed=True)
slm.fit(np.concatenate((sd_grad1.reshape(-1, 1), sd_grad2.reshape(-1, 1)), axis=1))

pval = stats.t.sf(np.abs(slm.t), slm.df) * 2
pcor = multitest.multipletests(pval.ravel(), alpha=0.05, method='bonferroni')

df = pd.DataFrame({'Values': sd_grad2, 'Group': grp})
sns.boxplot(data=df, x='Group', y='Values')

net_var = np.zeros((nTot, nNets, nGr))
for g in range(nGr):
    for n in range(nNets):
        idx = np.where(atlas_label == nets[n])[0]
        for s in range(nTot):
            net_var[s, n, g] = np.std(ind_gradients[s, idx, g])

slm = SLM(model, contrast, two_tailed=True)
slm.fit(net_var[:, :, 1])

pval = stats.t.sf(np.abs(slm.t), slm.df) * 2
pcor = multitest.multipletests(pval.ravel(), alpha=0.05, method='bonferroni')

########################################################################################################################
# Network gradients comparison

slm = SLM(model, contrast, two_tailed=True)
slm.fit(nets_gradient)

pval = stats.t.sf(np.abs(slm.t), slm.df) * 2
pcor = multitest.multipletests(pval.ravel(), alpha=0.05, method='bonferroni')

sig_idx = np.where(pcor[1] < 0.05)
print(sig_idx[0])

for n in range(nNets):
    sig_t = np.zeros((nroi, 1))
    idx = np.where(atlas_label == nets[n])[0]
    sig_t[idx] = slm.t.T[n][0]
    data = map_to_labels(sig_t[:, 0], labeling, mask, fill=np.nan)
    plot_hemispheres(surf_lh, surf_rh, array_name=data, size=(1200, 600), cmap='Reds', color_bar=True,
                     label_text=['Significant t'], zoom=1.25, transparent_bg=True, layout_style='grid',
                     color_range=(0, 3.13))

df = pd.DataFrame(dict(value=np.squeeze(slm.t.T), variable=networks))
fig = px.line_polar(df, r='value', theta='variable', line_close=True, markers=True, color_discrete_sequence=['red'],
                    range_r=[0, 3.5])
fig.show()

sns.set(font_scale=2)
df = pd.DataFrame()
for g in range(0, nGr):
    plt.figure()
    for n in range(nNets):
        idx = np.where(atlas_label == nets[n])[0]
        values = ind_gradients[:, idx, g]
        for sub in range(nTot):
            df_sub = pd.DataFrame({'Value': values[sub].flatten(),
                                   'Net': [networks[n]]*values.shape[1],
                                   'Group': [grp[sub]]*values.shape[1]})
            df = pd.concat([df, df_sub])

    fig = sns.violinplot(data=df, x='Net', y='Value', hue='Group', split=True, palette=['red', 'cyan'])
    plt.show()
    plt.legend([], [], frameon=False)

########################################################################################################################
# Dispersion analysis

cog = np.full((nTot, nGr, nNets), np.nan)
wn_disp = np.full((nTot, nNets), np.nan)

for n in range(nNets):
    idx = np.where(atlas_label == nets[n])[0]
    for s in range(nTot):
        cog[s, :, n] = np.mean(ind_gradients[s, idx, :], axis=0)
        dist_to_centre = []
        for ii in range(0, len(idx)):
            dist_to_centre.append(cdist(ind_gradients[s, idx[ii], :].reshape(-1, 1).T, cog[s, :, n].reshape(-1, 1).T)[0][0])

        wn_disp[s, n] = np.sum(np.square(dist_to_centre))

slm = SLM(model, contrast, two_tailed=True)
slm.fit(wn_disp)

pval = stats.t.sf(np.abs(slm.t), slm.df) * 2
pcor = multitest.multipletests(pval.ravel(), alpha=0.05, method='bonferroni')

sig_idx = np.where(pcor[1] < 0.05)
sig_t = np.zeros((nNets, 1))
sig_t[sig_idx] = slm.t.T[sig_idx]
print(sig_idx[0])

for n in range(nNets):
    sig_t = np.zeros((nroi, 1))
    idx = np.where(atlas_label == nets[n])[0]
    sig_t[idx] = slm.t.T[n][0]
    data = map_to_labels(sig_t[:, 0], labeling, mask, fill=np.nan)
    plot_hemispheres(surf_lh, surf_rh, array_name=data, size=(1200, 600), cmap='Reds_r', color_bar=True,
                     label_text=['Significant t'], zoom=1.25, transparent_bg=True, layout_style='grid',
                     color_range=(-2.91, 0))

df = pd.DataFrame(dict(value=np.squeeze(slm.t.T), variable=networks))
fig = px.line_polar(df, r='value', theta='variable', line_close=True, markers=True, color_discrete_sequence=['red'])
fig.show()

net_dist = np.full((nNets, nNets, nTot), np.nan)
net_dist_long = []

for s in range(0, nTot):
    for n1 in range(0, nNets):
        for n2 in range(0, nNets):

            net_dist[n1, n2, s] = cdist(cog[s, :, n1].reshape(-1, 1).T, cog[s, :, n2].reshape(-1, 1).T)[0][0]

    net_dist_long.append(squareform(net_dist[:, :, s]))

slm = SLM(model, contrast, two_tailed=True)
slm.fit(np.asarray(net_dist_long))

pval = stats.t.sf(np.abs(slm.t), slm.df) * 2
pcor = multitest.multipletests(pval.ravel(), alpha=0.05, method='bonferroni')

sig_idx = np.where(pcor[1] < 0.05)
sig_t = np.zeros((pval.shape[1], 1))
sig_t[sig_idx] = slm.t.T[sig_idx]
print(sig_idx[0])

p_uncorr_matrix = squareform(np.squeeze(pval))
np.fill_diagonal(p_uncorr_matrix, 1)
p_corr_matrix = squareform(np.squeeze(pcor[1]))
np.fill_diagonal(p_corr_matrix, 1)
t_matrix = squareform(np.squeeze(slm.t.T))

t_uncorr = pd.DataFrame(np.abs(np.where(p_uncorr_matrix < 0.05, np.triu(t_matrix), 0)), index=networks, columns=networks)
t_corr = pd.DataFrame(np.abs(np.where(p_corr_matrix < 0.05, np.triu(t_matrix), 0)), index=networks, columns=networks)

circos = Circos.initialize_from_matrix(t_uncorr, space=5, cmap='tab10', label_kws=dict(size=24),
                                       link_kws=dict(ec='black', lw=0.5, direction=0))
circos.savefig('/home/ws/Immagini/t_uncorr.pdf')
circos = Circos.initialize_from_matrix(t_corr, space=5, cmap='tab10', label_kws=dict(size=24),
                                       link_kws=dict(ec='black', lw=0.5, direction=0))
circos.savefig('/home/ws/Immagini/t_corr.pdf')

########################################################################################################################
# Correlation with clinical variables

sdmt_corrected = report_MS.SDMT - 1.029 * (report_MS.education - 12.4)
sdmt_z = (sdmt_corrected - 50.9) / 9.4
sdmt = sdmt_z
#sdmt = report_MS.SDMT

edss = report_MS.EDSS

sex_bin = pd.factorize(sex)[0]

for k in range(0, 2):
    for i in range(0, 7):

        df = pd.DataFrame([nets_gradient[:nMS, i, k], sdmt, edss, age[:nMS], sex_bin[:nMS], FD[:nMS], ll[:nMS], bpf[:nMS], vol[:nMS], dd[:nMS]]).T
        df.columns = ['values', 'sdmt', 'edss', 'age', 'sex', 'fd', 'll', 'bpf', 'vol', 'dd']

        se = 1/np.sqrt(df.shape[0]-3)
        z = norm.ppf(1-0.05/2)

        r_sdmt, p_sdmt = spearmanr(df['values'], df['sdmt'])
        z_sdmt = np.arctanh(r_sdmt)
        ci_sdmt = [np.tanh(z_sdmt-z*se), np.tanh(z_sdmt+z*se)]
        r_edss, p_edss = spearmanr(df['values'], df['edss'])
        z_edss = np.arctanh(r_edss)
        ci_edss = [np.tanh(z_edss-z*se), np.tanh(z_edss+z*se)]
        r_ll, p_ll = spearmanr(df['values'], df['ll'])
        z_ll = np.arctanh(r_ll)
        ci_ll = [np.tanh(z_ll-z*se), np.tanh(z_ll+z*se)]
        r_vol, p_vol = spearmanr(df['values'], df['vol'])
        z_vol = np.arctanh(r_vol)
        ci_vol = [np.tanh(z_vol-z*se), np.tanh(z_vol+z*se)]
        r_dd, p_dd = spearmanr(df['values'], df['dd'])
        z_dd = np.arctanh(r_dd)
        ci_dd = [np.tanh(z_dd-z*se), np.tanh(z_dd+z*se)]
        r_bpf, p_bpf = spearmanr(df['values'], df['bpf'])
        z_bpf = np.arctanh(r_bpf)
        ci_bpf = [np.tanh(z_bpf-z*se), np.tanh(z_bpf+z*se)]

        if p_sdmt < 0.05/14:
            print('Significant correlation between SDMT and values from gradient %i of the %s: r = %.2f (p = %.3f) CI = [%.2f - %.2f], '
                  % (k+1, networks[i], r_sdmt, p_sdmt, ci_sdmt[0], ci_sdmt[1]))

        if p_edss < 0.05/14:
            print('Significant correlation between EDSS and values from gradient %i of the %s: r = %.2f (p = %.3f) CI = [%.2f - %.2f], '
                  % (k+1, networks[i], r_edss, p_edss, ci_edss[0], ci_edss[1]))

        if p_ll < 0.05/14:
            print('Significant correlation between lesion load and values from gradient %i of the %s: r = %.2f (p = %.3f) CI = [%.2f - %.2f], '
                  % (k+1, networks[i], r_ll, p_ll, ci_ll[0], ci_ll[1]))

        if p_vol < 0.05/14:
            print('Significant correlation between brain volume and values from gradient %i of the %s: r = %.2f (p = %.3f) CI = [%.2f - %.2f], '
                  % (k+1, networks[i], r_vol, p_vol, ci_vol[0], ci_vol[1]))

        if p_dd < 0.05/14:
            print('Significant correlation between disease duration and values from gradient %i of the %s: r = %.2f (p = %.3f) CI = [%.2f - %.2f], '
                  % (k+1, networks[i], r_dd, p_dd, ci_vol[0], ci_vol[1]))

        if p_bpf < 0.05/14:
            print('Significant correlation between BPF and values from gradient %i of the %s: r = %.2f (p = %.3f) CI = [%.2f - %.2f], '
                  % (k+1, networks[i], r_bpf, p_bpf, ci_bpf[0], ci_bpf[1]))


########################################################################################################################
# Machine learning analysis

sdmt_corrected = report_MS.SDMT - 1.029 * (report_MS.education - 12.4)
sdmt_z = (sdmt_corrected - 50.9) / 9.4

columns = ['Visual_G1', 'Somatomotor_G1', 'Dorsal_G1', 'Ventral_G1', 'Limbic_G1', 'Frontoparietal_G1', 'Default_G1',
           'Visual_G2', 'Somatomotor_G2', 'Dorsal_G2', 'Ventral_G2', 'Limbic_G2', 'Frontoparietal_G2', 'Default_G2']

columns = np.concatenate((columns, np.asarray(['SDMT'])))

grad1 = nets_gradient[:, :, 0]
grad2 = nets_gradient[:, :, 1]
full_grad = np.concatenate((grad1, grad2), axis=1)

X = full_grad[:nMS, :]
y = sdmt_z.ravel()

FG_dataset = pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1))
FG_dataset.columns = columns

regression_nested_CV(FG_dataset, 'gradients_2grad_vReviews')
