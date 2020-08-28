# %%
import numpy as np
import pandas as pd
from outcome_association_utils import regression_model, logistic_regression_model, random_forest_model, plot_rsquared_covariates, unpack_disease


# %%
# Linear regression on exercise ECGs
########### Pretest exercise ECGs ###########################
# Read phenotype and covariates
phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/pretest_covariates.csv')
prs_score = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/hrr_score.csv')
# phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/rest_covariates.csv')
# phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/overlap_covariates.csv')

phenotypes = phenotypes.merge(prs_score[['SID', 'hrr_score', 'pc_1', 'pc_2', 'pc_3', 'pc_4', 'pc_5']], left_on='sample_id', right_on='SID')
phenotypes['50_hrr_actual_binary'] = (phenotypes['50_hrr_actual'] < phenotypes['50_hrr_actual'].quantile(0.33)).apply(float)
label_dic = {
    'hrr_score': ['HRR PRS', ''],
    '50_hrr_actual': ['HRR', 'beats'],
    '50_hrr_actual_binary': ['lowest tertile HRR', 'beats'],
    '50_hrr_downsample_augment_prediction': ['HRR$_{pred}$', 'beats'],
    '50_hrr_downsample_augment_prediction_2': ['HRR$_{pred}$', 'beats'],
    'resting_hr': ['Rest HR', 'beats'],
    'age': ['Age', 'yrs'],
    'male': ['Male', ''],
    'nonwhite': ['Nonwhite', ''],
    'bmi': ['BMI', 'units'],
    'cholesterol': ['Cholesterol', 'mmol/L'],
    'HDL': ['HDL', 'mmol/L'],
    'current_smoker': ['Current smoker', ''],
    'diastolic_bp': ['Diastolic blood pressure', 'mmHg'],
    'systolic_bp': ['Systolic blood pressure', 'mmHg'],
    # 'gfr': ['eGFR', 'mL/min/1.73 m2'],
    # 'creatinine': ['Creatinine', 'umol/L'],
    'Diabetes_Type_2_prevalent': ['Prevalent DM', ''],
    'c_lipidlowering': ['Lipid lowering drugs', ''],
    'c_antihypertensive': ['Antihypertensive drugs', '']
}

dont_scale = ['male', 'nonwhite', 'current_smoker', 'c_lipidlowering', 'c_antihypertensive', '50_hrr_actual_binary', '50_hrr_downsample_augment_prediction_binary', 'Diabetes_Type_2_prevalent']

# %%
# Read diseases and unpack
diseases = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/bq_diseases.tsv', sep='\t')
diseases['censor_date'] = pd.to_datetime(diseases['censor_date'])

disease_list = [['Diabetes_Type_2', 'type 2 diabetes'],
                ]
diseases_unpack = unpack_disease(diseases, disease_list, phenotypes)

tmp_data = phenotypes.merge(diseases_unpack[['sample_id', f'Diabetes_Type_2_censor_date', f'Diabetes_Type_2_prevalent']], on='sample_id')

# %%
import matplotlib.pyplot as plt
covariates = ['age', 'male', 'bmi', 'pc_1', 'pc_2', 'pc_3', 'pc_4', 'pc_5', 'hrr_score']
dont_scale = ['50_hrr_actual', 'resting_hr', '50_hrr_downsample_augment_prediction', 'male', 'nonwhite', 'current_smoker', 'c_lipidlowering', 'c_antihypertensive', '50_hrr_actual_binary', '50_hrr_downsample_augment_prediction_binary']

beta_values = {}
f, ax = plt.subplots()
f.set_size_inches(3.5, 3.5)
yticks = []
labels = []

for i, pheno in enumerate(['50_hrr_actual', '50_hrr_downsample_augment_prediction', 'resting_hr']):
    res = regression_model(phenotypes, [pheno], label_dic, covariates, dont_scale)
    beta_values[pheno] = {}
    beta_values[pheno]['beta'] = res.params['hrr_score']
    print(res.pvalues['hrr_score'])
    beta_values[pheno]['conf_int'] = res.conf_int().loc['hrr_score'][1] - res.params['hrr_score']
    ax.errorbar([beta_values[pheno]['beta']], [2-i], xerr=[beta_values[pheno]['conf_int']], marker='o', color='black')
    yticks.append(2-i)
    labels.append(label_dic[pheno][0])
ax.set_yticks(yticks)
ax.plot([0.0, 0.0], [-1.0, 3.0], 'k--')
ax.set_ylim(-0.5, 2.5)
ax.set_xlim([-0.8, 1.2])
ax.set_xticks(np.arange(-0.5, 1.1, 0.5))
ax.set_yticklabels(labels)
ax.set_xlabel('$\\beta$ (bpm) per 1-SD increase of HRR PRS')
plt.tight_layout()
f.savefig('hrr_prs_beta.png', dpi=500)


# %%
phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/pretest_covariates.csv')
prs_score = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/hrr_score_only.csv')
# phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/rest_covariates.csv')
# phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/overlap_covariates.csv')

phenotypes = phenotypes.merge(prs_score[['SID', 'hrr_score', 'pc_1', 'pc_2', 'pc_3', 'pc_4', 'pc_5']], left_on='sample_id', right_on='SID')

# %%
import matplotlib.pyplot as plt

beta_values = {}
f, ax = plt.subplots()
f.set_size_inches(3.5, 3.5)
yticks = []
labels = []

for i, pheno in enumerate(['50_hrr_actual', '50_hrr_downsample_augment_prediction', 'resting_hr']):
    res = regression_model(phenotypes, [pheno], label_dic, covariates, dont_scale)
    beta_values[pheno] = {}
    beta_values[pheno]['beta'] = res.params['hrr_score']
    print(beta_values[pheno]['beta'])
    beta_values[pheno]['conf_int'] = res.conf_int().loc['hrr_score'][1] - res.params['hrr_score']
    ax.errorbar([beta_values[pheno]['beta']], [2-i], xerr=[beta_values[pheno]['conf_int']], marker='o', color='black')
    yticks.append(2-i)
    labels.append(label_dic[pheno][0])
ax.set_yticks(yticks)
ax.plot([0.0, 0.0], [-1.0, 3.0], 'k--')
ax.set_ylim(-0.5, 2.5)
ax.set_xlim([-0.8, 1.2])
ax.set_xticks(np.arange(-0.5, 1.1, 0.5))
ax.set_yticklabels(labels)
ax.set_xlabel('$\\beta$ (bpm) per 1-SD increase of HRR PRS')
plt.tight_layout()
f.savefig('hrr_only_prs_beta.png', dpi=500)

# %%
phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/pretest_covariates.csv')
prs_score = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/hrr_shared_score.csv')
# phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/rest_covariates.csv')
# phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/overlap_covariates.csv')

phenotypes = phenotypes.merge(prs_score[['SID', 'hrr_score', 'pc_1', 'pc_2', 'pc_3', 'pc_4', 'pc_5']], left_on='sample_id', right_on='SID')

# %%
import matplotlib.pyplot as plt

beta_values = {}
f, ax = plt.subplots()
f.set_size_inches(3.5, 3.5)
yticks = []
labels = []

for i, pheno in enumerate(['50_hrr_actual', '50_hrr_downsample_augment_prediction', 'resting_hr']):
    res = regression_model(phenotypes, [pheno], label_dic, covariates, dont_scale)
    beta_values[pheno] = {}
    beta_values[pheno]['beta'] = res.params['hrr_score']
    print(beta_values[pheno]['beta'])
    beta_values[pheno]['conf_int'] = res.conf_int().loc['hrr_score'][1] - res.params['hrr_score']
    ax.errorbar([beta_values[pheno]['beta']], [2-i], xerr=[beta_values[pheno]['conf_int']], marker='o', color='black')
    yticks.append(2-i)
    labels.append(label_dic[pheno][0])
ax.set_yticks(yticks)
ax.plot([0.0, 0.0], [-1.0, 3.0], 'k--')
ax.set_ylim(-0.5, 2.5)
ax.set_xlim([-0.8, 1.2])
ax.set_xticks(np.arange(-0.5, 1.1, 0.5))
ax.set_yticklabels(labels)
ax.set_xlabel('$\\beta$ (bpm) per 1-SD increase of HRR PRS')
plt.tight_layout()
f.savefig('hrr_shared_prs_beta.png', dpi=500)

# %%
phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/pretest_covariates.csv')
prs_score = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/m2pam_score.csv')
# phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/rest_covariates.csv')
# phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/overlap_covariates.csv')

phenotypes = phenotypes.merge(prs_score[['SID', 'hrr_score', 'pc_1', 'pc_2', 'pc_3', 'pc_4', 'pc_5']], left_on='sample_id', right_on='SID')

# %%
import matplotlib.pyplot as plt

beta_values = {}
f, ax = plt.subplots()
f.set_size_inches(3.5, 3.5)
yticks = []
labels = []

for i, pheno in enumerate(['50_hrr_actual', '50_hrr_downsample_augment_prediction', 'resting_hr']):
    res = regression_model(phenotypes, [pheno], label_dic, covariates, dont_scale)
    beta_values[pheno] = {}
    beta_values[pheno]['beta'] = res.params['hrr_score']
    print(res.pvalues['hrr_score'])
    beta_values[pheno]['conf_int'] = res.conf_int().loc['hrr_score'][1] - res.params['hrr_score']
    ax.errorbar([beta_values[pheno]['beta']], [2-i], xerr=[beta_values[pheno]['conf_int']], marker='o', color='black')
    yticks.append(2-i)
    labels.append(label_dic[pheno][0])
ax.set_yticks(yticks)
ax.plot([0.0, 0.0], [-1.0, 3.0], 'k--')
ax.set_ylim(-0.5, 2.5)
ax.set_xlim([-0.8, 1.2])
ax.set_xticks(np.arange(-0.5, 1.1, 0.5))
ax.set_yticklabels(labels)
ax.set_xlabel('$\\beta$ (bpm) per 1-SD increase of HRR PRS')
plt.tight_layout()
f.savefig('hrr_m2pam_prs_beta.png', dpi=500)

# %%
phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/pretest_covariates.csv')
prs_score = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/hrr_shared_score.csv')
# phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/rest_covariates.csv')
# phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/overlap_covariates.csv')

phenotypes = phenotypes.merge(prs_score[['SID', 'hrr_score', 'pc_1', 'pc_2', 'pc_3', 'pc_4', 'pc_5']], left_on='sample_id', right_on='SID')

# %%
import matplotlib.pyplot as plt
covariates = ['age', 'male', 'bmi', 'pc_1', 'pc_2', 'pc_3', 'pc_4', 'pc_5', 'hrr_score']
dont_scale = ['50_hrr_actual', '50_hrr_downsample_augment_prediction', 'male', 'nonwhite', 'current_smoker', 'c_lipidlowering', 'c_antihypertensive', '50_hrr_actual_binary', '50_hrr_downsample_augment_prediction_binary']

beta_values = {}
f, ax = plt.subplots()
yticks = []
labels = []

for i, pheno in enumerate(['50_hrr_actual', '50_hrr_downsample_augment_prediction', 'resting_hr']):
    res = regression_model(phenotypes, [pheno], label_dic, covariates, dont_scale)
    beta_values[pheno] = {}
    beta_values[pheno]['beta'] = res.params['hrr_score']
    print(beta_values[pheno]['beta'])
    beta_values[pheno]['conf_int'] = res.conf_int().loc['hrr_score'][1] - res.params['hrr_score']
    ax.errorbar([beta_values[pheno]['beta']], [2-i], xerr=[beta_values[pheno]['conf_int']], marker='o', color='black')
    yticks.append(2-i)
    labels.append(label_dic[pheno][0])
ax.set_yticks(yticks)
ax.plot([0.0, 0.0], [-1.0, 3.0], 'k--')
ax.set_ylim(-0.5, 2.5)
ax.set_yticklabels(labels)
ax.set_xlabel('$\\beta$ (bpm) per 1-SD increase of HRR PRS')
f.savefig('hrr_shared_prs_beta.png', dpi=500)

# %%
covariates = []
rsquared = {}
for covariate in ['hrr_score', 'resting_hr', 'bmi', 'age', 'male']:
    covariates.append(covariate)

    res = regression_model(phenotypes, ['50_hrr_do'], label_dic, covariates, dont_scale)
    
    rsquared[covariate] = {}
    rsquared[covariate]['mean'] = np.mean(res.rsquared)
    rsquared[covariate]['std'] = np.std(0.0)

plot_rsquared_covariates(rsquared, label_dic, 'no_cnn', xlabel=False)

# %%
# Read diseases and unpack
diseases = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/bq_diseases.tsv', sep='\t')
diseases['censor_date'] = pd.to_datetime(diseases['censor_date'])

disease_list = [['Heart_Failure_V2', 'heart failure'],
                ['Diabetes_Type_2', 'type 2 diabetes'],
                ['composite_mi_cad_stroke_hf', 'CAD+stroke+HF+MI'],
                ]
diseases_unpack = unpack_disease(diseases, disease_list, phenotypes)


# %%
covariates = []
rsquared = {}
for covariate in ['50_hrr_downsample_augment_prediction', 'hrr_score', 'bmi', 'age', 'male']:
    covariates.append(covariate)

    res = regression_model(phenotypes, ['50_hrr_actual'], label_dic, covariates, dont_scale)
    
    rsquared[covariate] = {}
    rsquared[covariate]['mean'] = np.mean(res.rsquared)
    rsquared[covariate]['std'] = np.std(0.0)

plot_rsquared_covariates(rsquared, label_dic, 'with_cnn')


# %%
dont_scale = ['male', 'nonwhite', 'current_smoker', 'c_lipidlowering', 'c_antihypertensive', '50_hrr_actual_binary', '50_hrr_downsample_augment_prediction_binary']

covariates = ['resting_hr', 'bmi', 'age', 'male', 'cholesterol', 'HDL', 'current_smoker',
              'systolic_bp', 'c_antihypertensive', 'c_lipidlowering']
rsquared = {}

res = regression_model(phenotypes, ['50_hrr_downsample_augment_prediction'], label_dic, covariates, dont_scale)

# %%
covariates = []
rsquared = {}
for covariate in ['50_hrr_downsample_augment_prediction', 'bmi', 'age', 'male', 'cholesterol', 'HDL', 'current_smoker',
                  'diastolic_bp', 'systolic_bp', 'c_antihypertensive', 'c_lipidlowering', 'resting_hr']:
    covariates.append(covariate)

    res = random_forest_model(phenotypes, ['50_hrr_actual'], label_dic, covariates, dont_scale)
    rsquared[covariate] = {}
    rsquared[covariate]['mean'] = np.mean(res)
    rsquared[covariate]['std'] = np.std(res)

plot_rsquared_covariates(rsquared, label_dic)
# %%
covariates = []
rsquared = {}
for covariate in ['hrr_score', 'resting_hr', 'bmi', 'age', 'male', 'Diabetes_Type_2_prevalent', 'cholesterol', 'HDL', 'current_smoker',
                  'systolic_bp', 'c_antihypertensive', 'c_lipidlowering', '50_hrr_downsample_augment_prediction']:
    covariates.append(covariate)

    res = random_forest_model(tmp_data, ['50_hrr_actual'], label_dic, covariates, dont_scale)
    rsquared[covariate] = {}
    rsquared[covariate]['mean'] = np.mean(res)
    rsquared[covariate]['std'] = np.std(res)


# %%
covariates = []
rsquared2 = {}
for covariate in ['50_hrr_downsample_augment_prediction']:
    covariates.append(covariate)

    res = random_forest_model(tmp_data, ['50_hrr_actual'], label_dic, covariates, dont_scale)
    rsquared2[covariate] = {}
    rsquared2[covariate]['mean'] = np.mean(res)
    rsquared2[covariate]['std'] = np.std(res)

# %%
for key in rsquared:
    key2 = f'{key}_2' if key in rsquared2 else key
    rsquared2[key2] = rsquared[key]
# %%
import importlib
import outcome_association_utils
importlib.reload(outcome_association_utils)
outcome_association_utils.plot_rsquared_covariates(rsquared2, label_dic, 'HRRpredr2', horizontal_line_y=11.75, start_plus = 1)
# %%
# Logistic regression for binary
# %%
from matplotlib import pyplot as plt
covariates = {'Age + Sex': ['age', 'male'],
              'Clinical model': ['age', 'male', 'bmi', 'cholesterol', 'HDL', 'current_smoker',              
                                 'systolic_bp', 'c_antihypertensive', 'c_lipidlowering'],
              'Age + Sex + HRR$_{pred}$': ['age', 'male', '50_hrr_downsample_augment_prediction'],
              'Clinical + HRR$_{pred}$': ['age', 'male', 'bmi', 'cholesterol', 'HDL', 'current_smoker',
              'systolic_bp', 'c_antihypertensive', 'c_lipidlowering', 
              '50_hrr_downsample_augment_prediction'],
              }


rsquaredbinary = {}
f, ax = plt.subplots()
for i, (label, covariate) in enumerate(covariates.items()):
    rsquaredbinary['cov'] = logistic_regression_model(phenotypes, ['50_hrr_actual_binary'], label_dic, covariate, dont_scale)
    fpr = np.mean(rsquaredbinary['cov']['fpr'], axis=0)
    tpr = np.mean(rsquaredbinary['cov']['tpr'], axis=0)
    auc = np.mean(rsquaredbinary['cov']['auc'], axis=0)
    auc_min = np.min(rsquaredbinary['cov']['auc'], axis=0)
    auc_max = np.max(rsquaredbinary['cov']['auc'], axis=0)
    gray_color = np.ones(3) - (i+1) * 0.2
    ax.plot(fpr, tpr, label=f'{label} AUC={auc:.2f} [{auc_min:.2f}, {auc_max:.2f}]', color=gray_color, linewidth=3)
ax.set_aspect('equal')
ax.plot([0.0, 1.0], [0.0, 1.0], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xticks(np.arange(0.0, 1.1, 0.25))
ax.set_yticks(np.arange(0.0, 1.1, 0.25))
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.legend(prop={'size': 8})
plt.tight_layout()
f.savefig('roc_auc_hrr_lowest_tertile.png', dpi=500)
# for covariate in ['50_hrr_downsample_augment_prediction']:
#     covariates.append(covariate)

#     res = logistic_regression_model(phenotypes, ['50_hrr_actual_binary'], label_dic, covariates, dont_scale)
    # rsquaredbinary[covariate] = {}
    # rsquaredbinary[covariate]['mean'] = np.mean(res)
    # rsquaredbinary[covariate]['std'] = np.std(res)

# %%
f, ax = plt.subplots()
xticks = []
yticks = []
for quantile in np.arange(0.1, 1.0, 0.1):
    pred = phenotypes['50_hrr_downsample_augment_prediction'].quantile(quantile)
    hrr = phenotypes['50_hrr_actual'].quantile(quantile)
    xticks.append(hrr)
    yticks.append(pred)
    ax.plot(hrr, pred, 'ko')
    ax.plot([hrr, hrr], [0.0, pred], 'k--', linewidth=1)
    ax.plot([0.0, hrr], [pred, pred], 'k--', linewidth=1)
ax.set_aspect('equal')
ax.plot([0.0, 50.0], [0.0, 50.0], 'k-')
ax.set_xlim([13, 41])
ax.set_ylim([13, 41])
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([f'{tick:.0f}' for tick in xticks])
ax.set_yticklabels([f'{tick:.0f}' for tick in yticks])
ax.set_xlabel('HRR deciles (bpm)')
ax.set_ylabel('HRR$_{pred}$ deciles (bpm)')
plt.tight_layout()
f.savefig('decile_decile_plot.png', dpi=500)
# %%
from matplotlib import pyplot as plt
covariates = {'Age + Sex': ['age', 'male'],
              'Clinical model': ['age', 'male', 'bmi', 'cholesterol', 'HDL', 'current_smoker',              
                                 'systolic_bp', 'c_antihypertensive', 'c_lipidlowering'],
              'Age + Sex + Rest HR': ['age', 'male', 'resting_hr'],
              'Age + Sex + HRR$_{pred}$': ['age', 'male', '50_hrr_downsample_augment_prediction'],              
              }


rsquaredbinary = {}
f, ax = plt.subplots()
for i, (label, covariate) in enumerate(covariates.items()):
    rsquaredbinary['cov'] = logistic_regression_model(phenotypes, ['50_hrr_actual_binary'], label_dic, covariate, dont_scale)
    fpr = np.mean(rsquaredbinary['cov']['fpr'], axis=0)
    tpr = np.mean(rsquaredbinary['cov']['tpr'], axis=0)
    auc = np.mean(rsquaredbinary['cov']['auc'], axis=0)
    auc_min = np.min(rsquaredbinary['cov']['auc'], axis=0)
    auc_max = np.max(rsquaredbinary['cov']['auc'], axis=0)
    gray_color = np.ones(3) - (i+1) * 0.2
    ax.plot(fpr, tpr, label=f'{label} AUC={auc:.2f} [{auc_min:.2f}, {auc_max:.2f}]', color=gray_color, linewidth=3)
ax.set_aspect('equal')
ax.plot([0.0, 1.0], [0.0, 1.0], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xticks(np.arange(0.0, 1.1, 0.25))
ax.set_yticks(np.arange(0.0, 1.1, 0.25))
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.legend(prop={'size': 8})
plt.tight_layout()
f.savefig('roc_auc_hrr_rest_lowest_tertile.png', dpi=500)
# for covariate in ['50_hrr_downsample_augment_prediction']:
#     covariates.append(covariate)

#     res = logistic_regression_model(phenotypes, ['50_hrr_actual_binary'], label_dic, covariates, dont_scale)
    # rsquaredbinary[covariate] = {}
    # rsquaredbinary[covariate]['mean'] = np.mean(res)
    # rsquaredbinary[covariate]['std'] = np.std(res)

# %%
f, ax = plt.subplots()
xticks = []
yticks = []
for quantile in np.arange(0.1, 1.0, 0.1):
    pred = phenotypes['resting_hr'].quantile(quantile)
    hrr = phenotypes['50_hrr_actual'].quantile(quantile)
    xticks.append(hrr)
    yticks.append(pred)
    ax.plot(hrr, pred, 'ko')
    ax.plot([hrr, hrr], [0.0, pred], 'k--', linewidth=1)
    ax.plot([0.0, hrr], [pred, pred], 'k--', linewidth=1)
ax.set_aspect('equal')
ax.plot([0.0, 50.0], [0.0, 50.0], 'k-')
ax.set_xlim([13, 41])
ax.set_ylim([13, 41])
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([f'{tick:.0f}' for tick in xticks])
ax.set_yticklabels([f'{tick:.0f}' for tick in yticks])
ax.set_xlabel('HRR deciles (bpm)')
ax.set_ylabel('HRR$_{pred}$ deciles (bpm)')
plt.tight_layout()
f.savefig('decile_decile_rest_plot.png', dpi=500)
# %%
