# %%
import numpy as np
import pandas as pd
%load_ext google.cloud.bigquery

# %% 
## %%bigquery covariates
# select sample_id, FieldID, instance, value from `ukbb7089_202006.phenotype` 
# where FieldID = 21001 -- bmi
# or FieldID = 21003 -- age at assessment
# or FieldID = 22001 -- genetic sex
# or FieldID = 31 -- sex
# or FieldID = 30690 -- cholesterol
# or FieldID = 30760 -- HDL cholesterol
# or FieldID = 20116 -- smoking
# or FieldID = 4079 -- diastolic bp
# or FieldID = 4080 -- systolic bp
# or FieldID = 95 -- pulse rate
# or FieldID = 53 -- instance 0 date
# or FieldID = 30700 -- creatinine
# # %%
# covariates.to_csv('bq_covariates.tsv', sep='\t')

# %%
# %%bigquery diseases
# select disease, sample_id, incident_disease, prevalent_disease, censor_date from `ukbb7089_202006.disease` 
# where has_disease > 0.5

# # %%
# diseases.to_csv('bq_diseases.tsv', sep='\t')

# %%

covariates = pd.read_csv('bq_covariates.tsv', sep='\t')
diseases = pd.read_csv('bq_diseases.tsv', sep='\t')
diseases['censor_date'] = pd.to_datetime(diseases['censor_date'])

# %%
pretest = pd.read_csv('pretest_inference_no_covs.tsv', sep='\t')
all_ecgs = pd.concat([pretest])

# %%
pheno_dic = {21001: ['bmi', float],
             21003: ['age', float],
             22001: ['genetic sex', int],
             31: ['sex', int],
             30690: ['cholesterol', float],
             30760: ['HDL', float],
             20116: ['smoking', int],
             4079: ['diastolic_bp', int],
             4080: ['systolic_bp', int],
             30700: ['creatinine', float],
             53: ['instance0_date', pd.to_datetime]
            }

for pheno in pheno_dic:
    tmp_covariates = covariates[(covariates['FieldID']==pheno) &\
                                (covariates['instance']==0)]
    if pheno == 53:
        tmp_covariates['value'] = pd.to_datetime(tmp_covariates['value'])
    else:
        tmp_covariates['value'] = tmp_covariates['value'].apply(pheno_dic[pheno][1])
    if (pheno == 4079) or (pheno == 4080):
        tmp_covariates = tmp_covariates[['sample_id', 'value']].groupby('sample_id').mean().reset_index(level=0)
    
    all_ecgs = all_ecgs.merge(tmp_covariates[['sample_id', 'value']], left_on=['sample_id'], right_on=['sample_id'])
    all_ecgs[pheno_dic[pheno][0]] = all_ecgs['value']
    all_ecgs = all_ecgs.drop(columns=['value'])
    print(pheno_dic[pheno][0], len(tmp_covariates), len(all_ecgs))

# %%
def gfr(x):
    k = 0.7 + 0.2*x['sex']
    alpha = -0.329 - 0.082*x['sex']
    f1 = x['creatinine']/88.4/k
    f1[f1>1.0] = 1.0
    f2 = x['creatinine']/88.4/k
    f2[f2<1.0] = 1.0
    
    gfr = 141.0 * f1**alpha \
          *f2**(-1.209) \
          *0.993**x['age'] \
          *(1.0+0.018*(1.0-x['sex']))
        
    return gfr
all_ecgs['gfr'] = gfr(all_ecgs)   

# %%
%matplotlib inline
import matplotlib.pyplot as plt
f, ax = plt.subplots()
f.set_size_inches(16, 9)
all_ecgs.hist(ax=ax)
plt.tight_layout()

# %%
disease_list = [
    ['Atrial_fibrillation_or_flutter_v2', 'atrial_fibrillation'],
    ['Bradyarrhythmia_general_inclusive_definition', 'bradyarrhythmia'],
    ['Cardiac_surgery', 'cardiac surgery'],
    ['Congenital_heart_disease', 'congenital heart disease'],
    ['Coronary_Artery_Disease_SOFT', 'coronary heart disease'],
    ['DCM_I42', 'dilated cardiomyopathy'],
    ['Diabetes_Type_1', 'diabetes type 1'],
    ['Diabetes_Type_2', 'diabetes type 2'],
    ['Heart_Failure_V2', 'heart failure'],
    ['Hypertension', 'hypertension'],
    ['Myocardial_Infarction', 'myocardial infarction'],
    ['Peripheral_vascular_disease', 'peripheral vascular disease'],
    ['Pulmonary_Hypertension', 'pulmonary hypertension'],
    ['Sarcoidosis', 'sarcoidosis'],
    ['Stroke', 'stroke'],
    ['Supraventricular_arrhythmia_General_inclusive_definition', 'supraventricular arrhythmia'],
    ['Venous_thromboembolism', 'venous thromboembolism'],
    ['Chronic_kidney_disease', 'chronic kidney disease'],
    ['composite_af_chf_dcm_death', 'AF+CHF+DCM+death'],
    ['composite_cad_dcm_hcm_hf_mi', 'CAD+DCM+HCM+HF+MI'],
    ['composite_chf_dcm_death', 'CHF+DCM+death'],
    ['composite_mi_cad_stroke', 'MI+CAD+stroke'],
    ['composite_mi_cad_stroke_death', 'MI+CAD+stroke+death'],
    ['composite_mi_cad_stroke_hf', 'MI+CAD+stroke+HF'],
    ['composite_mi_death', 'MI+death']]

# disease_list = [['Heart_Failure_V2', 'heart failure'], ['Myocardial_Infarction', 'myocardial infarction'],
#                 ['Atrial_fibrillation_or_flutter_v2', 'atrial fibrillation'], ['Diabetes_Type_2', 'type 2 diabetes'],
#                 ['Stroke', 'stroke'], ['Coronary_Artery_Disease_SOFT', 'coronary artery disease'],
#                 ['Hypertension', 'hypertension'], ['Pulmonary_Hypertension', 'pulmonary hypertension'],
#                 ['Peripheral_vascular_disease', 'peripheral vascular disease']
#                ]

# %%
diseases_unpack = pd.DataFrame()
diseases_unpack['sample_id'] = np.unique(np.hstack([diseases['sample_id'], all_ecgs['sample_id']]))
for disease, disease_label in disease_list:
    tmp_diseases = diseases[(diseases['disease']==disease) &\
                            (diseases['incident_disease'] > 0.5)]
    tmp_diseases_unpack = diseases_unpack.merge(tmp_diseases[['sample_id', 'incident_disease', 'censor_date']], how='left', on='sample_id')
    diseases_unpack[f'{disease}_incident'] = tmp_diseases_unpack['incident_disease']
    diseases_unpack[f'{disease}_censor_date'] = tmp_diseases_unpack['censor_date']
    tmp_diseases = diseases[(diseases['disease']==disease) &\
                             (diseases['prevalent_disease'] > 0.5)]
    tmp_diseases_unpack = diseases_unpack.merge(tmp_diseases[['sample_id', 'prevalent_disease', 'censor_date']], how='left', on='sample_id')
    diseases_unpack[f'{disease}_prevalent'] = tmp_diseases_unpack['prevalent_disease']
    diseases_unpack[f'{disease}_censor_date'] = tmp_diseases_unpack['censor_date']
    diseases_unpack.loc[diseases_unpack[f'{disease}_censor_date'].isna(), f'{disease}_censor_date'] = pd.to_datetime('2017-03-31')
#     tmp_diseases = diseases[(diseases['disease']==disease) &\
#                             (diseases['has_disease'] < 0.5)]
#     tmp_diseases_unpack = diseases_unpack.merge(tmp_diseases[['sample_id', 'prevalent_disease', 'censor_date']], how='left', on='sample_id')
#     diseases_unpack[f'{disease}_censor_date'] = tmp_diseases_unpack['censor_date']

# %%
diseases_unpack = diseases_unpack.fillna(0)
all_ecgs = all_ecgs.fillna(0)
all_ecgs.loc[(all_ecgs['smoking'] > 0.5), 'smoking'] = 1.0
all_ecgs.loc[(all_ecgs['smoking'] <= 0.5), 'smoking'] = 0.0

# %%
label_dic = {
    '50_hrr_actual': ['HRR50', 'beats'],
    'pretest_baseline_model_50_hrr_predicted': ['HRR50-restHR', 'beats'],
    'pretest_model_50_hrr_predicted': ['HRR50-pretest', 'beats'],
    'pretest_hr_achieved_model_50_hrr_predicted': ['HRR50-pretest-maxHR', 'beats'],
    '50_hr_actual': ['HR50', 'beats'],
    'pretest_baseline_model_50_hr_predicted': ['HR50-restHR', 'beats'],
    'pretest_model_50_hr_predicted': ['HR50-pretest', 'beats'],
    'pretest_hr_achieved_model_50_hr_predicted': ['HRR50-pretest-maxHR', 'beats'],
    '0_hr_actual': ['HR0', 'beats'],
    'pretest_baseline_model_0_hr_predicted': ['HR0-restHR', 'beats'],
    'pretest_model_0_hr_predicted': ['HR0-pretest', 'beats'],
    'pretest_hr_achieved_model_0_hr_predicted': ['HR0-pretest-maxHR', 'beats'],
    'bmi': ['BMI', 'units'],
    'age': ['Age', 'yrs'],
    'sex': ['Male', ''],
    'cholesterol': ['Cholesterol', 'mmol/L'],
    'HDL': ['HDL', 'mmol/L'],
    'smoking': ['Smoking', ''],
    'diastolic_bp': ['Diastolic blood pressure', 'mmHg'],
    'systolic_bp': ['Systolic blood pressure', 'mmHg'],   
}

# %%
# scaled
import statsmodels.api as sm
or_dic = {}
for pheno in all_ecgs:
    if pheno in ['instance0_date', 'sample_id']: 
        continue
    or_dic[pheno] = {}
    tmp_pheno = all_ecgs[['sample_id', pheno]]
    for disease, disease_label in disease_list:
        for occ in ['incident', 'prevalent']:
            or_dic[pheno][f'{disease}_{occ}'] = {}
            tmp_data = tmp_pheno.merge(diseases_unpack[['sample_id', f'{disease}_{occ}']], left_on='sample_id', right_on='sample_id')
            if pheno not in ['sex', 'smoking']:
                std = np.std(tmp_data[pheno].values)
                tmp_data[pheno] = (tmp_data[pheno].values - np.mean(tmp_data[pheno].values))/std
                std = ", %.1f" % std
            else:
                std = ''
            tmp_data['intercept'] = 1.0
            res = sm.Logit(tmp_data[f'{disease}_{occ}'], tmp_data[[pheno, 'intercept']]).fit()
            or_dic[pheno][f'{disease}_{occ}']['OR'] = np.exp(res.params[0])
            or_dic[pheno][f'{disease}_{occ}']['CI'] = np.exp(res.conf_int().values[0])
            or_dic[pheno][f'{disease}_{occ}']['p'] = res.pvalues[pheno]
            or_dic[pheno][f'{disease}_{occ}']['n'] = np.sum(tmp_data[f'{disease}_{occ}'])
            or_dic[pheno][f'{disease}_{occ}']['std'] = std

# %%
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
dis_plot_list = disease_list
phenos = or_dic['50_hrr_actual'].keys()
for dis, dis_label in disease_list:
    for occ in ['incident', 'prevalent']: 
        ors = []
        cis_minus = []
        cis_plus = []
        labels = []
        for pheno in or_dic:
            if 'genetic' in pheno: 
                continue
            if 'hrr' in pheno:
                scale = 1.0
            elif 'age' in pheno:
                scale = 1.0
            else:
                scale = 1.0
            ors.append(np.exp(np.log(or_dic[pheno][f'{dis}_{occ}']['OR'])*scale))
            cis_minus.append(np.exp(np.log(or_dic[pheno][f'{dis}_{occ}']['OR']-or_dic[pheno][f'{dis}_{occ}']['CI'][0])*scale))
            cis_plus.append(np.exp(np.log(or_dic[pheno][f'{dis}_{occ}']['CI'][1]-or_dic[pheno][f'{dis}_{occ}']['OR'])*scale))      
            labels.append(f'{label_dic[pheno][0]}{or_dic[pheno][dis+"_"+occ]["std"]} {label_dic[pheno][1]}')
            if or_dic[pheno][f'{dis}_{occ}']['p'] < (0.05/len(or_dic)):
                labels[-1] += '*'
        f, ax = plt.subplots()
        f.set_size_inches(6, 3.5)
        ax.errorbar(ors, np.arange(len(ors)), xerr=(cis_minus, cis_plus), marker='o', linestyle='', color='black')  
        ax.plot([1.0, 1.0], [-1.0, len(ors)], 'k--')
        ax.set_yticks(np.arange(len(ors)))
        ax.set_yticklabels(labels)
        ax.set_xscale('log', basex=np.exp(1))   
        ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
        ax.set_xticklabels(map(str, [0.25, 0.5, 1.0, 2.0, 4.0]))
#         locmin = matplotlib.ticker.LogLocator(base=np.exp(1),subs=(0.3,0.35,0.4,0.45,
#                                                                0.6,0.7,0.8,0.9,
#                                                                1.2,1.4,1.6,1.8,
#                                                                2.4,2.8,3.2,3.6))
#         ax.xaxis.set_minor_locator(locmin)
#         ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        #ax.set_xticks(np.logspace(-0.1, 8, 6, base=np.exp(1)))
        #ax.set_xticklabels(["%.2f" %d for d in np.exp(np.arange(-0.5, 2.1, 0.5))])
        ax.set_xlabel('Odds ratio (per 1-SD increase)')
        ax.set_title(f'{occ} {dis_label}\n n$_+$ = {int(or_dic[pheno][dis+"_"+occ]["n"])} / {len(all_ecgs)}')
        ax.set_ylim([-1.0, len(ors)])
        ax.set_xlim([0.25, 4.0])
        plt.tight_layout()
        f.savefig(f'{dis}_{occ}_or_hrronly.png', dpi=500)

# %%
# scaled
import statsmodels.api as sm
hr_dic = {}
for pheno in all_ecgs:
    if pheno in ['sample_id', 'instance0_date']: 
        continue
    hr_dic[pheno] = {}
    tmp_pheno = all_ecgs[['sample_id', pheno, 'instance0_date']]
    for disease, disease_label in disease_list:
        for occ in ['incident']:
            hr_dic[pheno][f'{disease}_{occ}'] = {}
            tmp_data = tmp_pheno.merge(diseases_unpack[['sample_id', f'{disease}_{occ}', f'{disease}_censor_date']], left_on='FID', right_on='sample_id')
            if pheno not in ['bmi', 'sex', 'smoking']:
                std = np.std(tmp_data[pheno].values)
                tmp_data[pheno] = (tmp_data[pheno].values - np.mean(tmp_data[pheno].values))/std
                std = ", %.1f" % std
            else:
                std = ''
            tmp_data['futime'] = (tmp_data[f'{disease}_censor_date']-tmp_data['instance0_date']).dt.days
            tmp_data['entry'] = 0.0
            tmp_data['intercept'] = 1.0
            tmp_data = tmp_data[tmp_data['futime']>0]
            res = sm.PHReg(tmp_data['futime'], tmp_data[pheno], 
                           tmp_data[f'{disease}_{occ}'], tmp_data['entry']).fit()
            hr_dic[pheno][f'{disease}_{occ}']['HR'] = np.exp(res.params[0])
            hr_dic[pheno][f'{disease}_{occ}']['CI'] = np.exp(res.conf_int()[0])
            hr_dic[pheno][f'{disease}_{occ}']['p'] = res.pvalues[0]
            hr_dic[pheno][f'{disease}_{occ}']['n'] = np.sum(tmp_data[f'{disease}_{occ}'])
            hr_dic[pheno][f'{disease}_{occ}']['std'] = std