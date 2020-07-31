import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

def unpack_disease(diseases, disease_list, phenotypes):

    diseases_unpack = pd.DataFrame()
    diseases_unpack['sample_id'] = np.unique(np.hstack([diseases['sample_id'], phenotypes['sample_id']]))
    for disease, disease_label in disease_list:
        tmp_diseases = diseases[(diseases['disease']==disease) &\
                                (diseases['incident_disease'] > 0.5)]
        tmp_diseases_unpack = diseases_unpack.merge(tmp_diseases[['sample_id', 'incident_disease', 'censor_date']], how='left', on='sample_id')
        diseases_unpack[f'{disease}_incident'] = tmp_diseases_unpack['incident_disease']
        incident_cases = np.logical_not(diseases_unpack[f'{disease}_incident'].isna())
        diseases_unpack.loc[incident_cases, f'{disease}_censor_date'] = tmp_diseases_unpack.loc[incident_cases, 'censor_date']
        tmp_diseases = diseases[(diseases['disease']==disease) &\
                                (diseases['prevalent_disease'] > 0.5)]
        tmp_diseases_unpack = diseases_unpack.merge(tmp_diseases[['sample_id', 'prevalent_disease', 'censor_date']], how='left', on='sample_id')
        diseases_unpack[f'{disease}_prevalent'] = tmp_diseases_unpack['prevalent_disease']
        prevalent_cases = np.logical_not(diseases_unpack[f'{disease}_prevalent'].isna())
        diseases_unpack.loc[prevalent_cases, f'{disease}_censor_date'] = tmp_diseases_unpack.loc[prevalent_cases, 'censor_date']
        diseases_unpack.loc[diseases_unpack[f'{disease}_censor_date'].isna(), f'{disease}_censor_date'] = pd.to_datetime('2020-03-31')

    # If NaN, disease is absent
    diseases_unpack = diseases_unpack.fillna(0)
    return diseases_unpack

def odds_ratios(phenotypes, diseases_unpack, labels, disease_labels,
                covariates, instance, dont_scale):
    or_multi_dic = {}
    for pheno in labels:
        if pheno in covariates:
            continue
        or_multi_dic[pheno] = {}
        tmp_pheno = phenotypes[['sample_id', pheno, f'instance{instance}_date'] + covariates]
        tmp_pheno[f'instance{instance}_date'] = pd.to_datetime(tmp_pheno[f'instance{instance}_date'])
        for disease, disease_label in disease_labels:
            or_multi_dic[pheno][f'{disease}_prevalent'] = {}
            tmp_data = tmp_pheno.merge(diseases_unpack[['sample_id', f'{disease}_censor_date', f'{disease}_prevalent', f'{disease}_incident']], on='sample_id')
            tmp_data[f'{disease}_prevalent'] = (tmp_data[f'instance{instance}_date'] >= tmp_data[f'{disease}_censor_date']).apply(float)
            if pheno in dont_scale:
                std = ''
            else:
                std = np.std(tmp_data[pheno].values)
                tmp_data[pheno] = (tmp_data[pheno].values - np.mean(tmp_data[pheno].values))/std
                std = f', {std:.1f}'
            covariates_scale = [covariate for covariate in covariates if covariate not in dont_scale]    
            tmp_data[covariates_scale] = (tmp_data[covariates_scale].values \
                                          - np.mean(tmp_data[covariates_scale].values, axis=0))/\
                                          np.std(tmp_data[covariates_scale].values, axis=0)
            tmp_data['intercept'] = 1.0
            res = sm.Logit(tmp_data[f'{disease}_prevalent'], tmp_data[[pheno, 'intercept']+covariates]).fit(disp=False)
            res_predict = res.predict(tmp_data[[pheno, 'intercept']+covariates])
            or_multi_dic[pheno][f'{disease}_prevalent']['OR'] = np.exp(res.params[0])
            or_multi_dic[pheno][f'{disease}_prevalent']['CI'] = np.exp(res.conf_int().values[0])
            or_multi_dic[pheno][f'{disease}_prevalent']['p'] = res.pvalues[pheno]
            or_multi_dic[pheno][f'{disease}_prevalent']['n'] = np.sum(tmp_data[f'{disease}_prevalent'])
            or_multi_dic[pheno][f'{disease}_prevalent']['ntot'] = len(tmp_data)
            or_multi_dic[pheno][f'{disease}_prevalent']['std'] = std
            or_multi_dic[pheno][f'{disease}_prevalent']['auc'] = roc_auc_score(tmp_data[f'{disease}_prevalent'], res_predict)
    return or_multi_dic


def hazard_ratios(phenotypes, diseases_unpack, labels, disease_labels,
                  covariates, instance, dont_scale):
    hr_multi_dic = {}
    for pheno in labels:
        if pheno in covariates:
            continue
        hr_multi_dic[pheno] = {}
        tmp_pheno = phenotypes[['sample_id', pheno, 'instance0_date'] + covariates]
        tmp_pheno[f'instance{instance}_date'] = pd.to_datetime(tmp_pheno[f'instance{instance}_date'])
        for disease, disease_label in disease_labels:
            hr_multi_dic[pheno][f'{disease}_incident'] = {}
            tmp_data = tmp_pheno.merge(diseases_unpack[['sample_id', f'{disease}_censor_date', f'{disease}_prevalent', f'{disease}_incident']], on='sample_id')
            tmp_data[f'{disease}_prevalent'] = (tmp_data[f'instance{instance}_date'] >= tmp_data[f'{disease}_censor_date']).apply(float)
            tmp_data[f'{disease}_incident'] = (np.logical_and((tmp_data[f'instance{instance}_date'] < tmp_data[f'{disease}_censor_date']), 
                                                            (tmp_data[f'{disease}_censor_date'] < pd.to_datetime('2020-03-31')))).apply(float)
            tmp_data = tmp_data[tmp_data[f'{disease}_prevalent']<0.5]

            if pheno in dont_scale:
                std = ''
            else:
                std = np.std(tmp_data[pheno].values)
                tmp_data[pheno] = (tmp_data[pheno].values - np.mean(tmp_data[pheno].values))/std
                std = f', {std:.1f}'
            covariates_scale = [covariate for covariate in covariates if covariate not in dont_scale]    
            tmp_data[covariates_scale] = (tmp_data[covariates_scale].values \
                                         - np.mean(tmp_data[covariates_scale].values, axis=0))/\
                                         np.std(tmp_data[covariates_scale].values, axis=0)
            tmp_data['intercept'] = 1.0
            tmp_data['futime'] = (tmp_data[f'{disease}_censor_date']-tmp_data['instance0_date']).dt.days
            tmp_data['entry'] = 0.0
            tmp_data = tmp_data[tmp_data['futime']>0]  
            res = sm.PHReg(tmp_data['futime'], tmp_data[[pheno]+covariates], 
                           tmp_data[f'{disease}_incident'], tmp_data['entry']).fit()
            res_predict = res.predict()
            hr_multi_dic[pheno][f'{disease}_incident']['HR'] = np.exp(res.params[0])
            hr_multi_dic[pheno][f'{disease}_incident']['CI'] = np.exp(res.conf_int()[0])
            hr_multi_dic[pheno][f'{disease}_incident']['p'] = res.pvalues[0]
            hr_multi_dic[pheno][f'{disease}_incident']['n'] = np.sum(tmp_data[f'{disease}_incident'])
            hr_multi_dic[pheno][f'{disease}_incident']['ntot'] = len(tmp_data)
            hr_multi_dic[pheno][f'{disease}_incident']['std'] = std
            hr_multi_dic[pheno][f'{disease}_incident']['auc'] = roc_auc_score(tmp_data[[f'{disease}_incident']], res_predict.predicted_values)
    return hr_multi_dic


def plot_or_hr(or_dic, label_dic, disease_list, suffix, occ='prevalent'):
    ratio_type = 'OR' if 'prevalent' in occ else 'HR'
    ratio_label = 'Odds' if 'prevalent' in occ else 'Hazard'
    for dis, dis_label in disease_list:
        ors = []
        cis_minus = []
        cis_plus = []
        labels = []
        for pheno in or_dic:
            ors.append(np.exp(np.log(or_dic[pheno][f'{dis}_{occ}'][ratio_type])))
            cis_minus.append(np.exp(np.log(or_dic[pheno][f'{dis}_{occ}'][ratio_type]-or_dic[pheno][f'{dis}_{occ}']['CI'][0])))
            cis_plus.append(np.exp(np.log(or_dic[pheno][f'{dis}_{occ}']['CI'][1]-or_dic[pheno][f'{dis}_{occ}'][ratio_type])))
            label = f'{label_dic[pheno][0]}{or_dic[pheno][dis+"_"+occ]["std"]} {label_dic[pheno][1]}'
            if 'ent' in occ:
                label += f'  {or_dic[pheno][dis+"_"+occ]["auc"]:.2f}'
            labels.append(label)

            if or_dic[pheno][f'{dis}_{occ}']['p'] < 0.05:
                labels[-1] += '*'

        f, ax = plt.subplots()
        f.set_size_inches(6, 4)
        ax.errorbar(ors, np.arange(len(ors)), xerr=(cis_minus, cis_plus), marker='o', linestyle='', color='black')  
        ax.plot([1.0, 1.0], [-1.0, len(ors)], 'k--')
        ax.set_yticks(np.arange(len(ors)))
        ax.set_yticklabels(labels)
        ax.set_xscale('log', basex=np.exp(1))
        ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
        ax.set_xticklabels(map(str, [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]))
        ax.set_xlabel(f'{ratio_label} ratio (per 1-SD increase)')
        ax.set_title(f'{occ} {dis_label}\n n$_+$ = {int(or_dic[pheno][dis+"_"+occ]["n"])} / {int(or_dic[pheno][dis+"_"+occ]["ntot"])}')
        ax.set_ylim([-1.0, len(ors)])
        ax.set_xlim([min(0.5, min(ors)*0.667), max(2.0, max(ors)*1.5)])
        plt.tight_layout()
        f.savefig(f'{dis}_{occ}_{suffix}.png', dpi=500)
