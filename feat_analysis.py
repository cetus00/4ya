import numpy as np
import pandas as pd

import os
from joblib import Memory
from sklearn import metrics

from sklearn.feature_selection import mutual_info_classif

project_path = r"C:\Users\111\Documents\work\pyhton\\"
feat_table = pd.read_csv(project_path + "features_table.csv", sep = "\t")


cashe_dir = os.path.join(project_path,r"cache")
memory = Memory(cashe_dir,verbose=1)

def join_rare_features(feat_table,normalize=False):
    for i in range(21):
        feat_table["F"+str(i)] = feat_table[x for x in feat_table.columns if x.startswith("F"+str(i))]
    if "feat_categ" in feat_table.columns:
        feat_table = pd.concat((feat_table,pd.get_dummies(feat_table["feat_category"], prefix="categ_")), axis=1)
        del feat_table["feat_categ"]

    age = np.zeros_like(feat_table["reported_age"])
    age[np.isnan(feat_table["reported_age"])] = 1
    age = feat_table["reported_age"] + feat_table["predicted_age"]*age
    age[np.isnan(age)] = np.mean(age)
    feat_table["age"] = age

    del feat_table["reported_age"]
    
def regression_score(y_pred, y_test, treshold, use_mae = True):
    diff = np.abs(y_pred - y_test)
    confidence = diff < treshold
    score = np.count_nonzero(confidence)/diff.shape[0]
    if use_mae:
        error = metrics.r2_score(y_test, y_pred)
        score = score*(1+error)**3

    return score

def regres_score_roc(y_pred,y_test):
    accuracies = []
    tresholds = []
    for treshold in np.linspace(0,0.15,20):
        acc = regres_score(y_pred,y_test,treshold,True)
        accuracies.append(acc)
        tresholds.append(treshold)

    score  = np.mean(accuracies)
    return score

#function to count symmetrical conditional probability for ngrams (NB: ngram must be bigger than unigram)
def scp(ngram_range_tuple, ngram_table,column_position_dict, ngrams_positions, probability_table):
    ngram_names = np.array([k for k,v in sorted(ngrams_positions.items(), key=lambda a_entry: a_entry[1])])
    if ngram_range_tuple[1] ==1: # donʼt count scp for unigrams, i.e. given tuple is (x,1)
        return None
    elif ngram_range_tuple[0] > 1:
        min_ngram = ngram_range_tuple[0]
    elif ngram_range_tuple[0] ==1:
        min_ngram = ngram_range_tuple[0] + 1 # exclude unigrams from weighting with scp

    scp_values = np.ones((ngram_table.shape[1],))

    for n in range(min_ngram,ngram_range_tuple[1]+1):
        fst_column = column_position_dict[n-1] #position of the first column with n-gram in the table
        last_column = column_position_dict[n]
        extracted_columns = ngram_table[:,fst_column:last_column+1]

        for col in range(fst_column,last_column + 1):
            denominator = 0
            ngram = ngram_names[col]
            splt = ngram_names.split("||")

            for j in range(len(splt)-1):
                fst_part = "||".join(splt[: j+1])
                scnd_part = "||".join(splt[j+1:])

                fst_ngram_prob = probability_table[ngrams_positions[fst_part]]
                scnd_ngram_prob = probability_table[ngrams_positions[scnd_part]]

                denominator += fst_ngram_prob*scnd_ngram_prob

            denominator /= n-1
            scp_value = probability_table[col]**2
            scp_values[col] = scp_value

    return scp_values

def get_scp_table(prob_table, ngram_range_tuple, ngram_table, gram_frames, ngram_positions):
    scp_table = scp(ngram_range_tuple,ngram_table, gram_frames, ngram_positions, np.array(prob_table).reshape(-1))
    return scp_table

def mutual_info_classification(feat_table, target_vec):
    mi = mutual_info_classif(feat_table, target_vec)
    column_names = feat_table.columns
    mi_table = np.column_stack((column_names, mi))

    return mi_table

if __name__ == "__main__":
    proj_path = project_path
    all_counter = pd.read_excel(proj_path+"all_counter.xlsx")
    all_table = pd.read_csv(proj_path + "all_table", sep = "\t")
    mi_matrix = []

    scp_table = get_scp_table((1,4),all_table,all_counter.ngrams_edges__,all_counter.vocabulary_)

    scp_treshold = 0.004
    scp_filter = np.array(scp_table)>scp_treshold