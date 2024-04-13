# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:26:41 2024

@author: hbehrooz
"""
from sklearn.metrics import silhouette_samples, silhouette_score 

from sklearn.manifold import TSNE
from scipy.stats import kendalltau
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KernelDensity
import seaborn as sns
from tslearn.clustering import TimeSeriesKMeans, silhouette_score, KernelKMeans
from tslearn.metrics import dtw as ts_dtw
from tslearn.utils import to_time_series_dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from pathlib import Path
#import sys
import zipfile
#import re
from sklearn import cluster
from matplotlib import pyplot
from sklearn.preprocessing import OrdinalEncoder
import pandas_profiling as pp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from random import randint
from scipy.stats import kendalltau


def optimal_number_of_clusters(wcss):
    """
    The function take a list of wcss from kmean various # of cluster from 2 to max value 
    and find the point on the curve with the maximum distance to the line between
    first and last point of the curve as best # of cluster

    Parameters
    ----------
    wcss : List of the wcss value for the various kmean # of clusters  

    Returns
    -------
    int
        The optimal # of cluster .

    """
#    coordination of the line between the first and last wcss points
    x1, y1 = 2, wcss[0]
    x2, y2 = len(wcss)+2, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = ((y2 - y1)**2 + (x2 - x1)**2)**.5
        distances.append(numerator/denominator)

    return distances.index(max(distances)) + 2


def consolidate_NBI(source_dir,output_file='NBI92_22.csv'):
    """
    Read the indivisual zip file for each year NBI data and save them in a 
    output CSV file

    Parameters
    ----------
    source_dir : string
        the source directory contians the indivisua NBI files for each year.
    output_file : string, optional
        The ouput files for consolidated NBI file in csv format. The default is 'NBI92_22.csv'.

    Returns
    -------
    The consolidated dataframe.

    """    
    zfiles = list(source_dir.iterdir())
    first=1
    year=[]
    for zfile in zfiles:
        yy=int(zfile.stem[:4])
        print('read:%s'%zfile)
        zip1 = zipfile.ZipFile(zfile)
        if (zip1.namelist()[0].endswith('txt')):  fl=zip1.namelist()[0]
        else: fl = zip1.namelist()[1]
    
        df = pd.read_csv(zip1.open(fl),low_memory=False, encoding='ISO-8859-1')
        year=year +[yy]*len(df)
        if(first):
            first=0
            dff=df.copy()
        else:
            dff=pd.concat([dff, df], ignore_index=True)
    dff['Year']=year
    dff.to_csv(output_file)
    return(dff)

def read_states(file_path,state_list,chunksize = 100000):
    """
    Read specific states that their codes are in state_list and return as 
    a dataframe
    Parameters
    ----------
    file_path : string
        file path of the original NBI records
    state_list : LIST
        list of the numnerical code of the states that needed to read out.
    chunksize : numerical, optional
        The chunk size of reading portion from CSV file. The default is 100000.

    Returns
    -------
    dataframe contains the states records.

    """    
    i = 0
    len_r = 0
    first = 1
    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
        mask =chunk['STATE_CODE_001'].isin(state_list)
        #(chunk['STATE_CODE_001'] == 34)|(chunk['STATE_CODE_001']==36)|(chunk['STATE_CODE_001']==9)
        filtered_chunk = chunk[mask]
        len_r += len(filtered_chunk)
        if(first):
            df = filtered_chunk.copy()
            first = 0
        else:
            df = pd.concat([filtered_chunk, df], ignore_index=True)
    
        i += 1
        if(i % 10 == 0):
            print("%d/%d read" % (i*chunksize, len_r))
    return(df)

def fill_missing_years(df,bridge,current_year,minimum_exisiting_years=3):
    """
    filling missing years in between bridge inspection record

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    bridge : TYPE
        DESCRIPTION.
    current_year : TYPE
        DESCRIPTION.
    minimum_exisiting_years : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    None.

    """
    mask = df['STRUCTURE_NUMBER_008'] == bridge
    av_years = df.loc[mask, 'Year'].values
    missing_yeas = np.array([~(a == av_years).any() for a in range(
        int(av_years.min()), int(current_year+1))]).sum()
    if missing_yeas>0:
        first_year_inspected = int(np.min(av_years))
        # drop bridges with less than a tershold of avaialbe years data
        if len(av_years) < minimum_exisiting_years:
            df = df.drop(index=df.loc[mask].index)
        else:
            #tmp = pd.Series([y if (y==av_years).any() else np.nan for y in range(first_year_inspected,current_year+1)])
            # range(np.min(av_years),np.max(av_years)+1):
            for y in range(first_year_inspected, current_year+1):
                if y not in av_years:
                    nearest_y = av_years[np.sum(y >= av_years)-1]
                    sample = df.loc[(mask) & (df['Year'] == nearest_y)].squeeze()
                    sample['IMP_LEN_MT_076'] = np.nan
                    sample['min_cond'] = np.nan
                    sample["Year"] = y
                    sample["ADT_029"] = np.nan
                    df.loc[len(df)] = sample
                    mask = df['STRUCTURE_NUMBER_008'] == bridge   

        
    
    
    
def preprocess(df,NF,NNF,Cond):
    """
    This function take df dataframe as input and remove invalid catagories from fields
    and also fill numnerica field with valid values. It also add a field 'min_cond' a minum value
    between the COnd fields. finally it fills missing years record for the bridges in between the firs year of 
    their appearance and current year.

    Parameters
    ----------
    df : DataFrame
        the original NBI dataframe .
    NF : TYPE
        Numerical fields of df.
    NNF : TYPE
        Non Numericla Fields of df.
    Cond : TYPE
        Condition rate fields.

    Returns
    -------
    None.

    """
    # df = df.drop(columns=list(set(df.columns)-set(NF+NNF +
    #              ['min_cond', 'STRUCTURE_NUMBER_008', 'Year'])))
    #fill nan values with valid nearst values 
    df['FUNCTIONAL_CLASS_026'] = pd.to_numeric(
        df['FUNCTIONAL_CLASS_026'], errors='coerce')
    acc_values = [1, 2, 6, 7, 8, 9, 11, 12, 14, 16, 17, 19]  # page 13 of manual
    df.loc[~df['FUNCTIONAL_CLASS_026'].isin(
        acc_values), 'FUNCTIONAL_CLASS_026'] = np.nan
    
    df['DESIGN_LOAD_031'] = pd.to_numeric(df['DESIGN_LOAD_031'], errors='coerce')
    df.loc[(df['DESIGN_LOAD_031'] < 0) | (
        df['DESIGN_LOAD_031'] > 9), 'DESIGN_LOAD_031'] = np.nan    
    
    df['STRUCTURE_KIND_043A'] = pd.to_numeric(
        df['STRUCTURE_KIND_043A'], errors='coerce')
    df.loc[(df['STRUCTURE_KIND_043A'] < 0) | (
        df['STRUCTURE_KIND_043A'] > 9), 'STRUCTURE_KIND_043A'] = np.nan
    
    df['STRUCTURE_TYPE_043B'] = pd.to_numeric(
        df['STRUCTURE_TYPE_043B'], errors='coerce')
    df.loc[(df['STRUCTURE_TYPE_043B'] < 0) | (
        df['STRUCTURE_TYPE_043B'] > 22), 'STRUCTURE_TYPE_043B'] = np.nan
    
    df['APPR_KIND_044A'] = pd.to_numeric(df['APPR_KIND_044A'], errors='coerce')
    df.loc[(df['APPR_KIND_044A'] < 0) | (
        df['APPR_KIND_044A'] > 9), 'APPR_KIND_044A'] = np.nan
    
    df['APPR_TYPE_044B'] = pd.to_numeric(df['APPR_TYPE_044B'], errors='coerce')
    df.loc[(df['APPR_TYPE_044B'] < 0) | (
        df['APPR_TYPE_044B'] > 22), 'APPR_TYPE_044B'] = np.nan
    
    df['TRAFFIC_DIRECTION_102'] = pd.to_numeric(
        df['TRAFFIC_DIRECTION_102'], errors='coerce')
    df.loc[(df['TRAFFIC_DIRECTION_102'] < 0) | (
        df['TRAFFIC_DIRECTION_102'] > 3), 'TRAFFIC_DIRECTION_102'] = np.nan
    
    df['DECK_STRUCTURE_TYPE_107'] = pd.to_numeric(
        df['DECK_STRUCTURE_TYPE_107'], errors='coerce')
    df.loc[(df['DECK_STRUCTURE_TYPE_107'] < 0) | (df['DECK_STRUCTURE_TYPE_107'] > 9)
           | (df['DECK_STRUCTURE_TYPE_107'] == 'N')
           | (df['DECK_STRUCTURE_TYPE_107'] != df['DECK_STRUCTURE_TYPE_107']), 'DECK_STRUCTURE_TYPE_107'] = 0
    
    
    for cc in Cond:
        df[cc] = pd.to_numeric(df[cc], errors='coerce')
        df.loc[df[cc] == 'N', cc] = np.nan
    #creat a new filed min_cond as a minmum of three conditional fileds

    df['min_cond'] = df[Cond].apply(np.nanmin,axis=1) 

    bridges=np.unique(df['STRUCTURE_NUMBER_008'])
    current_year=int(df['Year'].max())
    #interpolate  condtions fields in both way to fill empty fields
    #except for IMP_LEN_MT_076 interpolate numenricla field with similar nearest value 
    i=0
    print(len(bridges),"Bridges will preprocess")
    print(             "----------------------------------")
    for b in bridges:
        i+=1
        if (i%1000)==0 :
            print("%d from %d processed"%(i,len(bridges)))
        fill_missing_years(df,b,current_year,minimum_exisiting_years=3)
        mask = df['STRUCTURE_NUMBER_008'] == b
        df.loc[mask,'min_cond']=df.loc[mask, 'min_cond'].interpolate(limit_direction='both').round()
        for nf in NF+NNF:
            if(nf != "IMP_LEN_MT_076"):
                idx = df.loc[mask, nf].first_valid_index()
                valid_value = df.loc[mask, nf][idx] if idx is not None else np.nan
                if (np.isnan(valid_value)):
                    df=df.loc[~mask]
                else:
                    df.loc[mask, nf] = df.loc[mask, nf].interpolate(
                        method='nearest', limit_direction='both')
                    df.loc[mask, nf] = df.loc[mask, nf].fillna(valid_value)                   
                    
       

    #min_cond lower than 4 should be set to 3
    df.loc[df['min_cond'] < 4, 'min_cond'] = 3
    df.loc[df['min_cond'] > 8, 'min_cond'] = 9
    df['min_cond'] =df[ 'min_cond'] -3 
    
    df['min_cond'] = df['min_cond'].astype('Int64')    
  
    
    # if the reconstruction date is missing or not valid set it to construction date
    mask = (df['YEAR_RECONSTRUCTED_106'] != df['YEAR_RECONSTRUCTED_106']) |(df['YEAR_RECONSTRUCTED_106']==0)
    df.loc[mask, 'YEAR_RECONSTRUCTED_106'] = df.loc[mask, 'YEAR_BUILT_027']
    
    # the implmentation len of improvment will set to zero in case not valid number presents
    df['IMP_LEN_MT_076']=pd.to_numeric(df['IMP_LEN_MT_076'], errors='coerce')
    df.loc[(df['IMP_LEN_MT_076'] != df['IMP_LEN_MT_076']), 'IMP_LEN_MT_076'] = 0
    
    df = df.drop(columns=list(set(df.columns)-set(NF+NNF +
                 ['min_cond', 'STRUCTURE_NUMBER_008', 'Year'])))
    df = df.dropna()
    df[NNF]=df[NNF].astype(int)
    df[list(set(NF+NNF+['min_cond'])-set(['STRUCTURE_LEN_MT_049','ROADWAY_WIDTH_MT_051']))]=\
        df[list(set(NF+NNF+['min_cond'])-set(['STRUCTURE_LEN_MT_049','ROADWAY_WIDTH_MT_051']))].astype(int)
    return(df)


def add_TIC(df):
    """
    Add Time In Condition field for each bridge definig the period that the bridge
    keep its current condition in the past. 
    It also add std and average of the ADT and IMP_LEN_MT fields during TIC 

    Parameters
    ----------
    df : input NBI dataframe
        DESCRIPTION.

    Returns
    -------
    df dataframe.

    """
    df['TIC'] = np.nan  # Time In Condition
    #   mean and std for  'IMP_LEN_MT_076','ADT_029',
    df['ADT_029_mean'] = np.nan
    df['ADT_029_std'] = np.nan
    df['IMP_LEN_MT_076_mean'] = np.nan
    df['IMP_LEN_MT_076_std'] = np.nan
    bridges=np.unique(df['STRUCTURE_NUMBER_008'])
    #df['TTC'] = np.nan  # TIme To Change
    df['next_cond'] = np.nan  # the following year Condition
    for b in range(len(bridges)):
        print(b, "/", len(bridges))
        df_b_ind = df.loc[df['STRUCTURE_NUMBER_008']
                          == bridges[b]].sort_values('Year').index
        min_year = df.loc[df_b_ind]
        adt_mean_l = np.zeros(len(df_b_ind))
        imp_mean_l = np.zeros(len(df_b_ind))
        adt_std_l = np.zeros(len(df_b_ind))
        imp_std_l = np.zeros(len(df_b_ind))
        next_cond_l = np.zeros(len(df_b_ind))
        tic_l = np.zeros(len(df_b_ind))
    #    ttc_l = np.zeros(len(df_b_ind))
        for k in range(0, len(df_b_ind)):
            cond_base = df.loc[df_b_ind[k],'min_cond']
            if k < len(df_b_ind)-1:
                next_cond_l[k] = df.loc[df_b_ind[k+1]]['min_cond']
            else:
                next_cond_l[k] = np.nan
            for j in range(k, -1, -1):
    
                if(df.loc[df_b_ind[j]]['min_cond'] == cond_base):
                    tic_l[k] += 1
                    adt_mean_l[k] = df.loc[df_b_ind[j:k+1], 'ADT_029'].mean()
                    imp_mean_l[k] = df.loc[df_b_ind[j:k+1], 'IMP_LEN_MT_076'].mean()
                    if(k != j):
                        adt_std_l[k] = df.loc[df_b_ind[j:k+1], 'ADT_029'].std()
                        imp_std_l[k] = df.loc[df_b_ind[j:k+1], 'IMP_LEN_MT_076'].std()
                else:
                    break
    #        ttc_l[k] = len(df_b_ind)-k-1
            for j in range(k+1, len(df_b_ind)):
                if(df.loc[df_b_ind[j]]['min_cond']) > cond_base:
    #                ttc_l[k] = np.nan
                    break
                elif((df.loc[df_b_ind[j]]['min_cond']) < cond_base):
    #                ttc_l[k] = j-k-1
                    break
        df.loc[df_b_ind, 'TIC'] = tic_l
        df.loc[df_b_ind, 'ADT_029_mean'] = adt_mean_l
        df.loc[df_b_ind, 'ADT_029_std'] = adt_std_l
        df.loc[df_b_ind, 'IMP_LEN_MT_076_mean'] = imp_mean_l
        df.loc[df_b_ind, 'IMP_LEN_MT_076_std'] = imp_std_l
        df.loc[df_b_ind, 'next_cond'] = next_cond_l
    return(df)


# Multivariate DTW distance cacluation

# conda install -c conda-forge tslearn


def multidimensional_dtw(matrix1, matrix2):
    """
    Perform multidimensional dynamic time warping (MDTW) on two matrices.

    Parameters:
    - matrix1: First matrix (mxn).
    - matrix2: Second matrix (qxn).
n is number of features
q,m are time step for two multidimentional time series 
    Returns:
    - dtw_distance: DTW distance between the two matrices.
    - path: DTW path.
    """

    # Ensure that the matrices have the same number of features (columns)
    assert matrix1.shape[1] == matrix2.shape[1], "Both matrices must have the same number of features."
    matrix2=matrix2[~np.isnan(matrix2).all(axis=1)]
    matrix1=matrix1[~np.isnan(matrix1).all(axis=1)]
    
    # Calculate the DTW distance and path using fastdtw
    # , dist=lambda x, y: np.linalg.norm(x - y))
    dtw_distance = ts_dtw(matrix1, matrix2)
    if(dtw_distance!=dtw_distance): print("nan found in DTW distance calculation")
    return dtw_distance


def xgb_ord(df,save="",lag_year=5,train_size=.75,random_state=None,use_clusters=False):
    """
    an ordinal XGBoost logistic cascaded model traind based on df dataframe data
    

    Parameters
    ----------
    df : DataFrame
        input dataset.
    save : TYPE, optional
        DESCRIPTION. The default is "".
    lag_year : int, optional
        the lag year for training model. The default is 5.
    train_size : float, optional
        training size. The default is .75.
    random_state : TYPE, optional
        random state. The default is None.
    use_clusters : Boolean, optional
        if true apply the statified sampling to sample from df for trainig based on the
        clusters. else randomly select samples. The default is False.

    Returns
    -------
    None.

    """
    xgb_params = {
        'objective': 'binary:logistic',  # Use logistic regression for binary classification
        'eval_metric': 'error',           # Use classification error as the evaluation metric
        # Add other parameters as needed
        # 'n_estimators' :200,
        # 'learning_rate': 0.1,
        # 'max_depth':10,
        
    }
    ordinal=['G3','G4','G5','G6','G7','G8']
    dff=df.copy()
    if(random_state==None):
        np.random.seed()
    else:
        np.random.seed(42)

        
    if(use_clusters):
        dff=startified_sampling(dff, percent=train_size,thershold=64)

    else:
        dff['selected']=np.random.choice(2,size=len(dff),p=[train_size,1-train_size])        
    #dff=df.copy()
    min_cond = int(df['next_cond'].min())
    max_cond = int(df['next_cond'].max())
    for j in range(int(max_cond-min_cond)):
        dff[ordinal[j]]=dff['next_cond']>j
    if(lag_year!=0):
        current_year=dff['Year'].max()
        # ignore the records contains nan which are the the most recent year of data field
        #and those have improved condition
        
        dff = dff[dff['min_cond'] >= dff['next_cond']].dropna(axis=0)
    
        dff_lag =dff[dff['Year']<=current_year-lag_year]   
    else:
        dff_lag= dff.dropna(axis=0)
    # dff=df.dropna(axis=0)
    #dff=dff[list(set(df.columns)-set(NNF+['STRUCTURE_NUMBER_008', 'Year']+normalizing_features+['next_cond','n_next_cond','cluster']))].copy()

    classifier = []
    accur_l = []
    f_import = []
    #disp_l=[]
    selected_cols=['n_ADT_029','n_ADT_029_mean','n_ADT_029_std','n_IMP_LEN_MT_076',\
                    'n_IMP_LEN_MT_076_mean','n_IMP_LEN_MT_076_std','n_min_cond','n_TIC']
    # selected_cols = list(set(df.columns)-set(NNF+['STRUCTURE_NUMBER_008', 'Year'] +
    #                       normalizing_features+['next_cond', 'n_next_cond', 'cluster']))

    mask=dff_lag['selected']==1
    x_train=np.zeros((len(dff_lag.loc[mask]), len(selected_cols)+1))
    x_train[:,0]=dff_lag.loc[mask].index
    x_train[:,1:]=dff_lag.loc[mask,selected_cols].values
    y_train=dff_lag.loc[mask,['min_cond', 'next_cond']+ordinal].astype(int).values
    
    mask=dff_lag['selected']==0
    x_test=np.zeros((len(dff_lag.loc[mask]), len(selected_cols)+1))
    x_test[:,0]=dff_lag.loc[mask].index
    x_test[:,1:]=dff_lag.loc[mask,selected_cols].values
    y_test=dff_lag.loc[mask,['min_cond', 'next_cond']+ordinal].astype(int).values
    y_pred_l = np.zeros((y_test.shape[0],max_cond-min_cond))

    for k in range(min_cond, max_cond):

        # Encode logestic target variable
        # Define and train the XGBoost classifier
        xgb_classifier = xgb.XGBClassifier(**xgb_params)
        xgb_classifier.fit(x_train[:,1:], y_train[:, k+2])
        y_pred = xgb_classifier.predict(x_test[:,1:])
        #accuracy = kendal(y_test[:, k+2], y_pred)
        log_acc=np.sum((y_test[:, k+2]== y_pred))/len(y_pred)
        print("accuracy for Condition>%d :" %
              (k+3), np.round(log_acc,2))
        classifier.append(xgb_classifier)
        y_pred_l[:,k]=y_pred
        accur_l.append(log_acc)
        f_import.append(xgb_classifier.feature_importances_)
    predict = np.sum(y_pred_l,axis=1)

    # importance_df = pd.DataFrame(
    #     data=np.array(f_import), columns=selected_cols)
    prd_df = pd.DataFrame(data={'min_cond': y_test[:, 0], 'next_cond': y_test[:, 1],
                                'diff': y_test[:, 0]-y_test[:, 1], 'predict': predict,
                                'predict_diff': y_test[:, 1]-predict})
    tau, p_value = kendalltau(prd_df['next_cond'],prd_df['predict'])
    #accuracy = kendal(prd_df['next_cond'].values,prd_df['predict'].values)#.round(2)
    print("kendall tau-b concordance evaluation for all prediction :",np.round(tau,2))
    cm = confusion_matrix(prd_df['next_cond'].values,prd_df['predict'].values)
    cm=(cm/cm.sum(axis=1)).round(2)

    return(cm,np.round(tau,2))

def startified_sampling(df,percent,thershold=0):
    """
    

    use startified sampling method for sampling form df
    ----------
    df : DataFrame
        input data set.
    percent : float
        percent of sampling form the dataset.
    thershold : TYPE, optional
        Define a thershold for minimum of sample size for each startum.
        The default is 0.

    Returns
    -------
    None.

    """
    cl,cnt=np.unique(df['cluster'],return_counts=True)
    dff=df.copy()
    dff['selected']=0
    for clust in cl:
        mask=(dff['cluster']==clust)
        cl_cnt=len(dff[mask])
        num_cl=int(cl_cnt*percent)
        if(num_cl<thershold) :
            num_cl=thershold if cl_cnt>thershold else cl_cnt
        ind=dff[mask].index
        arg_ind=np.random.choice(cl_cnt, size=num_cl, replace=False)
        
        dff.loc[ind[arg_ind],'selected']=1
    # dff=dff[dff['selected']]  
    # dff=dff.drop('selected',axis=1)
    return(dff)


def normalize_one_hot(df,normalizing_features,regression_features,NNF):
    """
    normalize (minmax) numenrcial fields and apply one hot encoder for catagorical 
    fields

    Parameters
    ----------
    df : Dataframe
        input dataset.
    normalizing_features : field list
        fields that should be normalized.
    regression_features : fields list
        fileds that should be used regression for miising values on.
    NNF : TYPE
        Catagorical fields that should be ne hot enocoder apply to them.

    Returns
    -------
    None.

    """

    
    regressed_fields = ['n_'+x for x in regression_features]
    df[regressed_fields] = (df[regression_features]-df[regression_features].min()) / \
        (df[regression_features].max()-df[regression_features].min())
    
    
    normalize_fields = ['n_'+x for x in normalizing_features]
    df[normalize_fields] = (df[normalizing_features]-df[normalizing_features].min()) / \
        (df[normalizing_features].max()-df[normalizing_features].min())
    
    # Standaradization
    # df[normalized_features]=(df[normalized_features]-df[normalized_features].mean())/df[normalized_features].std()
    
    # One hot encoding for non numenrical features
    for nnf in NNF:
    
        encoder = OneHotEncoder(sparse_output=False)
        output_df = encoder.fit_transform(df[nnf].to_numpy().reshape(-1, 1))
        # Create a Pandas DataFrame of the hot encoded column
        fn = encoder.get_feature_names_out()
        columns = [nnf+fn[i] for i in range(len(fn)-1)]
        ohe_df = pd.DataFrame(output_df[:, :-1], columns=columns)
        df[columns] = output_df[:, :-1]
        # df=df.drop(nnf,axis=1)
    return(df)
#%%
file_path = 'NBI92_22.csv'  # Path to your CSV file

source_dir = Path('c:/NBIdata/')
# %%creating a complete dataset form all state files excute only one time

#dff=consolidate_NBI(source_dir,file_path=file_path)


# df=dff[dff['STATE_CODE_001']==34].copy()
# nj=df
# nj.to_csv('nj.csv')
# %% Read specific states data only

#read NY NJ CT records
state_list=[36,34,9]
#state_list=[36]

df=read_states(file_path,state_list,chunksize = 100000)
# keep the highway records only RECORD_TYPE_005A should be equal to 1
# and  SERVICE_ON_042A also should be equal to 1
df['RECORD_TYPE_005A'] = pd.to_numeric(df['RECORD_TYPE_005A'], errors='coerce')
df['SERVICE_ON_042A'] = pd.to_numeric(df['SERVICE_ON_042A'], errors='coerce')

mask = (df['RECORD_TYPE_005A'] == 1) & (df['SERVICE_ON_042A'] == 1)
df = df[mask].copy()
# profile = pp.ProfileReport(df[NF+NNF+RAT+['STRUCTURE_NUMBER_008']],title='Pandas Profiling Report',explorative=True)
# profile.to_file("nj_profiling.html")
df_c=df.copy()
# %% initialize required variables
Numeric_f = "Year, YEAR_BUILT_027, ADT_029, YEAR_RECONSTRUCTED_106, MAIN_UNIT_SPANS_045,\
   STRUCTURE_LEN_MT_049, ROADWAY_WIDTH_MT_051, IMP_LEN_MT_076"

NF = [x.strip() for x in Numeric_f.split(',')]
Non_numeric_f = "FUNCTIONAL_CLASS_026, DESIGN_LOAD_031,\
    STRUCTURE_KIND_043A, STRUCTURE_TYPE_043B, APPR_KIND_044A, APPR_TYPE_044B,\
    TRAFFIC_DIRECTION_102, DECK_STRUCTURE_TYPE_107"
NNF = [x.strip() for x in Non_numeric_f.split(',')]
#bridge condition fields
Cond = ['DECK_COND_058', 'SUPERSTRUCTURE_COND_059', 'SUBSTRUCTURE_COND_060']
#numerical fields that should be normalized
normalizing_features = ['IMP_LEN_MT_076', 'ADT_029', 'MAIN_UNIT_SPANS_045', 'ROADWAY_WIDTH_MT_051',
                        'STRUCTURE_LEN_MT_049', 'YEAR_BUILT_027', 'YEAR_RECONSTRUCTED_106', 'min_cond',
                        'ADT_029_mean', 'ADT_029_std', 'IMP_LEN_MT_076_mean', 'IMP_LEN_MT_076_std','TIC']
#numenrcial fields that should be regressed if they are missing
regression_features = ['ADT_029_mean', 'ADT_029_std',
                       'IMP_LEN_MT_076_mean', 'IMP_LEN_MT_076_std'] 

#%%
col = df.columns
#sort records based on bridge structure number and the inspection year
df = df.sort_values(['STRUCTURE_NUMBER_008', 'Year'])
bridges = np.unique(df['STRUCTURE_NUMBER_008'])

#change to numercial values the numerical features and ignore invalid values
for nf in NF:
    df[nf] = pd.to_numeric(df[nf], errors='coerce')
    df.loc[(df[nf] < 0), nf] = np.nan




# %%profiling dataset for analysing features
# profile = pp.ProfileReport(df,title='Pandas Profiling Report',explorative=True)
# profile.to_file("nj_profiling_cleaned.html")


# %% add TIC field and mean and sd for numenrical varaible fields ADT and IMPLmentation length
df=preprocess(df,NF,NNF,Cond)

df.to_csv('NJ_NY_CT_v3.csv', index=False)
#add TIC field
df=add_TIC(df)
df.to_csv('NJ_NY_CT_add_TIC_v3.csv', index=False)
#df= pd.read_csv('NJ_NY_CT_add_TIC.csv',low_memory=False)
#%%
#the number of inspected year that a bridge should have before entering our process
acceptable_years=3
#normalize and one hot encoder apply to fields
df=normalize_one_hot(df,normalizing_features,regression_features,NNF)
#ignore bridges that has less than 3 years data
bridges, year_count = np.unique(df['STRUCTURE_NUMBER_008'],return_counts=True)
df=df.loc[df['STRUCTURE_NUMBER_008'].isin(bridges[year_count>acceptable_years])]
df.to_csv('NJ_NY_CT_add_TIC_norm_v3.csv', index=False)
#df= pd.read_csv('NJ_NY_CT_add_TIC_norm.csv',low_memory=False)


# %%preparing X matrix of features for DTW clustering
# X = []
# bridges=np.unique(df['STRUCTURE_NUMBER_008'])
regressed_fields = ['n_'+x for x in regression_features]
keeping_columns = list(set(df.columns)-set(NNF+['STRUCTURE_NUMBER_008', 'Year',  'n_next_cond', 'TIC',  'next_cond'] +
                                           normalizing_features+regression_features+regressed_fields))
# Create a list of all unique years in the dataset
# all_years = sorted(df['Year'].unique())

# # Create a list of all unique bridge IDs in the dataset
# all_bridges = df['STRUCTURE_NUMBER_008'].unique()

# # Create a DataFrame with all combinations of bridge IDs and years
# bridge_year_combinations = pd.MultiIndex.from_product([all_bridges, all_years], names=['STRUCTURE_NUMBER_008', 'Year'])
# all_data = pd.DataFrame(index=bridge_year_combinations).reset_index()

# # Merge the existing data with the complete set of bridge-year combinations
# df_filled = pd.merge(all_data, df, on=['STRUCTURE_NUMBER_008', 'Year'], how='left')

# # Sort the DataFrame by Bridge_ID and Year
# df_filled = df_filled.sort_values(by=['STRUCTURE_NUMBER_008', 'Year']).reset_index(drop=True)



#X=[subset for subset in  (group[1][keeping_columns].values for group in df.groupby('STRUCTURE_NUMBER_008'))]
# X=np.zeros((len(bridges),len(all_years),len(keeping_columns)))
# X[:,:,:]=list((group[1][keeping_columns].values for group in df_filled.groupby('STRUCTURE_NUMBER_008')))
#preper the fields as an matrix for being ready for DTW clustering
X=list((group[1][keeping_columns].values for group in df.groupby('STRUCTURE_NUMBER_008')))
bridges=list((group[0] for group in df.groupby('STRUCTURE_NUMBER_008')))

# itterate over each bridge
# for brg in bridges:  # bridges
#     X = X+[df.loc[df1['STRUCTURE_NUMBER_008'] == brg, keeping_columns].values]
X = to_time_series_dataset(X)



# %% examine diffrent number of cluster to find best # of cluster
max_cluster = 20
model_l = []
for num_cluster in range(2, max_cluster):
    km_dba = TimeSeriesKMeans(n_clusters=num_cluster, metric="dtw", max_iter=57,
                              max_iter_barycenter=5,
                              random_state=0, dtw_inertia=True, verbose=0, n_jobs=10)
    y = km_dba.fit(X)
    model_l.append(y)
    print("cluster,inertia", num_cluster, np.round(y.inertia_,2))

wcss = np.array([item.inertia_ for item in model_l])
max_cluster=num_cluster
fig = pyplot.figure()
pyplot.plot(list(range(2, max_cluster)), wcss)
pyplot.xticks(list(range(2, max_cluster)), list(range(2, max_cluster)))
pyplot.xlabel("Numbe of Cluster")
pyplot.ylabel("Average distance to centroids")
fig.figure.savefig('elbow6state_final.png')
# find the optimal # of cluster
optim_n_cluster = optimal_number_of_clusters(wcss)
print("optimal number of clusters is:", optim_n_cluster)
y_optimal = model_l[optim_n_cluster-2]
clusters = y_optimal.labels_  # 9 clusters
# it is a tslearn method to save the model
for k in range(len(model_l)):
    model_l[k].to_hdf5("output/%d_tslean-model_final.hdf5"%(k+2))

    
#y_optimal.to_hdf5("tslean-model3states-6_final.hdf5")

# %% optimal number of cluster was selected

#y_optimal = TimeSeriesKMeans.from_hdf5("output/8_tslean-model_final.hdf5")
clusters = y_optimal.labels_  # 8 clusters

# add cluster field to dataset
brdg_cluster = dict(zip(bridges, clusters))
df['cluster'] = df['STRUCTURE_NUMBER_008'].map(brdg_cluster)

df.to_csv('output/NJ_NY_CT_add_TIC_norm_clustered8_final.csv', index=False)
#df= pd.read_csv('output/NJ_NY_CT_add_TIC_norm_clustered8_final.csv',low_memory=False)
#%%evaluate OXGBoost performance with various portion of training data and not use clusteres
# as base for sampling
print("-------------------------")
print('OXGBoost performance')
print("-------------------------")
k_acc=[]
k_std=[]
number_of_boost_round=1
#rewind= 6 #rewindin in time for training
train_l=[.01,.05,.1,.2,.3,.30,.50]#,.65,.70,.75,.80,.85,.9,.95,.98,.99]
for s in train_l:
    acc_l=[]
    for kk in range(number_of_boost_round):

        _,k_acurarcy=xgb_ord(df,save='_',lag_year=0,train_size=s,random_state=42,use_clusters=False)
        acc_l.append(k_acurarcy)        
    k_acc.append(np.mean(acc_l))
    k_std.append(np.std(acc_l))
    print(">>>>>>with %%%d of trianing accuracy average=%%%d"%(int((s)*100), int(k_acc[-1]*100)))
    print("-------------------------")
xgb_accuracy=pd.DataFrame({'Train%':train_l,'Average accuracy':np.round(k_acc,2)})#,'std accuracy':np.round(k_std,2)})
#%%evaluate OXGBoost performance with various portion of training data and  use clusteres
# as base for sampling
print("-------------------------")
print('OXGBoost performance with cluster')
print("-------------------------")
k_acc_cl=[]
k_std_cl=[]
#rewind= 6 #rewindin in time for training
#train_l=[.01,.05,.1,.2,.25,.55,.6,.65,.70,.75,.80,.85,.9,.95,.98,.99]
for s in train_l:
    acc_l=[]
    for kk in range(number_of_boost_round):
        _,k_acurarcy=xgb_ord(df,save='_',lag_year=0,train_size=s,random_state=42,use_clusters=True)
        acc_l.append(k_acurarcy)
        
    k_acc_cl.append(np.mean(acc_l))
    k_std_cl.append(np.std(acc_l))
    print(">>>>>>with %%%d of trianing data, accuracy average=%%%d"%(int(s*100), int(k_acc_cl[-1]*100)))
    print("-------------------------")
xgb_accuracy=pd.DataFrame({'Train%':train_l,'Acc_non_clustered':np.round(k_acc,2),
                           'std_non_clustered':k_std,'Acc_clustered':np.round(k_acc_cl,2),
                           'std_clustere':k_std_cl})#,'std accuracy':np.round(k_std,2)})
xgb_accuracy.to_csv("output/accuracy_new.csv",index=False)
#%%

org_col = ['STRUCTURE_NUMBER_008', 'Year', 'ADT_029', 'FUNCTIONAL_CLASS_026',
           'YEAR_BUILT_027', 'DESIGN_LOAD_031', 'STRUCTURE_KIND_043A',
           'STRUCTURE_TYPE_043B', 'APPR_KIND_044A', 'APPR_TYPE_044B',
           'MAIN_UNIT_SPANS_045', 'STRUCTURE_LEN_MT_049',
           'ROADWAY_WIDTH_MT_051', 'IMP_LEN_MT_076', 'TRAFFIC_DIRECTION_102',
           'YEAR_RECONSTRUCTED_106', 'DECK_STRUCTURE_TYPE_107', 'min_cond', 'cluster']

# piv_df['cluster']=piv_df.index.map(brdg_cluster)

cluster_desc = df[org_col].groupby('cluster').describe()
piv_df_ref = df.pivot(index='STRUCTURE_NUMBER_008', columns='Year')
cond = piv_df_ref.loc[:, ('min_cond', slice(None))]
cond = cond.fillna(0).astype(int).astype(str)  # change NAN to zero
# cond=cond.astype('Int64').astype(str)
cond = cond.apply(''.join, axis=1)
years=np.unique(df['Year'])
max_year = years.max()-years.min()+1  # 1992 to 2022 31 years
cond_title = ['Condition=9', 'Condition=8', 'Condition=7',
              'Condition=6', 'Condition=5', 'Condition=4', 'Condition<=3']
distribution_df=pd.DataFrame(columns=['STRUCTURE_NUMBER_008','last_year','condition','TIC','cluster'])
# check the sequence of bridge condition and keep track of TIC
#cont_l = np.zeros((len(bridges), len(cond_title)))
for b in range(len(bridges)):
    last_val = -1
    len_val = 0
    for j in range(len(cond[bridges[b]])+1):
        if(j!=len(cond[bridges[b]])):
            k=cond[bridges[b]][j]
        else: k='-1'
        if(int(k) != 0):
            if(int(k) == last_val):
                len_val += 1
            else:
                ind = (9-last_val if last_val > 3 else 6)
                # longer length and digrade to lower condition
                if(int(k) < last_val):
                    distribution_df.loc[len(distribution_df)]={'STRUCTURE_NUMBER_008':bridges[b],
                        'last_year':years.min()+j,'condition':last_val,'TIC':len_val,
                        'cluster':brdg_cluster[bridges[b]]}
                    

                last_val = int(k)
                len_val = 1

br_cond_df=distribution_df.pivot_table(values='last_year',index='cluster',columns='condition', aggfunc='count', fill_value=0)

fig = br_cond_df.plot.bar(stacked=True)
fig.figure.savefig("count.png")
# visulize the length of the period that bridges keep certain condition
# https://seaborn.pydata.org/tutorial/distributions.html
    
    

#%%present the tic distribution for randomly vs established clusters
# distribution of TIC for various clusters both if randomly selected clusters and real clusters value
fig, ax = pyplot.subplots(2,2,figsize=(10, 15))
distribution_df['condition']=distribution_df['condition']+3
distribution_df_random=distribution_df.copy()
distribution_df_random['cluster']=np.random.randint(distribution_df_random['cluster'].max()+1
                                                    ,size=len(distribution_df_random['cluster']))

g = sns.FacetGrid(distribution_df, row="condition", col="cluster", margin_titles=True)
g.map(sns.kdeplot, "TIC")
g.savefig('cluster-condition-distribution.png')


#%%number of cluster and quantity of bridges in each cluster in each final year condition
cl_val, cluster_cnt = np.unique(clusters, return_counts=True)
cluster_cond_table=df.loc[df['Year']==df['Year'].max()].pivot_table(index='min_cond', columns='cluster',aggfunc='count', fill_value=0)['STRUCTURE_NUMBER_008']


TIC_dist1=df.loc[df['min_cond']>df['next_cond'],['min_cond','cluster','TIC']]
TIC_dist1['random_cluster']=np.random.randint(TIC_dist1['cluster'].max()+1
                                                    ,size=len(TIC_dist1['cluster']))
TIC_dist1['min_cond']+=3
TIC_dist1=TIC_dist1.rename(columns={'min_cond':'Condition'})
fig, ax = pyplot.subplots(6,2,figsize=(12, 6))
snsp=sns.displot(data=TIC_dist1, x='TIC',hue='cluster',col='min_cond', fill=False, kind='kde',palette=sns.color_palette(
          "tab10"))
snsp.figure.savefig("output/clusters_distribution.png")
snsp=sns.displot(data=TIC_dist1, x='TIC',hue='random_cluster',col='min_cond', fill=False, kind='kde',palette=sns.color_palette(
          "tab10"))
snsp.figure.savefig("output/random_clusters_distribution.png")

#here we create a comparison figure between random clusteres and predicted clusteres for TIC distribution
# to find out the goodness of fit of cluster prediction compare to random clusteres
pyplot.rcParams["font.family"] = "Times New Roman"
pyplot.rcParams["font.size"] =12
fig, ax = pyplot.subplots(6,2,figsize=(8, 16))


for con in range(4,10): 
    mask=TIC_dist1['Condition']==con
    snd1=sns.kdeplot(data=TIC_dist1[mask],x='TIC',hue='cluster',palette=sns.color_palette(
             "tab10"),ax=ax[con-4][0],legend=False )
    if(con!=9):snd1.set(xticklabels=[])
    #snd.set(yticklabels=[])
    snd1.set(ylabel="Condition=%d"%con)
    if(con!=9):snd1.set(xlabel=None)
    snd1.set(xlim=(-5,30),ylim=(0,.05))
    snd2=sns.kdeplot(data=TIC_dist1[mask],x='TIC',hue='random_cluster',palette=sns.color_palette(
             "tab10"),ax=ax[con-4][1],legend= False )
    if(con!=9):snd2.set(xticklabels=[])
    snd2.set(yticklabels=[])
    snd2.set(ylabel=None)
    if(con!=9):snd2.set(xlabel=None)
    snd2.set(xlim=(-5,30),ylim=(0,.05))
fig.subplots_adjust(wspace=0.1,hspace=.1)
fig.savefig("output/Comparison_random_clusters-8.png",dpi=300 )

    
bridges=list((group[0] for group in df.groupby('STRUCTURE_NUMBER_008')))






# %%        Calculate the dtw distance between bridges
dist_mat = np.zeros((len(bridges), len(bridges)))
for k in range(len(bridges)):
    for l in range(k+1, len(bridges)):
        mx1 = X[k, :, :]  # a matrix timesteps X features
        mx2 = X[l, :, :]  # a matrix timesteps X features
        dtw_distance = multidimensional_dtw(mx1, mx2)
        # print(k,l,"distance:",round(dtw_distance,2))
        dist_mat[l, k] = dtw_distance
        dist_mat[k, l] = dtw_distance
    print("Bridge:", k)

np.save('distance_matrix.npy', dist_mat)
# %%visulaizing the clustinring accuracy through t-SNE method , preplexity measure have important
#role in visulaization
#dist_mat=np.load('distance_matrix.npy')
tsne = TSNE(n_components=2, perplexity=300,
            metric="precomputed", init='random')
# mds = MDS(n_components=2, dissimilarity="precomputed")
# isomap = Isomap(n_components=2, metric="precomputed")
df_tsne = tsne.fit_transform(dist_mat)

fig = pyplot.figure(figsize=(8, 8))

ls = []
for l in np.unique(clusters):
    ix = np.where(clusters == l)
    ls = ls+[pyplot.scatter(df_tsne[ix, 0], df_tsne[ix, 1])]
pyplot.legend(ls, np.unique(clusters), title="Clusters")

# plt.scatter(X_pca[:,0],X_pca[:,1]);
pyplot.ylabel("First principal component")
pyplot.xlabel("Second principal component")
pyplot.title("t-SNE clustering : NBI dataset")
fig.savefig('output/t-SNE-8.png')
#%%  number of bridge in each (cluster - last codition) as a matrix#################################
distribution_mat=np.zeros((len(cond_title),len(np.unique(clusters))))
for b in bridges:
    mask=(df['STRUCTURE_NUMBER_008']==b)
    cl,con=df.loc[mask,['cluster','min_cond']].values[-1]
    distribution_mat[int(con),int(cl)]+=1
    
    
