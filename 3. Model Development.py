"""
@author: yilun
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle
from treeinterpreter import treeinterpreter as ti
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor


# plot ROC, KS
def modeloutcome(model,X_train,y_train,X_vali,y_vali):
        model.fit(X_train, y_train)
        #y_pred = model.predict(X_vali)
        probs = model.predict_proba(X_vali)
        preds = probs[:,1]
        fpr, tpr, threshold = metrics.roc_curve(y_vali, preds)
        roc_auc = metrics.auc(fpr, tpr)
        
        fig, axs = plt.subplots(nrows=2, ncols=2,figsize=(10,8))
    # ROC for validation
        axs[0, 1].title.set_text('Validation ROC and KS')
        axs[0, 1].plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
        axs[0, 1].legend(loc = 'lower right')
        axs[0, 1].plot([0, 1], [0, 1],'r--')
        axs[0, 1].set_ylabel('True Positive Rate')
        axs[0, 1].set_xlabel('False Positive Rate')
        axs[0, 1].set_xlim([0, 1])
        axs[0, 1].set_ylim([0, 1])  
    # KS for validation
        axs[1, 1].plot(1-threshold, tpr,label = 'KS = %0.3f' % max(tpr-fpr))
        axs[1, 1].plot(1-threshold, fpr)
        axs[1, 1].set_ylabel('TPR / FPR')
        axs[1, 1].set_xlabel('Threshold')
        axs[1, 1].legend(loc = 'lower right')
        axs[1, 1].set_xlim([0, 1])
        axs[1, 1].set_ylim([0, 1])  
        
        probs_train = model.predict_proba(X_train)
        preds_train = probs_train[:,1]
        fpr_t, tpr_t, threshold_t = metrics.roc_curve(y_train, preds_train)
        roc_auc_t = metrics.auc(fpr_t, tpr_t)
    # ROC for training
        axs[0, 0].title.set_text('Training ROC and KS')
        axs[0, 0].plot(fpr_t, tpr_t, 'b', label = 'AUC = %0.3f' % roc_auc_t)
        axs[0, 0].legend(loc = 'lower right')
        axs[0, 0].plot([0, 1], [0, 1],'r--')
        axs[0, 0].set_xlim([0, 1])
        axs[0, 0].set_ylim([0, 1])
        axs[0, 0].set_ylabel('True Positive Rate')
        axs[0, 0].set_xlabel('False Positive Rate')
    # KS for training
        axs[1, 0].plot(1-threshold_t, tpr_t,label = 'KS = %0.3f' % max(tpr_t-fpr_t))
        axs[1, 0].plot(1-threshold_t, fpr_t)
        axs[1, 0].legend(loc = 'lower right')
        axs[1, 0].set_xlim([0, 1])
        axs[1, 0].set_ylim([0, 1])
        axs[1, 0].set_ylabel('TPR / FPR')
        axs[1, 0].set_xlabel('Threshold')


def searchthreshold(model,threshold):
    for i in threshold:
        print('Threshold = %s' % i)
        y_pred=(model.predict_proba(X_test)[:,1]>=i).astype(int)
        print('Classification report for validation dataset')   
        print(classification_report(y_test, y_pred)[0:163])
        
    plt.figure(figsize=(12,6))   
    precision, recall, _ = precision_recall_curve(y_test, (model.predict_proba(X_test))[:,1])
    plt.step(recall, precision, color='#004a93', alpha=0.2,where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='#48a6ff')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve', fontsize=16)


def learningcurve(model,X,y):
    X_shuf, y_shuf = shuffle(X, y)
    train_sizes, train_scores, test_scores = learning_curve(model, X_shuf, y_shuf, cv=5, train_sizes=list(np.arange(0.1,1,0.1)),scoring = 'accuracy')
    train_error =  1-np.mean(train_scores,axis=1)
    test_error = 1- np.mean(test_scores,axis=1)
    plt.plot(train_sizes,train_error,'o-',color = 'r',label = 'training')
    plt.plot(train_sizes,test_error,'o-',color = 'g',label = 'validation')
    plt.legend(loc='best')
    plt.xlabel('traing examples')
    plt.ylabel('error')
    plt.show()
    

def dataprepare(sample):
    # Numerical variables
    for i in ['FICO','DTI', 'CLTV']:
        sample[i]= sample[i].transform(lambda x: (x-x.mean())/x.std())

    # Categorical variables
    temp = pd.get_dummies(sample[['First Time Homebuyer Flag','Occupancy','Channel', 'Property Type', 'Purpose','Number of borrowers']])
    sample = sample.drop(['First Time Homebuyer Flag','Occupancy','Channel', 'Property Type', 'Purpose','Number of borrowers'],axis=1)
    sample = sample.reset_index(drop=True)
    temp = temp.reset_index(drop=True)
    sample = sample.merge(temp,left_index=True, right_index=True)
    sample = sample.dropna(axis=0,how='any')
    sample['Number of borrowers_1']=2-sample['Number of borrowers']
    sample = sample.drop(['Units_1','First Time Homebuyer Flag_N','Occupancy_P','Channel_B','Property Type_PU','Purpose_P','Number of borrowers'],axis=1)
    
    return (sample)


def plot_single_feat_contrib(feat_name, contributions, features_df,
                             class_index=0, class_name='', add_smooth=False,
                             frac=2/3, **kwargs):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    """Plots a single feature's values across all observations against
    their corresponding contributions.

    Inputs:
    feat_name - The name of the feature
    contributions - The contributions from treeinterpreter
    features_df - A Pandas DataFrame with the features
    class_index - The index of the class to plot (Default: 0)
    class_name - The name of the class being plotted (Default: '')
    add_smooth - Add a lowess smoothing trend line (Default: False)
    frac - The fraction of data used when estimating each y-value
           (Default: 2/3)
    """


    # Create a DataFrame to plot the contributions
    def _get_plot_df():
        """Gets the feature values and their contributions."""

        if len(contributions.shape) == 2:
            contrib_array = contributions[:, feat_index]
        elif len(contributions.shape) == 3:
            contrib_array = contributions[:, feat_index, class_index]
        else:
            raise Exception('contributions is not the right shape.')

        plot_df = pd.DataFrame({'feat_value': features_df[feat_name].tolist(),
                                'contrib': contrib_array
                               })
        return plot_df

    def _get_title():
        # Set title according to class_
        if class_name == '':
            return 'Contribution of {}'.format(feat_name)
        else:
            return 'Conribution of {} ({})'.format(feat_name, class_name)

    def _plot_contrib():
        # If a matplotlib ax is specified in the kwargs, then set ax to it
        # so we can overlay multiple plots together.
        if 'ax' in kwargs:
            ax = kwargs['ax']
            # If size is not specified, set to default matplotlib size
            if 's' not in kwargs:
                kwargs['s'] = 40
            plot_df\
                .sort_values('feat_value')\
                .plot(x='feat_value', y='contrib', kind='scatter', **kwargs)
            ax.axhline(0, c='black', linestyle='--', linewidth=2)
            ax.set_title(title)
            ax.set_xlabel(feat_name)
            ax.set_ylabel('Contribution')
        else:
            plt.scatter(plot_df.feat_value, plot_df.contrib, **kwargs)
            plt.axhline(0, c='black', linestyle='--', linewidth=2)
            plt.title(title)
            plt.xlabel(feat_name)
            plt.ylabel('Contribution')

    def _plot_smooth():
        # Gets lowess fit points
        x_l, y_l = lowess(plot_df.contrib, plot_df.feat_value, frac=frac).T
        # Overlays lowess curve onto data
        if 'ax' in kwargs:
            ax = kwargs['ax']
            ax.plot(x_l, y_l, c='black')
        else:
            plt.plot(x_l, y_l, c='black')

    # Get the index of the feature
    feat_index = features_df.columns.get_loc(feat_name)
    # Gets the DataFrame to plot
    plot_df = _get_plot_df()
    title = _get_title()
    _plot_contrib()

    if add_smooth:
        _plot_smooth()



# Read data from teammates
dataTrain = pd.read_csv('trainsample_4.0.csv')
dataTrain.columns = ['Loanid','FICO', 'First Time Homebuyer Flag', 'Units','Occupancy','CLTV', 'DTI','Channel', 
                     'Property Type','Purpose', 'Number of borrowers','Default Status','OYear','First Payment Date',
                     'Property State','MSA','OLTV','OUPB','First Loan Age','OInterest Rate']
dataTrain['Units_1'] = dataTrain['Units'].apply(lambda x: 0 if x!=1 else 1)
dataTrain['Units_234'] = dataTrain['Units'].apply(lambda x: 1 if x!=1 else 0)
dataTrain = dataTrain.drop(['Units','OInterest Rate','First Payment Date','MSA','OLTV','OUPB','Property State','First Loan Age','Loanid','OYear'],axis=1)

dataTest = pd.read_csv('ValidationSample_4.0.csv')
dataTest = dataTest.drop('def_age',axis=1)
dataTest.columns = ['FICO', 'First Payment Date','First Time Homebuyer Flag', 'MSA','Units','Occupancy','CLTV', 'DTI','OUPB','OLTV',
                'OInterest Rate','Channel', 'Property State', 'Property Type','Loanid','Purpose','Number of borrowers', 'First Loan Age',
                 'Default Status']
dataTest['Units_1'] = dataTest['Units'].apply(lambda x: 0 if x!=1 else 1)
dataTest['Units_234'] = dataTest['Units'].apply(lambda x: 1 if x!=1 else 0)
dataTest = dataTest.drop(['Units','OInterest Rate','First Payment Date','MSA','OLTV','OUPB','Property State','First Loan Age','Loanid'],axis=1)



X_test = dataprepare(dataTest).drop(['Default Status'],axis=1)
y_test = dataprepare(dataTest).loc[:,'Default Status']
X_train = dataprepare(dataTrain).drop(['Default Status'],axis=1)
y_train = dataprepare(dataTrain).loc[:,'Default Status']

# Comparison between different balance technique (undersample, oversample, imbalanced dataset)
'''
data = dataTrain.append(dataTest)
sample = data.sample(1000000,replace=False, random_state=0)

# imbalanced data 
X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size = 0.25, random_state = 0)

# downsample
X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size = 25/700, random_state = 0)

insample = X_train.merge(pd.DataFrame(y_train),left_index=True,right_index=True)
insample['Default Status'].value_counts()[1]/insample['Default Status'].value_counts().sum()

insample_bal =  insample[insample['Default Status']==1]
temp = insample[insample['Default Status']==0].sample(insample[insample['Default Status']==1].count()[1],replace=False, random_state=0)
insample_bal = insample_bal.append(temp)
insample_bal['Default Status'].value_counts()[1]/insample_bal['Default Status'].value_counts().sum()
insample_bal = insample_bal.sample(frac=1).reset_index(drop=True)
X_train = insample_bal.drop(['Default Status'],axis=1)
y_train = insample_bal.loc[:, 'Default Status']

# oversample SMOTE
X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size = 25/65, random_state = 0)

from imblearn.over_sampling import SMOTE
smo = SMOTE(random_state=0)
X_smo, y_smo = smo.fit_sample(X_train, y_train)
insample_smo = pd.DataFrame(X_smo).merge(pd.DataFrame(y_smo),left_index=True,right_index=True)
insample_smo.columns = ['FICO', 'DTI', 'OLTV', 'Units_234','First Time Homebuyer Flag_Y', 'Occupancy_I', 'Occupancy_S',
       'Channel_C', 'Channel_R','Channel_T', 'Property Type_CO', 'Property Type_CP','Property Type_MH', 'Property Type_SF', 'Purpose_C', 'Purpose_N',
       'Number of borrowers_1','Default Status']
insample_smo['Default Status'].value_counts()[1]/insample_smo['Default Status'].value_counts().sum()
insample_smo = insample_smo.sample(frac=1).reset_index(drop=True)
X_train = insample_smo.drop(['Default Status'],axis=1)
y_train = insample_smo.loc[:, 'Default Status']
'''

# Multicolinearity check
vif = pd.DataFrame()
vif["features"] = X_train.columns
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]


# Build models 
'''Logistic Regression'''
# Logistic regression with stats
logit_model = sm.Logit(y_train, sm.add_constant(X_train)).fit()
print('ALL',logit_model.summary())

# Logistic regression with sklearn
log = LogisticRegression(random_state = 0,solver='lbfgs')
modeloutcome(log,X_train,y_train,X_test,y_test)
searchthreshold(log,[0.3,0.4,0.5,0.6,0.7,0.8])
learningcurve(log,X_train,y_train)
log.coef_

# C_parameter to regularize
C_param = [0.0001,0.001,0.01,0.1,1,10,100]
for c in C_param:
    log = LogisticRegression(random_state = 0,C=c)
    print(c)
    print(modeloutcome(log,X_train,y_train,X_test,y_test))
    print(log.coef_)


'''Tree model'''
# rf = RandomForestClassifier(n_estimators = 10,random_state = 0)
rf = RandomForestClassifier(random_state = 0)
rf = RandomForestClassifier(n_estimators = 50, max_depth=20, min_samples_split=300, criterion = 'entropy', random_state = 0)
rf = RandomForestClassifier(n_estimators = 50, max_depth=20, min_samples_split=0.01, max_leaf_nodes=1000,criterion = 'entropy', random_state = 0)

modeloutcome(rf,X_train,y_train,X_test,y_test)
searchthreshold(rf,[0.3,0.4,0.5,0.6,0.7,0.8])
learningcurve(rf,X_train,y_train)


# Hyperparameter grid search
param_grid = {
    'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
    'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
    'max_leaf_nodes': [None] + list(np.linspace(10, 500, 10).astype(int)),
    'min_samples_split': np.linspace(0.1, 0.5).astype(float),
    #'min_samples_leaf': [200]+np.linspace(0.1, 0.5).astype(float)
}

estimator = RandomForestClassifier(n_estimators = 50, oob_score= True, random_state = 0)
rs = RandomizedSearchCV(estimator, param_grid, n_jobs = -1, 
                        scoring = 'roc_auc', cv = 3, 
                        n_iter = 30, verbose = 2, random_state=50)

rs.fit(X_train, y_train)
rs.best_params_, rs.best_score_

'''
# Gridsearch
param_test = {'max_leaf_nodes': [None] + list(np.linspace(10, 500, 10).astype(int))}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(n_estimators = 50, min_samples_split=200,random_state=0), 
                       param_grid = param_test, scoring='roc_auc',cv=3)
gsearch1.fit(X_train,y_train)
gsearch1.best_params_, gsearch1.best_score_
'''

# Tree interpret
# Feature importance
pd.DataFrame(rf.feature_importances_, index = X_train.columns,columns=['importance'])

# Variable plots
X_test_sample = X_test
rf_pred, rf_bias, rf_contrib = ti.predict(rf, X_test_sample)

for i in X_test_sample.columns:
    name = plot_single_feat_contrib(i, rf_contrib, X_test_sample,class_index=1, add_smooth=True, frac=0.3)
    plt.show()


'''XGboost model'''
from xgboost import XGBClassifier
classifier = XGBClassifier()
modeloutcome(classifier,X_train,y_train,X_test,y_test)
searchthreshold(classifier,[0.3,0.4,0.5,0.6,0.7,0.8])
