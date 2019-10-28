import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_auc(X,y,**kwargs):
    # Construct training and testing set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Choose classifier.
    classifier = LogisticRegression(**kwargs)
    classifier.fit(X_train, y_train)
    probas_ = classifier.predict_proba(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    #print "Area under the ROC curve : %f" % roc_auc
    return fpr, tpr, roc_auc, thresholds

def generate_auc_tree(X_train,X_test,y_train,y_test,tree_depth):
    # Construct training and testing set.
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Choose SVC classifier.
    classifier = RandomForestClassifier(max_depth=tree_depth)
    classifier.fit(X_train, y_train)
    probas_ = classifier.predict_proba(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, thresholds
    
from scipy.io import loadmat

#initial data first processed with MATLAB
MAT = loadmat('send6-1.mat') #Uses subject 6 as example
X_all = MAT['X_all']
y_all = MAT['Y_EEG_TRAIN']
y_all = np.ravel(y_all)
X_test1 = X_all[73:,:]
X_all = X_all[0:73,:]

y_all.shape
X_all.shape

#Use PCA to reduce dimensions

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_all,y_all)
X_all_fit = pca.transform(X_all)
X_test2 = pca.transform(X_test1)
X_all = X_all_fit


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

#no L1 penalty
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=42)
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test)
print(y_predict)
print(y_test)

y_predict_prob = classifier.predict_proba(X_test)
fpr,tpr,thresholds = roc_curve(y_test,y_predict_prob[:,1])
roc_auc = auc(fpr,tpr)    
    
plt.plot(fpr, tpr, '.-',label='ROC curve (area = %0.2f)' % roc_auc)
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
print(roc_auc)

alphas = np.logspace(0.01,1,1000)
scores = []

for i in alphas:
    fpr,tpr,roc_auc,thresholds = generate_auc(X_all,y_all,C=i,penalty='l1')
    scores.append(roc_auc)


    
plt.plot(alphas,scores)
best_alpha = alphas[np.argmax(scores)]
print(best_alpha)
print(np.max(scores))

#Cross-validate results
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)
kf.get_n_splits(X_all)
fpr_all = []
tpr_all = []
coeffs = []

plt.figure(figsize=(8,8))
i = 1
for train_index, test_index in kf.split(X_all):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_all[train_index], X_all[test_index]
    y_train, y_test = y_all[train_index], y_all[test_index]
    classifier = LogisticRegression(C=best_alpha,penalty='l1')
    y_predict_prob = classifier.fit(X_train,y_train).predict_proba(X_test)
    fpr,tpr,thresholds = roc_curve(y_test,y_predict_prob[:,1])
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr, '.-',
             label='ROC fold %i (AUC = %0.2f)' % (i,roc_auc))
    plt.legend(loc="lower right")
    fpr_all.extend(fpr)
    tpr_all.extend(tpr)
    i += 1
    coeffs.append(classifier.coef_)
    
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

coeffs = np.squeeze(np.array(coeffs))
coeff_mean = np.mean(coeffs,axis=0)
dfCoeffs = pd.DataFrame({'window':np.linspace(1,X_all.shape[1],X_all.shape[1]), 'coef': coeff_mean})

#Leave-one-out 
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
proba = []
for train_index, test_index in loo.split(X_all):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_all[train_index], X_all[test_index]
    y_train, y_test = y_all[train_index], y_all[test_index]
    classifier = LogisticRegression(penalty='l1')
    y_predict_prob = classifier.fit(X_train,y_train).predict_proba(X_test)
    proba.extend(y_predict_prob)
proba = np.array(proba)
fpr,tpr,thresholds = roc_curve(y_all,proba[:,1])
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr, '.-', 
         label='AUC = %0.2f' % (roc_auc))
plt.legend(loc="lower right")
print(roc_auc)

dfCoeffs_sorted = dfCoeffs.sort_values(['coef'])[::-1]
dfCoeffs_sorted.plot(x='window',y='coef',kind='bar',figsize=(45,15))

X_train = X_all
y_train = y_all
classifier = LogisticRegression(C=best_alpha,penalty='l1')
classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test2)
print(y_predict)
