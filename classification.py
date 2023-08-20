import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix,precision_recall_curve, f1_score, roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn. ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

file_address = r'C:\Users\akand\Desktop\ML_programming_assignment2\data_csv.csv'
df = pd.read_csv(file_address)

df['famhist'] = df['famhist'].replace({'Absent': 0, 'Present': 1})


x = df.drop('chd', axis = 1)
y = df.chd

x.set_index('row.names', inplace = True)

#model = SVC(kernel = 'linear')
#model = GaussianNB()
#model = KNeighborsClassifier(n_neighbors = 10, weights = 'uniform')
model = RandomForestClassifier(n_estimators = 100)
x_array = np.array(x)
y_array = np.array(y)

from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(model, x_array, y_array, cv = 10)

conf_mat = confusion_matrix(y_array, y_pred)
print(f'Confusion Matrix: {conf_mat}')

num_classes = len(np.unique(y_array))
for i in range(num_classes):
    tp = conf_mat[i, i]
    fp = np.sum(conf_mat[:, i]) - tp
    fn = np.sum(conf_mat[i, :]) - tp
    tn = np.sum(conf_mat) - tp - fp - fn

    
    tp_rate = tp / (tp + fn)
    fp_rate = fp / (fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = f1_score(y_array, y_pred, pos_label=i)
    fpr, tpr, thresholds = roc_curve(y_array, y_pred, pos_label=i)
    roc_auc = roc_auc_score(y_array, y_pred, multi_class='ovr')
    
    print(f'Class {i} metrics:')
    print(f'TP rate: {tp_rate}')
    print(f'FP rate: {fp_rate}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 score: {f1}')
    print(f'ROC AUC score: {roc_auc}')
   

correct_indices = np.where(y == y_pred)[0]
currect_len = len(correct_indices)
print(f'Correct Instances: {currect_len}')

incorrect_len = len(y_array) - len(correct_indices)
print(f'Incorrect Instances : {incorrect_len}')

accuracy = (currect_len / (currect_len + incorrect_len)) * 100
print(f'Accuracy: {accuracy} %')