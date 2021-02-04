# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 22:04:08 2020

@author: vominhthuan
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


# =================KHAI BÁO DATASET===============================================
data = pd.read_csv("F:\\Mon_Hoc_Nam_3\\NghienCuuKhoaHoc\\Anomaly-Detection-in-Networks-Using-Machine-Learning-master\\attacks\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

#==================GÁN LẠI NHÃN =================================================
new_Label=[]
for i in data[" Label"]:
    if i =="BENIGN":
        new_Label.append(0)
    else:
        new_Label.append(1)           
data[" Label"]=new_Label

#================LOẠI BỎ GIÁ TRỊ NULL=============================================
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame),"df cần được chuyển sang pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
data = clean_dataset(data)

#================TÍNH ĐỘ QUAN TRỌNG CỦA TỪNG ĐẶC TRƯNG===========================================

y = data[" Label"].values  
X = data.drop(' Label',axis=1).values

forest = RandomForestRegressor(n_estimators=250,random_state=0)
forest.fit(X, y)    
importances = forest.feature_importances_
features=list(data.columns.values)
impor = pd.DataFrame({'Features':features[0:20],'importance':importances[0:20]})
impor = impor.sort_values('importance',ascending=False).set_index('Features')   
print(impor.head(20))

#=====================Hiện thị đồ thị về độ quan trọng của thuộc tính===============================

plt.rcParams['figure.figsize'] = (10, 5)
impor.plot.bar();
plt.title("DDoS Attack - Feature Importance")
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

#========================Training với bộ 4 thuộc tính có trọng lượng gini cao nhất với thuật toán RandomForest=====================

selected_features = list(impor.index)[:4]
print("Những đặc trưng được chọn")
print(selected_features)
selected_features.append(' Label')
data  = data[selected_features]
y =data[' Label'].values
X = data.drop(' Label',axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.30)
rf = RandomForestClassifier(n_estimators=100)                                                                       
rf.fit(X_train, y_train)
predict =rf.predict(X_test)

#=========================Đo các thông số hiệu suất của===================================

from sklearn import metrics
acc = metrics.accuracy_score(y_test, predict)
print("Accuracy: ",acc)
rc=metrics.recall_score(y_test, predict)
print("Recall: ",rc)
pr=metrics.precision_score(y_test, predict)
print("Precision: ",pr)
f_1=metrics.f1_score(y_test, predict)
print("F_measure: ",f_1)

#=========================Trích xuất 1 cây quyết định trong Rừng ngẫu nhiên===================================================
from sklearn import tree
estimator = rf.estimators_[1]
tree.plot_tree(estimator) 











