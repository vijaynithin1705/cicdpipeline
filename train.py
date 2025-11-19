import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
import pickle

dataset = pd.read_csv("Iris.csv")

dataset = dataset.rename(columns={
    "Species": "target",
    "SepalLengthCm": "sepal_length",
    "SepalWidthCm": "sepal_width",
    "PetalLengthCm": "petal_length",
    "PetalWidthCm": "petal_width"
    })
l_en = LabelEncoder()

dataset['target'] = l_en.fit_transform(dataset['target'])


dataset.columns = [f.strip(' (cm)').replace(' ', '_') for f in dataset.columns.tolist()]
feature_names = dataset.columns.tolist()[1:5]

dataset['sepal_length_width_ratio'] = dataset['sepal_length'] / dataset['sepal_width']
dataset['petal_length_width_ratio'] = dataset['petal_length'] / dataset['petal_width']

dataset = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 
                   'sepal_length_width_ratio', 'petal_length_width_ratio', 'target']]

train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=44)
X_train = train_data.drop('target', axis=1).values.astype("float32")
y_train = train_data.loc[:, 'target'].values.astype("int32")
X_test = test_data.drop('target', axis=1).values.astype("float32")
y_test = test_data.loc[:, 'target'].values.astype("int32")

log_reg = LogisticRegression(C =0.0001 ,solver='lbfgs', max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
predictions_lr = log_reg.predict(X_test)

cm_lr = confusion_matrix(y_test, predictions_lr)
precision_lr = precision_score(y_test, predictions_lr, average='micro')
recall_lr = recall_score(y_test, predictions_lr, average='micro')
f1_lr = f1_score(y_test, predictions_lr, average='micro')

train_acc_lr = log_reg.score(X_train, y_train) * 100
test_acc_lr = log_reg.score(X_test, y_test) * 100

r_forest = RandomForestClassifier()
r_forest.fit(X_train, y_train)
predictions_rf = r_forest.predict(X_test)
predictions_rf_class = np.round(predictions_rf).astype(int)

cm_rf = confusion_matrix(y_test, predictions_rf_class)
precision_rf = precision_score(y_test, predictions_rf_class, average='micro')
recall_rf = recall_score(y_test, predictions_rf_class, average='micro')
f1_rf = f1_score(y_test, predictions_rf_class, average='micro')

train_acc_rf = r_forest.score(X_train, y_train) * 100
test_acc_rf = r_forest.score(X_test, y_test) * 100

with open("model.pkl", "wb") as f:
    pickle.dump(r_forest, f)

with open("scores.txt", "w") as score:
    score.write(f"Random Forest Train Accuracy: {train_acc_rf:.2f}%\n")
    score.write(f"Random Forest Test Accuracy: {test_acc_rf:.2f}%\n")
    score.write(f"F1 Score: {f1_rf:.2f}\n")
    score.write(f"Recall Score: {recall_rf:.2f}\n")
    score.write(f"Precision Score: {precision_rf:.2f}\n")






