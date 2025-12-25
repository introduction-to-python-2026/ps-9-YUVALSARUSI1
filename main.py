# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit
import pandas as pd

df = pd.read_csv('parkinsons.csv')
df.dropna(inplace=True)
df.head()
x = df[['MDVP:Fo(Hz)','MDVP:Jitter(%)']]
y = df ['status']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_scales = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(x_scales,y,test_size=0.2)
y_train
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit (x_train, y_train)
from sklearn.metrics import accuracy_score
y_pred = knn.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print(accuracy)
import joblib

joblib.dump(knn, 'my_model.joblib')
