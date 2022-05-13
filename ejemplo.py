import network
import network2
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


data=pd.read_csv('creditcard.csv')

y=data.Class
x=data.drop(['Time', 'Class'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.head()


x_train_f= x_train.values.tolist()
x_test_f= x_test.values.tolist()
y_train_f= y_train.values.tolist()
y_test_f= y_test.values.tolist()

x_entren=np.array(x_train_f)
x_entren_f=np.expand_dims(x_entren,axis = -1)
x_prueba=np.array(x_test_f)
x_prueba_f=np.expand_dims(x_prueba,axis = -1)

y_entr = []

for i in range(0,len(x_entren_f)):
    if y_train_f[i] == 0:
        y_entr.append([1,0])
    else:
        y_entr.append([0,1])
        
y_entren = np.array(y_entr)
y_entren_f=np.expand_dims(y_entren,axis = -1)

train = []
for i in range(0,len(x_entren_f)):
    train.append((x_entren_f[i],y_entren_f[i]))
        
test = []
for i in range(0,len(x_prueba_f)):
    if y_train_f[i] ==0:
        test.append((x_prueba_f[i],0))
    else:
        test.append((x_prueba_f[i],1))

net=network2.Network([29,10,2]) #inicializada la red

net.SGD( train[:2000], 31, 80, 0.001,lmbda = 100, evaluation_data=test[:200],monitor_evaluation_accuracy=True,monitor_training_cost=True) #entrena la red:datos entr, 
#epocas, mini-batch, learning rate, datos de prueba


