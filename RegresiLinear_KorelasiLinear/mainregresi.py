import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.DataFrame([[151,63],[174,81],[138,56],[186,91],[128,47],[136,57],[179,76],[163,72],[152,62],[131,48]])
df.columns = ['x', 'y']
x_train = df['x'].values[:,np.newaxis]
y_train = df['y'].values
lm = LinearRegression()
lm.fit(x_train,y_train) #fase training
print("-------------------------")
print("| Mikael Prapaskalis G - 152017104|")
print("| Pemograman Simulasi Kelas B     |")
print("-------------------------\n")
print('Coefficient : ' + str(lm.coef_))
print('Intercept : ' + str(lm.intercept_))
x_test = [[170],[171]] #data yang akan diprediksi
p = lm.predict(x_test) #fase prediksi
print(p) #hasil prediksi
#prepare plot (menampilkan grafik)
pb = lm.predict(x_train)
dfc = pd.DataFrame({'x': df['x'],'y':pb})
plt.scatter(df['x'],df['y'])
plt.plot(dfc['x'],dfc['y'],color='red',linewidth=1)
plt.xlabel('Tinggi (cm)')
plt.ylabel('Berat (Kg)')
plt.show()
