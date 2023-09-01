import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn

x = [1, 2, 3, 4, 5, 6, 7]

a = int(input('Gastos com *ALIMENTAÇÃO* dia 1: '))
b = int(input('Gastos com *ALIMENTAÇÃO* dia 2: '))
c = int(input('Gastos com *ALIMENTAÇÃO* dia 3: '))
d = int(input('Gastos com *ALIMENTAÇÃO* dia 4: '))
e = int(input('Gastos com *ALIMENTAÇÃO* dia 5: '))
f = int(input('Gastos com *ALIMENTAÇÃO* dia 6: '))
g = int(input('Gastos com *ALIMENTAÇÃO* dia 7: '))

y = [a, b, c, d, e, f, g]

h = int(input('Gastos com *VESTUÁRIO* dia 1: '))
j = int(input('Gastos com *VESTUÁRIO* dia 2: '))
k = int(input('Gastos com *VESTUÁRIO* dia 3: '))
l = int(input('Gastos com *VESTUÁRIO* dia 4: '))
m = int(input('Gastos com *VESTUÁRIO* dia 5: '))
n = int(input('Gastos com *VESTUÁRIO* dia 6: '))
o = int(input('Gastos com *VESTUÁRIO* dia 7: '))

y1 = [h, j, k, l, m, n, o]

p = int(input('Gastos com *TRANSPORTE* dia 1: '))
q = int(input('Gastos com *TRANSPORTE* dia 2: '))
r = int(input('Gastos com *TRANSPORTE* dia 3: '))
s = int(input('Gastos com *TRANSPORTE* dia 4: '))
t = int(input('Gastos com *TRANSPORTE* dia 5: '))
u = int(input('Gastos com *TRANSPORTE* dia 6: '))
v = int(input('Gastos com *TRANSPORTE* dia 7: '))

y2 = [p, q, r, s, t, u, v]

plt.plot(x,y, marker='D')
plt.plot(x,y1, marker='D', color='orange')
plt.plot(x,y2, marker='D', color='green')
plt.ylim(7, 500)
plt.legend(["ALIMENTAÇÃO","VESTUÁRIO","TRANSPORTE"])
plt.title('Despesas da semana')
plt.xlabel('Dia')
plt.ylabel('Despesas em R$')
plt.show()

x_ = np.array(x)
y_ = np.array(y)
media_x=np.mean(x)*2
model=LinearRegression(fit_intercept=True)
model.fit(x_[:, np.newaxis], y_)
xfit=np.linspace(0, (media_x), 100)
yfit=model.predict(xfit[:,np.newaxis])
plt.plot(xfit,yfit, color='black', label = 'Regressão Linear Alimentação')
plt.plot(x_,y_, marker='D')
plt.title('Regressão Linear')
plt.xlabel('Dia')
plt.ylabel('Despesas em R$')
plt.legend(["Regressão Linear Alimentação","ALIMENTAÇÃO"])
plt.ylim(7, 500)
plt.show()

x_ = np.array(x)
y1_ = np.array(y1)
media_x=np.mean(x)*2
model=LinearRegression(fit_intercept=True)
model.fit(x_[:, np.newaxis], y1_)
xfit=np.linspace(0, (media_x), 100)
y1fit=model.predict(xfit[:,np.newaxis])
plt.plot(xfit,y1fit, color='gray', label = 'Regressão Linear')
plt.plot(x_,y1_, marker='D', color='orange')
plt.title('Regressão Linear')
plt.xlabel('Dia')
plt.ylabel('Despesas em R$')
plt.legend(["Regressão Linear Vestuário","VESTUÁRIO"])
plt.ylim(7, 500)
plt.show()

x_ = np.array(x)
y2_ = np.array(y2)
media_x=np.mean(x)*2
model=LinearRegression(fit_intercept=True)
model.fit(x_[:, np.newaxis], y2_)
xfit=np.linspace(0, (media_x), 100)
y2fit=model.predict(xfit[:,np.newaxis])
plt.plot(xfit,y2fit, color='brown', label = 'Regressão Linear')
plt.plot(x_,y2_, marker='D', color='green')
plt.title('Regressão Linear')
plt.xlabel('Dia')
plt.ylabel('Despesas em R$')
plt.legend(["Regressão Linear Transporte","TRANSPORTE"])
plt.ylim(7, 500)
plt.show()

