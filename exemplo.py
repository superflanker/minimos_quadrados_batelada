
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

with open('dryer.dat') as file:
  medidas = [list(map(float, line.split())) for line in file]

medidas = pd.DataFrame(medidas)
print(medidas.shape)
medidas.columns = ['u', 'y']
print(medidas)

u = medidas['u']
y = medidas['y']

print(u, y)


# Limpar variáveis, fechar gráficos e limpar a tela
np. set_printoptions(precision=3) # apenas para exibir valores com 3 casas decimais
plt.close('all')
print("\n\n Método dos mínimos quadrados em batelada\n\n")
# Carregar conjunto de dados
npts = medidas.shape[0]
u = medidas['u'].to_numpy()
y = medidas['y'].to_numpy()

# Vetor de medidas fi
fi = np.empty((0, 4))
for j in range(npts):
  if j <= 1:
      y1, y2, u1, u2 = y[0], y[0], 0, 0 # condições iniciais
  else:
      y1, y2, u1, u2 = y[j-1], y[j-2], u[j-1], u[j-2] # atrasos nas entradas e saídas
  fi = np.vstack((fi, [-y1, -y2, u1, u2]))
# Vetor de parâmetros estimados
theta = np.linalg.inv(fi.T @ fi) @ fi.T @ y
numparametros = theta.size
print(f'Número de parâmetros estimados (na + nb): {numparametros}\n')
for i in range(numparametros):
    print(f'theta({i+1}) = {theta[i]:.6f}')

# Previsão n passos à frente e um passo à frente
yest_n = np.zeros(npts)
yest_1 = np.zeros(npts)
for t in range(2):
    yest_n[t] = y[t]
    yest_1[t] = y[t]
a1, a2, b1, b2 = theta[0], theta[1], theta[2], theta[3]
for t in range(2, npts):
    # previsão n passos a frente (previsão/simulação livre, n passos à frente)
    yest_n[t] = -a1*yest_n[t-1] - a2*yest_n[t-2] + b1*u[t-1] + b2*u[t-2]
    # previsão um passo a frente (previsão de curtíssimo prazo, um passo à frente)
    yest_1[t] = -a1*y[t-1] - a2*y[t-2] + b1*u[t-1] + b2*u[t-2]
# Mean Squared Error (MSE)
MSE_1 = np.sum((y-yest_1)**2) / (len(y)-2)
MSE_n = np.sum((y-yest_n)**2) / (len(y)-2)
print(f'\nMSE_1: {MSE_1:.6f}\nMSE_n: {MSE_n:.6f}\n')

# Plotar gráfico
plt.plot(y, 'g')
plt.plot(yest_1, 'r')
plt.plot(yest_n, 'b:')
desempenho = f'Desempenho do modelo - MSE_1: {MSE_1:.6f} MSE_n: {MSE_n:.6f}'
plt.title(desempenho)
plt.legend(['real', 'um passo à frente', 'n passos à frente'])
plt.xlabel('amostra')
plt.ylabel('saída')
plt.show()