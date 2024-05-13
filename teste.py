'''import numpy as np
import matplotlib.pyplot as plt

# Função que gera as coordenadas dos APs
def distribuir_APs(M):
    if M not in [1, 4, 9, 16, 25, 36, 49, 64, 100]:
        return None

    tamanho_quadrado = 1000
    lado_quadrado = int(np.sqrt(M))

    tamanho_celula = tamanho_quadrado // lado_quadrado

    # Criar coordenadas usando meshgrid
    x, y = np.meshgrid(np.arange(0.5 * tamanho_celula, tamanho_quadrado, tamanho_celula),
                      np.arange(0.5 * tamanho_celula, tamanho_quadrado, tamanho_celula))

    coordenadas_APs = np.column_stack((x.ravel(), y.ravel()))

    return coordenadas_APs

a = distribuir_APs(4)
plt.scatter(a[:, 0], a[:, 1])
plt.grid()
plt.show()'''

'''import numpy as np
def find_shadowing(passos):
    shadowing = []

    valor_atual = np.random.lognormal(0, 2)  # Inicializa o shadowing com sigma = 2 [LINEAR]
    for i in range(passos):
        if (i) % 10 == 0 and i != 0:  # Atualiza o shadowing a cada 10 passos, exceto no passo 0
            valor_atual = np.random.lognormal(0, 2)
        shadowing.append(valor_atual)  # Adiciona o valor atual à lista
    return shadowing

a = find_shadowing(100)
b = a[0] - a[9]
print(b)'''

import numpy as np

# Sua lista ou array
valores = np.array([1, 3, 2, 4, 5])

# Número de índices máximos a serem encontrados
N = 3

# Use np.argsort para ordenar os valores, e pegue os últimos N índices para os maiores valores
indices_maiores = np.argsort(valores)[-N:][::-1]

print(indices_maiores)


