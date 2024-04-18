import numpy as np
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


# Função que gera a distância entre a UE e a AP
def dAPUE(x_coord, y_coord, M):
  dAPUE = np.linalg.norm(np.array([x_coord, y_coord]) - M)
  return dAPUE

#Função que define o shadowing e muda ele a cada 10 passos
def find_shadowing(passos):
    shadowing = []

    valor_atual = np.random.lognormal(0, 2)  # Inicializa o shadowing com sigma = 2 [LINEAR]
    for i in range(passos):
        if (i) % 10 == 0 and i != 0:  # Atualiza o shadowing a cada 10 passos, exceto no passo 0
            valor_atual = np.random.lognormal(0, 2)
        shadowing.append(valor_atual)  # Adiciona o valor atual à lista
    return shadowing

# Função que calcula o path gain
def path_gain(dist, shadowing):
    k = 1e-4
    n = 4
    path_gain_result = shadowing * (k / ((dist) ** n)) #Obtem o path_gain local de cada AP

    return path_gain_result # [LINEAR]

# Função que calcula a potência recebida
def pot_rec(pot_trans, dist, d_0, shadowing):
    k = 1e-4
    n = 4
    if dist >= d_0:
        pot_rec_result = shadowing * (pot_trans * (k / ((dist) ** n))) #Obtem a potência recebida de cada AP

    return pot_rec_result # [LINEAR]


def calculate_snr(B_t, p_t, d_0, K_0, M, N, passos, shadowing):

    #UE irá se mover metro por metro e irá iniciar do ponto (0, 500) e irá até (1000, 500)
    x_coord = np.zeros(passos)
    y_coord = np.zeros(passos)
    for i in range (passos):
        x_coord[i] = (i+1)
        y_coord[i] = 500 

    #Definindo váriaveis locais
    ap_coord = distribuir_APs(M)
    p_n = K_0*(B_t/N)
    distancia = np.zeros(M)
    path_g = np.zeros(M)
    power_rec = np.zeros(passos)
    SNR = np.zeros(passos)

    #Fazendo o Handover
    for i in range(passos):
        for j in range(M):
            distancia[j] = dAPUE(x_coord[i], y_coord[i], ap_coord[j])
            path_g[j] = path_gain(distancia[j], shadowing[i])
        j_ = np.argmax(path_g)
        power_rec[i] = pot_rec(p_t, distancia[j_], d_0, shadowing[i])
        SNR[i] = power_rec[i]/p_n
    return SNR

def calculate_capacity(B_t, p_t, d_0, K, M, N, passos, shadowing):
    SNR = calculate_snr(B_t, p_t, d_0, K, M, N, passos, shadowing)
    B_c = B_t/N
    Capacity = B_c * np.log2(1 + SNR)
    return Capacity

B_t, p_t, d_0, K_0 = 100e6, 1e3, 1, 1e-17 # Em MHz, mW, metros, mW/Hz respectivamente
M, K, N = 100, 1, 1
passos = 1000
shadow = find_shadowing(passos)
SNR = calculate_snr(B_t, p_t, d_0, K_0, M, N, passos, shadow)
Capacity = calculate_capacity(B_t, p_t, d_0, K_0, M, N, passos, shadow)
passos_array = np.arange(passos)
cap = np.sort(Capacity)

# Plota o número de passos no eixo x e a capacidade no eixo y
plt.plot(passos_array, Capacity)
plt.xlabel('Passos')
plt.ylabel('Capacidade (bps)')
plt.title('Capacidade da Rede 6G')
plt.grid()
plt.show()

cap = np.sort(Capacity)

# Plota o número de passos no eixo x e a capacidade no eixo y
plt.plot(cap, np.arange(0, len(Capacity)) / len(Capacity))
plt.xlabel('Capacidade (bps)')
plt.ylabel('Porcentagem')
plt.title('Capacidade da Rede 6G')
plt.grid()
plt.show()
