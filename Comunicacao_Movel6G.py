import numpy as np
import random
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
    valor_atual = np.random.lognormal(0, 2)  # Inicializa o shadowing
    for i in range(passos):
        if i % 10 == 0 and i != 0:  # Atualiza o shadowing a cada 10 passos, exceto no passo 0
            valor_atual = np.random.lognormal(0, 2)
        shadowing.append(valor_atual)  # Adiciona o valor atual à lista
    return shadowing

# Função que calcula a potência recebida
def pot_rec(pot_trans, dist, d_0, passos):
    k = 1e-4
    n = 4
    pot_rec_result = [] # Inicializa a variável local
    shadowing = find_shadowing(passos)
    for i in range(passos):
        if dist >= d_0:
            result = shadowing[i] * pot_trans * (k / ((dist) ** n))
            pot_rec_result.append(result)

    return pot_rec_result

# Função que calcula o path gain
def path_gain(dist, M, passos):
    k = 1e-4
    n = 4
    shadowing = find_shadowing(passos)
    path_gain_result = [] # Inicializa a variável local
    for i in range(passos):
        result = shadowing[i] * (k / ((dist) ** n))
        path_gain_result.append(result)
    
    return path_gain_result

def simular_experimento(B_t, p_t, d_0, K, M, N, passos):

    #UE irá se mover metro por metro e irá iniciar do ponto (0, 500) e irá até (1000, 500)
    x_coord = np.zeros(passos)
    y_coord = np.zeros(passos)
    for i in range (passos):
        x_coord[i] = (i+1)
        y_coord[i] = 500 
    ap_coord = distribuir_APs(M)
    for i in range(passos):
        for j in range(M):
            distancia = dAPUE(x_coord[i], y_coord[i], ap_coord[j])
            power_rec = pot_rec(p_t, distancia, d_0, passos)
            path_g = path_gain(distancia, M, passos)

    distanciaAP_UE = np.zeros(M)
    potencia_recebida = np.zeros(M)
    p_n = K_0*(B_t/N)
    SNR = np.zeros(M)
    for i in range(M):
        distanciaAP_UE[i] = dAPUE(x_coord, y_coord, coordAp[i])
        distan = distanciaAP_UE[i]
        potencia_recebida[i] = pot_rec(p_t, distan, d_0)
        SNR[i] = potencia_recebida[i]/p_n

    #Calculando a Capacidade
    B_c = B_t / N
    snr_2 = np.sum(SNR)
    Capacidade = np.zeros(K)
    for i in range(K):
      Capacidade[i] = B_c * np.log2(1+snr_2)

    return SNR

