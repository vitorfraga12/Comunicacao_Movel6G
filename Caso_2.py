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
def find_path_gain(dist, shadowing):
    k = 1e-4
    n = 4
    path_gain_result = shadowing * (k / ((dist) ** n)) #Obtem o path_gain local de cada AP

    return path_gain_result # [LINEAR]

# Função que calcula a potência recebida
def find_pot_rec(pot_trans, dist, d_0, shadowing):
    k = 1e-4
    n = 4
    if dist >= d_0:
        pot_rec_result = shadowing * (pot_trans * (k / ((dist) ** n))) #Obtem a potência recebida de cada AP

    return pot_rec_result # [LINEAR]

def calculate_snr(B_t, p_t, d_0, K_0, M, N, passos, shadowing, cluster):
    #UE irá se mover metro por metro e irá iniciar do ponto (0, 500) e irá até (1000, 500)
    x_coord = np.zeros(passos)
    y_coord = np.zeros(passos)
    for passo in range (passos):
        x_coord[passo] = (passo+1)
        y_coord[passo] = 500 

    #Definindo váriaveis locais
    ap_coord = distribuir_APs(M)
    power_noise = K_0*(B_t/N)
    distance = np.zeros(M)
    path_gain = np.zeros(M)
    power_rec = np.zeros(passos)
    snr = np.zeros(cluster)
    snr_final = []
    
    #Fazendo o Handover
    for passo in range(passos):
        for index_AP in range(M):
            distance[index_AP] = dAPUE(x_coord[passo], y_coord[passo], ap_coord[index_AP])
            path_gain[index_AP] = find_path_gain(distance[index_AP], shadowing[passo])
        max_index = np.argsort(path_gain)[-cluster:][::-1]
        for clust in range(len(max_index) ):
            indice = max_index[clust]
            power_rec[clust] = find_pot_rec(p_t, distance[indice], d_0, shadowing[passo])
            snr[clust] = power_rec[clust]/power_noise
        snr_sum = np.sum(snr)
        snr_final.append(snr_sum)
    return snr_final

def calculate_capacity(B_t, p_t, d_0, K, M, N, passos, shadowing, cluster):
    snr = calculate_snr(B_t, p_t, d_0, K, M, N, passos, shadowing, cluster)
    B_c = B_t/N
    capacity = np.zeros(passos)
    for i in range(passos):
        capacity[i] = B_c * np.log2(1 + snr[i])
        
    return capacity

B_t, p_t, d_0, K_0 = 100e6, 1e3, 1, 1e-17 # Em MHz, mW, metros, mW/Hz respectivamente
ap, ue, channel = 100, 1, 1
passos = 1000
cluster = 1
shadow = find_shadowing(passos)
snr = calculate_snr(B_t, p_t, d_0, K_0, ap, channel, passos, shadow, cluster)
capacity = calculate_capacity(B_t, p_t, d_0, K_0, ap, channel, passos, shadow, cluster)
passos_array = np.arange(passos)
cdf_capacity = np.sort(capacity)

# Plotando a Capacidade pela distância percorrida
plt.plot(passos_array, capacity)
plt.xlabel('Passos')
plt.ylabel('Capacidade (bps)')
plt.title('Capacidade da Rede Cellfree')
plt.grid()
plt.show()

# Plotando a CDF da Capacidade
plt.plot(cdf_capacity, np.arange(0, len(capacity)) / len(capacity))
plt.xlabel('Capacidade (bps)')
plt.ylabel('Porcentagem')
plt.title('CDF da Capacidade da Rede Cellfree')
plt.grid()
plt.show()

snr_db = 10*np.log10(snr)
cdf_snr = np.sort(snr_db)

# Plotando a CDF da SNR
plt.plot(cdf_snr, np.arange(0, len(capacity)) / len(capacity))
plt.xlabel('SNR (dB)')
plt.ylabel('Porcentagem')
plt.title('CDF da SNR da Rede Cellfree')
plt.grid()
plt.show()
