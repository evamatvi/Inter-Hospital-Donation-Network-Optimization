import numpy as np
import random
from timeit import default_timer as timer
import math
import matplotlib.pyplot as plt

from my_bns import *
from inferencia import rejection_sampling
from inferencia import weighted_sampling
from inferencia import variable_elimination

#Funció per calcular la distància Euclidiana entre dos punts
def distancia(p1, p2):
    """ Distància Euclidiana entre dos punts"""
    return np.linalg.norm(p1 - p2)

#Funció per crear un diccionari a partir de dues llistes: descriptors i valors.
def obtenir_valors_per_variables(descriptors, valors, sufix=""):
    assert len(descriptors) == len(valors)
    e = {} #Diccionari on s'emmagatzemaran els valors vàlids
    for i in range(len(descriptors)):
        if not np.isnan(valors[i]): # Comprova si el valor no és NaN
            e[descriptors[i]+sufix]=valors[i]
    return e


#Funció per calcular la funció objectiu per una assignació donada
def funcio_objectiu(assignacio, coordenades, descriptors, matchBN, ordre_anc, mu=0.5, n_grups=4):
    N = len(assignacio)
    suma_ponderada = 0
    # Recorrem cada grup g per calcular les mètriques
    for g in range(n_grups):
         # Obtenim la llista d'hospitals que pertanyen al grup g
        hospitals_xarxa = [i for i in range(N) if assignacio[i] == g]
        Ng = len(hospitals_xarxa)

        if Ng > 1:
            # Càlcul de la distància intra-xarxa
            distancia_intra_xarxa = sum(
                distancia(coordenades[hospitals_xarxa[i]], coordenades[hospitals_xarxa[j]])
                for i in range(Ng - 1) for j in range(i + 1, Ng)
            )
            distancia_intra_xarxa *= (2 / (Ng * (Ng - 1)))

            # Si la distància calculada és NaN, la convertim a 0 per evitar errors
            if np.isnan(distancia_intra_xarxa):
                distancia_intra_xarxa = 0

            # Càlcul de la similitud intra-xarxa
            similitud_intra_xarxa = 0
            for i in range(Ng - 1):
                for j in range(i + 1, Ng):
                    hospital_1 = hospitals_xarxa[i]
                    hospital_2 = hospitals_xarxa[j]
                    # Comprovem que els hospitals tenen descriptors vàlids abans de continuar
                    if hospital_1 < len(descriptors) and hospital_2 < len(descriptors):
                        # Convertim els descriptors en un format adequat i eliminem NaN
                        valors = np.concatenate((
                            np.nan_to_num(descriptors[hospital_1]), 
                            np.nan_to_num(descriptors[hospital_2])
                        ))
                        # Creem la consulta per obtenir les variables rellevants
                        consulta = obtenir_valors_per_variables(["I1", "J1", "K1", "I2", "J2", "K2"], valors)
                        
                        # Apliquem un mètode de sampling per obtenir la probabilitat de match
                        probabilitat_match = rejection_sampling("M", consulta, matchBN, ordre_anc, N=5).__getitem__(True)
                        
                        if np.isnan(probabilitat_match):
                            probabilitat_match = 0
                        
                        similitud_intra_xarxa += probabilitat_match
            # Normalitzem la similitud intra-xarxa de manera similar a la distància intra-xarxa
            similitud_intra_xarxa *= (2 / (Ng * (Ng - 1)))
            if np.isnan(similitud_intra_xarxa):
                similitud_intra_xarxa = 0
            # Sumem el valor ponderat d'aquest grup a la funció objectiu
            suma_ponderada += (Ng / N) * (distancia_intra_xarxa - mu * similitud_intra_xarxa)

    return suma_ponderada

#Funció per generar veïns canviant aleatòriament un hospital d'un grup
def generar_veins(assignacio, n_grups=4, max_veins=10):
    """Genera un nombre limitat de veïns canviant un hospital de xarxa aleatòriament"""
    veïns = []
    #Seleccionem hospitals aleatoris
    hospitals_a_canviar = random.sample(range(len(assignacio)), min(max_veins, len(assignacio)))  # Seleccionar un nombre aleatori de hospitals
    #Per a cada hospital seleccionat i cada grup, si el grup no és el mateix, copiem l'assignació actual
    #i canviem de grup el hospital i afegim la nova assignació com a veí
    for i in hospitals_a_canviar:
        for g in range(n_grups):
            if assignacio[i] != g:
                veí = assignacio.copy()
                veí[i] = g
                veïns.append(veí)

    return veïns



#Funció de cerca local per feixos amb la funció objectiu adaptada
def cerca_local_beam(problema, beam_size=5, iteracions=100, tolerancia_beam=1e-2, max_sense_millora=10):
    # Extraiem les dades del diccionari del problema
    n_elements = problema["n_elements"]
    coordenades = problema["coordenades"]
    descriptors = problema["descriptors"]
    matchBN = problema["matchBN"]
    ordre_anc = problema["ordre_anc"]
    # Generació inicial del feix amb solucions aleatòries
    beam = [np.random.randint(0, 4, size=n_elements) for _ in range(beam_size)]
    # Avaluació inicial del feix segons la funció objectiu
    avaluacions = [funcio_objectiu(assignacio, coordenades, descriptors, matchBN, ordre_anc) for assignacio in beam]
    iteracions_sense_millora = 0
    for it in range(iteracions): # Bucle principal de cerca
        nous_veïns = []
         # Generem nous veïns per a cada assignació del feix
        for assignacio in beam:
            veïns = generar_veins(assignacio) # Troba possibles solucions properes
            nous_veïns.extend(
                (veí, funcio_objectiu(veí, coordenades, descriptors, matchBN, ordre_anc)) # Avaluem els veïns
                for veí in veïns
            )
        # Si no s'han generat nous veïns, s'atura la cerca
        if not nous_veïns:
            print("No s'han generat nous veïns. Aturant la cerca.")
            break
        # Ordenem els nous veïns segons la seva avaluació (millor primer)
        nous_veïns.sort(key=lambda x: x[1])
        millors_avaluacions = [x[1] for x in nous_veïns[:beam_size]]
        # Criteri d'aturada: si la millora és inferior a la tolerància
        if abs(avaluacions[0] - millors_avaluacions[0]) < tolerancia_beam:
            break
        # Comprovació d'iteracions sense millora
        if avaluacions[0] == millors_avaluacions[0]:
            iteracions_sense_millora += 1
            if iteracions_sense_millora >= max_sense_millora:
                break
        else:
            iteracions_sense_millora = 0 # reestableix el comptador si hi ha millores
         # Actualitzem el feix amb les millors assignacions trobades
        beam = [x[0] for x in nous_veïns[:beam_size]]
        avaluacions = millors_avaluacions
    return beam[0], avaluacions  # retorna la millor solució trobada i la seva avaluació

#Funció per calcular la funció objectiu per una assignació donada
def funcio_objectiu_inicial(assignacio, coordenades, n_grups=4):
    N = len(assignacio) #Nombre total d'elements
    suma_ponderada = 0 #Inicialitzem  la suma ponderada
    for g in range(n_grups):
        hospitals_xarxa = [i for i in range(N) if assignacio[i] == g] #Hospitals assignats al grup g
        Ng = len(hospitals_xarxa) #Nombre d'hospitals al grup
        if Ng > 1: #Si hi ha més d'un hospital al grup
            distancia_intra_xarxa = 0
            for i in range(Ng - 1): #Hem iterat sobre els hospitals del grup i els restants
                for j in range(i, Ng):
                    distancia_intra_xarxa += distancia(coordenades[hospitals_xarxa[i]], coordenades[hospitals_xarxa[j]])
            suma_ponderada += (Ng / N) * (2 / (Ng * (Ng - 1))) * distancia_intra_xarxa
    return suma_ponderada

#Funció de cerca local per feixos
def cerca_local_beam_inicial(problema1, beam_size=5, iteracions=100, tolerancia=1e-3):
    n_elements = problema1["n_elements"] #Nombre d'elemnents
    coordenades = problema1["data"] #Coordenades hospitals
    beam = [np.random.randint(0, 4, size=n_elements) for _ in range(beam_size)]
    avaluacions = [funcio_objectiu_inicial(assignacio, coordenades) for assignacio in beam]

    millores_per_iteracio = []

    for _ in range(iteracions):
        nous_veïns = []
        for assignacio in beam:
            veïns = generar_veins(assignacio)
            #Afegim els veïns i les seves avaluacions
            nous_veïns.extend((veí, funcio_objectiu_inicial(veí, coordenades)) for veí in veïns)
        #Ordenem els veïns per la funció objectiu i seleccionem els millors segons beam size
        nous_veïns.sort(key=lambda x: x[1])
        millors_avaluacions = [x[1] for x in nous_veïns[:beam_size]]
        #Comprovem la millora actual i l'afegim a la llista
        millora_actual = abs(avaluacions[0] - millors_avaluacions[0])
        millores_per_iteracio.append(millora_actual)
        #Actualitzem el beam amb els millors veïns i actualitzem les avaluacions
        beam= [x[0] for x in nous_veïns[:beam_size]]
        avaluacions = millors_avaluacions
        #Aturem per criteri de tolerància
        if millora_actual < tolerancia:
            print(f"Aturat per criteri d'aturada, la millora és menor que {tolerancia}")
            break

    return beam[0], millores_per_iteracio

#Funció de cerca aleatòria
def cerca_aleatoria(problema1, iteracions=100, tolerancia=1e-3):
    """ Cerca aleatòria: genera solucions aleatòries i selecciona la millor """
    n_elements = problema1["n_elements"]
    coordenades = problema1["data"]

    # Estat inicial aleatori
    estat_actual = np.random.randint(0, 4, size=n_elements)
    millor_objectiu = funcio_objectiu_inicial(estat_actual, coordenades)

    millores_per_iteracio = [millor_objectiu] #Funció objectiu inicial

    for _ in range(iteracions):
        #Hem generat una nova assignació aleatòria
        nova_assignacio = np.random.randint(0, 4, size=n_elements)
        nou_objectiu = funcio_objectiu_inicial(nova_assignacio, coordenades)

        # Si la nova solució és millor, l'assignem com a millor solució
        if nou_objectiu < millor_objectiu:
            estat_actual = nova_assignacio
            millor_objectiu = nou_objectiu

        # Afegim el valor de l'objectiu per analitzar la convergència
        millores_per_iteracio.append(millor_objectiu)

        # Comprovem el criteri d'aturada
        if len(millores_per_iteracio) > 1 and abs(millores_per_iteracio[-1] - millores_per_iteracio[-2]) < tolerancia:
            print(f"Aturat per criteri d'aturada, la millora és menor que {tolerancia}")
            break

    return estat_actual, millores_per_iteracio


#Funció hill-climbing
def hill_climbing(problema1, iteracions=100, tolerancia=1e-3):
    """ Cerca Hill Climbing: selecciona el millor successor de forma greedy """
    n_elements = problema1["n_elements"]
    coordenades = problema1["data"]

    # Estat inicial aleatori
    estat_actual = np.random.randint(0, 4, size=n_elements)

    millores_per_iteracio = []

    for _ in range(iteracions):
        # Generem veïns per l'estat actual
        veïns = generar_veins(estat_actual)

        # Seleccionem el millor veí
        millor_veí = min(veïns, key=lambda x: funcio_objectiu_inicial(x, coordenades))

        # Comprovem si el veí és millor que l'estat actual
        if funcio_objectiu_inicial(millor_veí, coordenades) < funcio_objectiu_inicial(estat_actual, coordenades):
            estat_actual = millor_veí

        # Afegim el valor de l'objectiu per analitzar la convergència
        millores_per_iteracio.append(funcio_objectiu_inicial(estat_actual, coordenades))

        # Comprovem el criteri d'aturada
        if len(millores_per_iteracio) > 1 and abs(millores_per_iteracio[-1] - millores_per_iteracio[-2]) < tolerancia:
            print(f"Aturat per criteri d'aturada, la millora és menor que {tolerancia}")
            break

    return estat_actual, millores_per_iteracio


#Funció de Simulated Annealing
def simulated_annealing(problema1, iteracions=100, temperatura_inicial=1000, temperatura_final=1, decrement_temperatura=0.99, tolerancia=1e-3):
    """ Simulated Annealing: busca un millor estat amb probabilitat d'acceptar solucions pitjors """
    n_elements = problema1["n_elements"] #Nombre d'elements
    coordenades = problema1["data"] #Coordenades dels hospitals

    # Estat inicial aleatori
    estat_actual = np.random.randint(0, 4, size=n_elements)
    millor_objectiu = funcio_objectiu_inicial(estat_actual, coordenades)

    millores_per_iteracio = [] #Llista
    temperatura = temperatura_inicial #Inicialitzem la temperatura

    while temperatura > temperatura_final:
        for _ in range(iteracions):
            # Generem veïns per l'estat actual
            veïns = generar_veins(estat_actual)
            veí = random.choice(veïns)

            objectiu_veí = funcio_objectiu_inicial(veí, coordenades) #Funció objectiu veí

            # Avaluem si acceptem el veí
            if objectiu_veí < millor_objectiu or random.random() < math.exp((millor_objectiu - objectiu_veí) / temperatura):
                estat_actual = veí
                millor_objectiu = objectiu_veí

            # Afegim el valor de l'objectiu per analitzar la convergència
            millores_per_iteracio.append(millor_objectiu)

        #Criteri de tolerància
        if len(millores_per_iteracio) > 1 and abs(millores_per_iteracio[-1] - millores_per_iteracio[-2]) < tolerancia:
            print(f"Aturat per criteri d'aturada, la millora és menor que {tolerancia}")
            break

        # Reduïm la temperatura
        temperatura *= decrement_temperatura

    return estat_actual, millores_per_iteracio

# Funció per obtenir la convergència de tots els algorismes
def obtenir_convergencia(algorismes, problema1, **params):
    convergencies = {}
    for nom, algorisme in algorismes.items():
        print(f"Executant {nom}...")
        # Passem  els paràmetres específics per a cada algorisme
        if nom == "Cerca Local Beam":
            millor_assignacio, convergencia = algorisme(problema1, **params)
        else:
            millor_assignacio, convergencia = algorisme(problema1, iteracions=params['iteracions'], tolerancia=params.get('tolerancia', 1e-3))
        convergencies[nom] = convergencia
    return convergencies


def main():
    # ------------------------------------------------------------
    # 1. Configurem el problema ----------------------------------
    np.random.seed(11) # Fixem la llavor per reproducibilitat
    # Carreguem les dades
    X = np.loadtxt("data.csv", delimiter=",", dtype=float, skiprows=1)

    #Dades per les cerques
    problema1 =dict()
    problema1["data"] = X[:,:2]
    problema1["n_elements"] = X.shape[0]
    problema1["ndim"] = X.shape[1]
    problema1["n_groups"] = 4


    # ------------------------------------------------------------
    # 2. Configurem i executem les cerques---------------
    beam_size = 5  # B
    n_iterations = 100  # K
    tolerancia= 1e-3
    tolerancia_beam=1e-2


    #Cerca local beam
    t_start = timer()
    res = cerca_local_beam_inicial(problema1, beam_size, n_iterations,tolerancia)
    t_end = timer()
    print("Millor assignació trobada per la cerca local beam:", res[0])
    print("Millor funció objectiu per la cerca local beam:", min (res[1]))
    print("Temps per cerca local beam adaptada:", t_end - t_start, "segons.")
    print()

    #Hill climbing
    t_start = timer()
    res_hill = hill_climbing(problema1, n_iterations,tolerancia)
    t_end = timer()
    print("Millor assignació trobada per hill climbing:", res_hill[0])
    print("Millor funció objectiu pel hill climbing:", min (res_hill[1]))
    print("Temps per hill climbing:", t_end - t_start, "segons.")
    print()

    #Cerca aleatòria
    t_start = timer()
    res_aleatori = cerca_aleatoria(problema1, n_iterations,tolerancia)
    t_end = timer()
    print("Millor assignació trobada per a la cerca aleatòria:", res_aleatori[0])
    print("Millor funció objectiu per a la cerca aleatòria:", min(res_aleatori[1]))
    print("Temps per cerca aleatòria:", t_end - t_start, "segons.")
    print()

    #Simulated annealing
    temperatura_inicial = 1000
    temperatura_final = 1
    decrement_temperatura = 0.99
    t_start = timer()
    res_annealing = simulated_annealing(problema1, n_iterations, temperatura_inicial, temperatura_final, decrement_temperatura,tolerancia)
    t_end = timer()
    print("Millor assignació trobada per Simulated Annealing:", res_annealing[0])
    print("Millor funció objectiu per Simulated Annealing:", min(res_annealing[1]))
    print("Temps per Simulated Annealing:", t_end - t_start, "segons")
    print()

        # Diccionari amb els algorismes
    algorismes = {
        "Cerca Local Beam": cerca_local_beam_inicial,
        "Hill Climbing": hill_climbing,
        "Simulated Annealing": simulated_annealing,
        "Cerca Aleatoria": cerca_aleatoria
    }

  

    # ------------------------------------------------------------
    # 3. Configurem i executem la incorporació del coneixement ---
    # imprecís ----------------------------------------------------

    #Dades per la incorporació de coneixament imprecís
    problema = dict()
    problema["data"] = X
    problema["n_elements"] = X.shape[0]
    problema["ndim"] = 2
    problema["coordenades"] = X[:, :2] # coordenades x i y
    problema["n_groups"] = 4
    problema["descriptors"] = X[:, 2:]  # descriptors I, J i K
    problema["matchBN"] = matchBN  # xarxa Bayesiana
    problema["ordre_anc"] = ordre_anc_matchBN  # ordre ancestral

    #Cerca local beam
    t_start = timer()
    res = cerca_local_beam(problema, beam_size, n_iterations,tolerancia_beam)
    t_end = timer()
    print()
    print("Millor assignació trobada per la cerca local beam adaptada:", res[0])
    print("Millor funció objectiu per la cerca local beam adaptada:", min (res[1]))
    print("Temps per cerca local beam adaptada:", t_end - t_start, "segons.")
    print()

    # Càlcul de la distància entre dos punts
    d = problema["ndim"]
    print("Distància entre punts 1 i 2:", distancia(problema["data"][1,:d], problema["data"][2,:d]))
    print()

    # Rejection sampling
    t_start = timer()
    print("Rejection sampling")
    pd = rejection_sampling("M", {"I1": T, "I2": T}, matchBN, ordre_anc_matchBN, N=1000)
    t_end = timer()
    print("P(M|I1=True, I2=True)=", pd.show())
    print("P(M=True|I1=True, I2=True)=", pd[T])
    print("Temps per rejection sampling:", t_end - t_start, "segons.")
    print()

    # Variable elimination
    print("Variable elimination")
    t_start = timer()
    pd = variable_elimination("M", {"I1": T, "I2": T}, matchBN, elimination_order)
    t_end = timer()
    print("P(M|I1=True, I2=True)=", pd.show())
    print("P(M=True|I1=True, I2=True)=", pd[T])
    print("Temps per variable elimination:", t_end - t_start, "segons.")
    print()

    # Weighted sampling
    print("Weighted sampling")
    t_start = timer()
    pd = weighted_sampling("M", {"I1": T, "I2": T}, matchBN, ordre_anc_matchBN, N=1500)
    t_end = timer()
    print("P(M|I1=True, I2=True)=", pd.show())
    print("P(M=True|I1=True, I2=True)=", pd[T])
    print("Temps per weighted sampling:", t_end - t_start, "segons.")
    print()

    # Recuperem les dades del dataset d'una manera apropiada per treballar amb BayesNet
    dic = obtenir_valors_per_variables(["I", "J", "K"], problema["data"][1,d:])
    print(problema["data"][1,d:],"es transforma en",dic)

    # Fem servir el prefix per canviar el nom de les variables
    dic1 = obtenir_valors_per_variables(["I", "J", "K"], problema["data"][3,d:], sufix="a")
    dic2 = obtenir_valors_per_variables(["I", "J", "K"], problema["data"][4,d:], sufix="b")
    dic = dic1 | dic2
    print(problema["data"][3,d:]," i ", problema["data"][4,d:],"es combinen per formar",dic)
        # Obtenim les dades de convergència de cada algorisme
    
    convergencies = obtenir_convergencia(algorismes, problema1, beam_size=beam_size, iteracions=n_iterations,tolerancia=tolerancia)

    # Grafiquem la convergència de tots els algorismes
    for nom, convergencia in convergencies.items():
        convergencia = convergencia[:100]
        plt.plot(convergencia, label=nom)

    plt.xlabel('Iteracions')
    plt.ylabel('Millora de la funció objectiu')
    plt.title('Convergència dels Algorismes')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
