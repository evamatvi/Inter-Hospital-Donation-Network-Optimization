from bn import BayesNet

# Per simplicar la notació, redefinim True i False com T i F
T, F = True, False

criticalBN = BayesNet([
    ('I', '', 0.7),
    ('J', 'I', {T: 0.8, F: 0.3}),
    ('C', 'I J',
     {(T, T): 0.95, (T, F): 0.64, (F, T): 0.29, (F, F): 0.01}),
    ('K', 'C', {T: 0.7, F: 0.1})
])

# Variables auxiliars criticalBN
descriptors_criticalBN = ["I", "J", "K"]
var_depenent_criticalBN = "C"
ordre_anc_criticalBN = [0,1,2,3]

matchBN = BayesNet([
    ('I1', '', 0.7),
    ('J1', 'I1', {T: 0.8, F: 0.3}),
    ('C1', 'I1 J1',
     {(T, T): 0.95, (T, F): 0.64, (F, T): 0.29, (F, F): 0.01}),
    ('K1', 'C1', {T: 0.7, F: 0.1}),
    
    ('I2', '', 0.7),
    ('J2', 'I2', {T: 0.8, F: 0.3}),
    ('C2', 'I2 J2',
     {(T, T): 0.95, (T, F): 0.64, (F, T): 0.29, (F, F): 0.01}),
    ('K2', 'C2', {T: 0.7, F: 0.1}),
    
    ('M', 'C1 C2',
     {(T, T): 0.95, (T, F): 0.1, (F, T): 0.1, (F, F): 0.95})
])


# Variables auxiliares matchBN
descriptors_matchBN2 = ["I1", "J1", "K1", "I2", "J2", "K2"]
var_depenent_matchBN2 = "M"

def get_elimination_order(bn, X):
    """
    Calcula un ordre d'eliminació òptim per variable elimination.
    Prioritza eliminar primer les variables menys connectades.
    
    :param bn: La xarxa Bayesiana (BayesNet).
    :param X: Variable objectiu que no s'ha d’eliminar.
    :return: Llista amb l’ordre d'eliminació (noms de variables).
    """
    dependency_counts = {
        var: sum(var in bn.variable_node(v).parents for v in bn.variables)
        for var in bn.variables
    }

    # Ordenar de menys connexions a més connexions, excloent X
    elimination_order = sorted(
        [var for var in bn.variables if var != X],
        key=lambda v: dependency_counts[v]
    )

    return elimination_order

elimination_order = get_elimination_order(matchBN, "M") # ['K1', 'K2', 'J1', 'J2', 'I1', 'C1', 'I2', 'C2']


ordre_anc_matchBN= ["I1", "J1", "C1", "K1", "I2", "J2", "C2", "K2", "M"]
ordre_anc_matchBN = [matchBN.variables.index(var) for var in ordre_anc_matchBN] # [0, 1, 2, 3, 4, 5, 6, 7, 8]
