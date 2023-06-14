import numpy as np

def evaluate(index, fenetre, joueur):
    match index:
        case 0:
            score = aggressive(fenetre, joueur)
        case 1:
            score = moderate(fenetre, joueur)
        case 2:
            score = defensive(fenetre, joueur)

    return score

def get_strategy_name(index):
    match index:
        case 0:
            name = "Agressive"
        case 1:
            name = "Modérée"
        case 2:
            name = "Défensive"
    return name

# Fonction d'évaluation d'une fenêtre de 4 jetons
def aggressive(fenetre, joueur):
    score = 0
    if np.count_nonzero(fenetre == joueur) == 4:
        score += 1000
    elif np.count_nonzero(fenetre == joueur) == 3 and np.count_nonzero(fenetre == 0) == 1:
        score += 50
    elif np.count_nonzero(fenetre == joueur) == 2 and np.count_nonzero(fenetre == 0) == 2:
        score += 5
    elif np.count_nonzero(fenetre == joueur) == 1 and np.count_nonzero(fenetre == 0) == 3:
        score += 1

    return score


# Fonction d'évaluation d'une fenêtre de 4 jetons
def moderate(fenetre, joueur):
    score = 0
    adversaire = 3 - joueur
    if np.count_nonzero(fenetre == joueur) == 4:
        score += 1000
    elif np.count_nonzero(fenetre == joueur) == 3 and np.count_nonzero(fenetre == 0) == 1:
        score += 5
    elif np.count_nonzero(fenetre == joueur) == 2 and np.count_nonzero(fenetre == 0) == 2:
        score += 2

    if np.count_nonzero(fenetre == adversaire) == 3 and np.count_nonzero(fenetre == 0) == 1:
        score -= 100
    return score

# Fonction d'évaluation d'une fenêtre de 4 jetons
def defensive(fenetre, joueur):
    score = 0
    adversaire = 3 - joueur

    if np.count_nonzero(fenetre == joueur) == 4:
        score += 1000
    elif np.count_nonzero(fenetre == joueur) == 3 and np.count_nonzero(fenetre == 0) == 1:
        score += 5

    if np.count_nonzero(fenetre == adversaire) == 2 and np.count_nonzero(fenetre == 0) == 2:
        score -= 3
    if np.count_nonzero(fenetre == adversaire) == 3 and np.count_nonzero(fenetre == 0) == 1:
        score -= 100

    return score