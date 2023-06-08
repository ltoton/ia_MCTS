import math

import numpy as np


class bcolors:
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


# Fonction pour créer le plateau de jeu vide
def creer_plateau(lignes, colonnes):
    return np.zeros((lignes, colonnes), dtype=int)


# Fonction pour afficher la grille de jeu
def afficher_plateau(grille):
    for ligne in grille:
        ligne_affichage = "|"
        for cellule in ligne:
            if cellule == 1:
                ligne_affichage += bcolors.WARNING + "X" + bcolors.ENDC
            elif cellule == 2:
                ligne_affichage += bcolors.FAIL + "0" + bcolors.ENDC
            else:
                ligne_affichage += " "
            ligne_affichage += "|"
        print(ligne_affichage)
    print("\n")


# Fonction pour placer un jeton dans une colonne
def placer_jeton(plateau, colonne, joueur):
    for ligne in range(len(plateau) - 1, -1, -1):
        if plateau[ligne][colonne] == 0:
            plateau[ligne][colonne] = joueur
            return True
    return False


# Fonction pour vérifier si un joueur a gagné
def verifie_victoire(plateau, joueur):
    # Vérification des lignes
    for ligne in range(len(plateau)):
        for colonne in range(len(plateau[0]) - 3):
            if plateau[ligne][colonne] == joueur and plateau[ligne][colonne + 1] == joueur and plateau[ligne][
                colonne + 2] == joueur and plateau[ligne][colonne + 3] == joueur:
                return True

    # Vérification des colonnes
    for colonne in range(len(plateau[0])):
        for ligne in range(len(plateau) - 3):
            if plateau[ligne][colonne] == joueur and plateau[ligne + 1][colonne] == joueur and plateau[ligne + 2][
                colonne] == joueur and plateau[ligne + 3][colonne] == joueur:
                return True

    # Vérification des diagonales ascendantes
    for ligne in range(len(plateau) - 3):
        for colonne in range(len(plateau[0]) - 3):
            if plateau[ligne][colonne] == joueur and plateau[ligne + 1][colonne + 1] == joueur and plateau[ligne + 2][
                colonne + 2] == joueur and plateau[ligne + 3][colonne + 3] == joueur:
                return True

    # Vérification des diagonales descendantes
    for ligne in range(3, len(plateau)):
        for colonne in range(len(plateau[0]) - 3):
            if plateau[ligne][colonne] == joueur and plateau[ligne - 1][colonne + 1] == joueur and plateau[ligne - 2][
                colonne + 2] == joueur and plateau[ligne - 3][colonne + 3] == joueur:
                return True

    return False


# Fonction pour exécuter le jeu
def jouer_puissance4():
    lignes = 6
    colonnes = 7
    plateau = creer_plateau(lignes, colonnes)
    joueur = 1
    fin_partie = False

    while not fin_partie:
        afficher_plateau(plateau)

        # Demander au joueur courant de choisir une colonne
        colonne = int(input("Joueur " + str(joueur) + ", choisissez une colonne : "))

        # Placer le jeton dans la colonne choisie
        if placer_jeton(plateau, colonne, joueur):
            print(evaluer_position(plateau, joueur))
            # Vérifier si le joueur courant a gagné
            if verifie_victoire(plateau, joueur):
                afficher_plateau(plateau)
                print("Joueur " + str(joueur) + " a gagné !")
                fin_partie = True
            else:
                # Passer au joueur suivant
                joueur = 3 - joueur  # Alternance entre 1 et 2
        else:
            print("La colonne est pleine. Veuillez choisir une autre colonne.")


# Fonction d'évaluation
def evaluer_position(plateau, joueur):
    score = 0

    # Évaluation des lignes
    for ligne in range(len(plateau)):
        for colonne in range(len(plateau[0]) - 3):
            fenetre = plateau[ligne][colonne:colonne + 4]
            score += evaluer_fenetre(fenetre, joueur)

    # Évaluation des colonnes
    for colonne in range(len(plateau[0])):
        for ligne in range(len(plateau) - 3):
            fenetre = plateau[ligne:ligne + 4, colonne]
            score += evaluer_fenetre(fenetre, joueur)

    # Évaluation des diagonales ascendantes
    for ligne in range(len(plateau) - 3):
        for colonne in range(len(plateau[0]) - 3):
            fenetre = [plateau[ligne + i][colonne + i] for i in range(4)]
            score += evaluer_fenetre(fenetre, joueur)

    # Évaluation des diagonales descendantes
    for ligne in range(3, len(plateau)):
        for colonne in range(len(plateau[0]) - 3):
            fenetre = [plateau[ligne - i][colonne + i] for i in range(4)]
            score += evaluer_fenetre(fenetre, joueur)

    return score


# Fonction d'évaluation d'une fenêtre de 4 jetons
def evaluer_fenetre(fenetre, joueur):
    score = 0
    adversaire = 3 - joueur

    if np.count_nonzero(fenetre == joueur) == 4:
        score += 100
    elif np.count_nonzero(fenetre == joueur) == 3 and np.count_nonzero(fenetre == 0) == 1:
        score += 5
    elif np.count_nonzero(fenetre == joueur) == 2 and np.count_nonzero(fenetre == 0) == 2:
        score += 2

    if np.count_nonzero(fenetre == adversaire) == 3 and np.count_nonzero(fenetre == 0) == 1:
        score -= 4

    return score


def minimax(plateau, profondeur, joueur):
    eval, action = joueur_max(plateau, profondeur, joueur)
    return action


def joueur_max(n, p, joueur):
    if verifie_victoire(n, joueur) or p == 0:
        return evaluer_position(n, 2), None
    u = -math.inf
    alpha = None
    for c in range(7):
        if placer_jeton(n, c, joueur):
            v, _ = joueur_min(n, p - 1, joueur)
            if v > u:
                u = v
                alpha = c
            n[c][len(n[c]) - 1] = 0
    return u, alpha


def joueur_min(n, p, joueur):
    if verifie_victoire(n, joueur) or p == 0:
        return evaluer_position(n, 1), None
    u = math.inf
    alpha = None
    for c in range(7):
        if placer_jeton(n, c, joueur):
            v, _ = joueur_max(n, p - 1, joueur)
            if v < u:
                u = v
                alpha = c
            n[c][len(n[c]) - 1] = 0
    return u, alpha


def jouer_puissance4_avec_IA():
    lignes = 6
    colonnes = 7
    plateau = creer_plateau(lignes, colonnes)
    joueur = 1
    fin_partie = False

    while not fin_partie:
        afficher_plateau(plateau)

        if joueur == 1:
            colonne = obtenir_meilleur_coup(plateau, 3)
        else:
            colonne = obtenir_meilleur_coup(plateau, 5)  # Profondeur de recherche pour l'IA

        if placer_jeton(plateau, colonne, joueur):
            if verifie_victoire(plateau, joueur):
                afficher_plateau(plateau)
                print("Joueur " + str(joueur) + " a gagné !")
                fin_partie = True
            elif len(get_colonnes_valides(plateau)) == 0:
                afficher_plateau(plateau)
                print("Match nul !")
                fin_partie = True
            else:
                joueur = 3 - joueur
        else:
            print("La colonne est pleine. Veuillez choisir une autre colonne.")


def obtenir_meilleur_coup(plateau, profondeur):
    colonnes_valides = get_colonnes_valides(plateau)
    meilleur_score = float("-inf")
    meilleur_coup = colonnes_valides[0]
    for colonne in colonnes_valides:
        nouveau_plateau = plateau.copy()
        placer_jeton(nouveau_plateau, colonne, 1)
        score = minimax(nouveau_plateau, profondeur - 1, 2)
        if score > meilleur_score:
            meilleur_score = score
            meilleur_coup = colonne
    return meilleur_coup


# Fonction pour obtenir les colonnes valides
def get_colonnes_valides(plateau):
    colonnes_valides = []
    for colonne in range(len(plateau[0])):
        if plateau[0][colonne] == 0:
            colonnes_valides.append(colonne)
    return colonnes_valides


def play():
    print("Choix du mode de jeu :", "- J vs J : 1", "- J vs IA : 2", "- IA vs IA")
    choix = int(input("Votre choix : "))
    match choix:
        case "0":
            jouer_puissance4()
        case "1":
            jouer_puissance4_avec_IA()
        case "2":
            print("NON IMPLEMENT nullos")
        case _:
            print("Ce choix n'existe pas, recommencez..........")


# Lancer le jeu
# jouer_puissance4()

# Lancer le jeu avec l'IA utilisant Alpha-Beta
# jouer_puissance4_avec_IA_alpha_beta()

# Lancer le jeu avec l'IA
jouer_puissance4_avec_IA()
