import numpy as np
import evaluate_functions as eval_strategy

##### Variables #####
lignes = 6
colonnes = 7
strategy = [0, 0]
currentPlayer = 1
#####################


class bcolors:
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


################ Gestion du jeu ##################

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


# Fonction d'évaluation
def evaluer_position(plateau, joueur):
    global currentPlayer
    score = 0
    # Évaluation des lignes
    for ligne in range(len(plateau)):
        for colonne in range(len(plateau[0]) - 3):
            fenetre = plateau[ligne][colonne:colonne + 4]
            score += eval_strategy.evaluate(strategy[currentPlayer - 1], fenetre, joueur)

    # Évaluation des colonnes
    for colonne in range(len(plateau[0])):
        for ligne in range(len(plateau) - 3):
            fenetre = plateau[ligne:ligne + 4, colonne]
            score += eval_strategy.evaluate(strategy[currentPlayer - 1], fenetre, joueur)

    # Évaluation des diagonales ascendantes
    for ligne in range(len(plateau) - 3):
        for colonne in range(len(plateau[0]) - 3):
            fenetre = [plateau[ligne + i][colonne + i] for i in range(4)]
            score += eval_strategy.evaluate(strategy[currentPlayer - 1], fenetre, joueur)

    # Évaluation des diagonales descendantes
    for ligne in range(3, len(plateau)):
        for colonne in range(len(plateau[0]) - 3):
            fenetre = [plateau[ligne - i][colonne + i] for i in range(4)]
            score += eval_strategy.evaluate(strategy[currentPlayer - 1], fenetre, joueur)

    return score

# Fonction pour obtenir les colonnes valides
def get_colonnes_valides(plateau):
    colonnes_valides = []
    for colonne in range(colonnes):
        if plateau[0][colonne] == 0:
            colonnes_valides.append(colonne)
    return colonnes_valides





def minimax(plateau, profondeur, joueur):
    score, action = joueur_max(plateau, profondeur, joueur)
    return score


def joueur_max(plateau, profondeur, joueur, alpha=None, beta=None):
    colonnes_valides = get_colonnes_valides(plateau)
    if profondeur == 0 or verifie_victoire(plateau, 1) or verifie_victoire(plateau, 2) or len(colonnes_valides) == 0:
        # Si la profondeur maximale est atteinte ou si le jeu est terminé, retourner le score de la position
        score = evaluer_position(plateau, joueur)
        return score, None

    best_score = float("-inf")
    action = None
    for colonne in colonnes_valides:
        nouveau_plateau = plateau.copy()
        placer_jeton(nouveau_plateau, colonne, joueur)
        score, _ = joueur_min(plateau, profondeur - 1, 3 - joueur)
        if score > best_score:
            best_score = score
            action = colonne
        if alpha is not None and beta is not None:
            if best_score >= beta:
                return best_score, action
            alpha = max(alpha, best_score)
    return best_score, action


def joueur_min(plateau, profondeur, joueur, alpha=None, beta=None):
    colonnes_valides = get_colonnes_valides(plateau)
    if profondeur == 0 or verifie_victoire(plateau, 1) or verifie_victoire(plateau, 2) or len(colonnes_valides) == 0:
        # Si la profondeur maximale est atteinte ou si le jeu est terminé, retourner le score de la position
        score = evaluer_position(plateau, 3 - joueur)
        return score, None

    best_score = float("inf")
    action = None
    for colonne in colonnes_valides:
        nouveau_plateau = plateau.copy()
        placer_jeton(nouveau_plateau, colonne, joueur)
        score, _ = joueur_max(plateau, profondeur - 1, 3 - joueur)
        if score < best_score:
            best_score = score
            action = colonne
        if alpha is not None and beta is not None:
            if best_score <= alpha:
                return best_score, action
            beta = min(beta, best_score)

    return best_score, action


#######################################################################################

def alphabeta(plateau, profondeur, joueur):
    score, action = joueur_max(plateau, profondeur, joueur, float("-inf"), float("inf"))
    return score


def get_best_move(plateau, profondeur, joueur, algo):
    colonnes_valides = get_colonnes_valides(plateau)
    meilleur_score = float("-inf")
    meilleur_coup = colonnes_valides[0]
    for colonne in colonnes_valides:
        nouveau_plateau = plateau.copy()
        placer_jeton(nouveau_plateau, colonne, joueur)
        match algo:
            case 1:
                score = minimax(nouveau_plateau, profondeur, joueur)
            case 2:
                score = alphabeta(nouveau_plateau, profondeur, joueur)
            case 3:
                print("NON IMPLEMENT nullos")
        if score > meilleur_score:
            meilleur_score = score
            meilleur_coup = colonne
    return meilleur_coup



#################### Gestion affichage ############################
def agentNameFromIndex(playerNumber, agents, layers):
    playerNumber = playerNumber - 1
    match agents[playerNumber]:
        case 0:
            name = "Joueur"
        case 1:
            name = "Minimax profondeur " + str(layers[playerNumber])
        case 2:
            name = "Alpha-Beta profondeur " + str(layers[playerNumber])
        case 3:
            name = "MCTS profondeur " + str(layers[playerNumber])
    return name


def play(agents, layers):
    plateau = creer_plateau(lignes, colonnes)
    global currentPlayer
    joueur = 1
    fin_partie = False

    while not fin_partie:
        print(agentNameFromIndex(joueur, agents, layers))
        afficher_plateau(plateau)

        if agents[joueur - 1] == 0:
            colonne = int(input("Joueur " + str(joueur) + ", choisissez une colonne : ")) - 1
        else:
            colonne = get_best_move(plateau, layers[joueur - 1], joueur, agents[joueur - 1])

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
                currentPlayer = joueur
        else:
            print("La colonne est pleine. Veuillez choisir une autre colonne.")


def main():
    print("Connect 4")
    print("_________________________")
    print("0: Joueur physique\n1: Minimax\n2: AlphaBeta\n3: MCTS")
    print("_________________________")
    agent1 = int(input("Type de joueur 1 : "))
    agent2 = int(input("Type de joueur 2 : "))

    if agent1 == 1 or agent1 == 2 or agent1 == 3:
        layer1 = int(input("Profondeur IA 1: "))
        print("0: Agressive |  1: Modérée |  2: Défensive")
        strategy[0] = int(input("Stratégie IA 1: "))
    if agent2 == 1 or agent2 == 2 or agent2 == 3:
        layer2 = int(input("Profondeur IA 2: "))
        print("0: Agressive |  1: Modérée |  2: Défensive")
        strategy[1] = int(input("Stratégie IA 2: "))

    play([agent1, agent2], [layer1, layer2])


# main()
strategy = [1, 2]
play([2, 2], [1, 3])

