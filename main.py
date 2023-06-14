import random

import numpy as np
import evaluate_functions as eval_strategy
import copy
import math

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


################################## MCTS ###############################################
dx = [1, 1, 1, 0]
dy = [1, 0, -1, 1]


class Board(object):

    def __init__(self, board, last_move=[None, None]):
        self.board = board
        self.last_move = last_move

    def tryMove(self, move):
        if (move < 0 or move > 7 or self.board[0][move] != 0):
            return -1;

        for i in range(len(self.board)):
            if (self.board[i][move] != 0):
                return i - 1
        return len(self.board) - 1

    def terminal(self):
        for i in range(len(self.board[0])):
            if (self.board[0][i] == 0):
                return False
        return True

    def legal_moves(self):
        legal = []
        for i in range(len(self.board[0])):
            if (self.board[0][i] == 0):
                legal.append(i)

        return legal

    def next_state(self, turn):
        aux = copy.deepcopy(self)
        moves = aux.legal_moves()
        if len(moves) > 0:
            ind = random.randint(0, len(moves) - 1)
            row = aux.tryMove(moves[ind])
            aux.board[row][moves[ind]] = turn
            aux.last_move = [row, moves[ind]]
        return aux

    def winner(self):
        x = self.last_move[0]
        y = self.last_move[1]

        if x == None:
            return 0

        for d in range(4):

            h_counter = 0
            c_counter = 0

            for k in range(-3, 4):

                u = x + k * dx[d]
                v = y + k * dy[d]

                if u < 0 or u >= 6:
                    continue

                if v < 0 or v >= 7:
                    continue

                if self.board[u][v] == -1:
                    c_counter = 0
                    h_counter += 1
                elif self.board[u][v] == 1:
                    h_counter = 0
                    c_counter += 1
                else:
                    h_counter = 0
                    c_counter = 0

                if h_counter == 4:
                    return -1

                if c_counter == 4:
                    return 1

        return 0


class Node():
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.children_move = []
        self.parent = parent

    def addChild(self, child_state, move):
        child = Node(child_state, self)
        self.children.append(child)
        self.children_move.append(move)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_explored(self):
        if len(self.children) == len(self.state.legal_moves()):
            return True
        return False


def MTCS(maxIter, root, factor, player):
    for inter in range(maxIter):
        front, turn = treePolicy(root, player, factor)
        reward = defaultPolicy(front.state, turn)
        backup(front, reward, turn)

    ans = bestChild(root, 0)
    return ans


def treePolicy(node, turn, factor):
    while node.state.terminal() == False and node.state.winner() == 0:
        if node.fully_explored() == False:
            return expand(node, turn), -turn
        else:
            node = bestChild(node, factor)
            turn *= -1
    return node, turn


def expand(node, turn):
    tried_children_move = [m for m in node.children_move]
    possible_moves = node.state.legal_moves()

    for move in possible_moves:
        if move not in tried_children_move:
            row = node.state.tryMove(move)
            new_state = copy.deepcopy(node.state)
            new_state.board[row][move] = turn
            new_state.last_move = [row, move]
            break

    node.addChild(new_state, move)
    return node.children[-1]


def bestChild(node, factor):
    bestscore = -10000000.0
    bestChildren = []
    for c in node.children:
        exploit = c.reward / c.visits
        explore = math.sqrt(math.log(2.0 * node.visits) / float(c.visits))
        score = exploit + factor * explore
        if score == bestscore:
            bestChildren.append(c)
        if score > bestscore:
            bestChildren = [c]
            bestscore = score
    return random.choice(bestChildren)


def defaultPolicy(state, turn):
    while state.terminal() == False and state.winner() == 0:
        state = state.next_state(turn)
        turn *= -1
    return state.winner()


def backup(node, reward, turn):
    while node is not None:
        node.visits += 1
        node.reward -= turn * reward
        node = node.parent
        turn *= -1
    return


######################################## Gestion du jeu ###############################################

# Fonction pour créer le plateau de jeu vide
def creer_plateau():
    return np.zeros((lignes, colonnes), dtype=int)


def create_board():
    board = []
    for i in range(6):
        row = []
        for j in range(7):
            row.append(0)
        board.append(row)
    return board


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
            fenetre = np.array(
                [plateau[ligne][colonne], plateau[ligne + 1][colonne + 1], plateau[ligne + 2][colonne + 2],
                 plateau[ligne + 3][colonne + 3]])
            score += eval_strategy.evaluate(strategy[currentPlayer - 1], fenetre, joueur)

    # Évaluation des diagonales descendantes
    for ligne in range(3, len(plateau)):
        for colonne in range(len(plateau[0]) - 3):
            fenetre = np.array(
                [plateau[ligne][colonne], plateau[ligne - 1][colonne + 1], plateau[ligne - 2][colonne + 2],
                 plateau[ligne - 3][colonne + 3]])
            score += eval_strategy.evaluate(strategy[currentPlayer - 1], fenetre, joueur)

    return score


# Fonction pour obtenir les colonnes valides
def get_colonnes_valides(plateau):
    colonnes_valides = []
    for colonne in range(colonnes):
        if plateau[0][colonne] == 0:
            colonnes_valides.append(colonne)
    return colonnes_valides


def get_board(plateau):
    board = []
    for i in range(6):
        row = []
        for j in range(7):
            if plateau[i][j] == 0:
                row.append(0)
            if plateau[i][j] == 1:
                row.append(1)
            if plateau[i][j] == 2:
                row.append(-1)
        board.append(row)
    return board


def get_plateau(board):
    plateau = []
    for i in range(6):
        row = []
        for j in range(7):
            if board[i][j] == 0:
                row.append(0)
            if board[i][j] == 1:
                row.append(1)
            if board[i][j] == -1:
                row.append(2)
        plateau.append(np.asarray(row))
    return np.asarray(plateau)


################################################## MINIMAX & ALPHABETA #############################################

def minimax(plateau, profondeur, joueur):
    score, action = joueur_max(plateau, profondeur, joueur)
    return score


def alphabeta(plateau, profondeur, joueur):
    score, action = joueur_max(plateau, profondeur, joueur, float("-inf"), float("inf"))
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


def get_mcts_move(plateau, joueur):
    node = Node(Board(get_board(plateau)))
    meilleur_coup = MTCS(3000, node, 2.0, 1 if joueur == 1 else -1)
    plateau = get_plateau(meilleur_coup.state.board)
    return plateau


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
        if score > meilleur_score:
            meilleur_score = score
            meilleur_coup = colonne
    return meilleur_coup


def end_game(plateau):
    if verifie_victoire(plateau, 1):
        return True
    elif verifie_victoire(plateau, 2):
        return True
    elif grille_pleine(plateau):
        return True
    else:
        return False


def grille_pleine(plateau):
    return np.all(plateau != 0)


#################### Gestion affichage ############################
def agentNameFromIndex(playerNumber, agents, layers):
    print("Joueur", playerNumber, ":")
    playerNumber = playerNumber - 1
    agent = agents[playerNumber]
    match agent:
        case 0:
            name = "Joueur"
        case 1:
            name = "Minimax"
        case 2:
            name = "Alpha-Beta"
        case 3:
            name = "MCTS"
    if (agent == 1 or agent == 2):
        name += " profondeur " + str(layers[playerNumber]) + " | Stratégie " + eval_strategy.get_strategy_name(
            strategy[playerNumber])

    return name


def play(agents, layers):
    plateau = creer_plateau()
    global currentPlayer
    player = 1
    fin_partie = False

    while not fin_partie:
        if agents[player - 1] == 0:
            colonne = int(input("Joueur " + str(player) + ", choisissez une colonne : ")) - 1
            if not placer_jeton(plateau, colonne, player):
                move = False
            else:
                move = True
        else:
            if agents[player - 1] == 1 or agents[player - 1] == 2:
                colonne = get_best_move(plateau, layers[player - 1], player, agents[player - 1])
                placer_jeton(plateau, colonne, player)
            else:
                plateau = get_mcts_move(plateau, player)
            move = True

        print(agentNameFromIndex(player, agents, layers))
        afficher_plateau(plateau)

        if move:
            if verifie_victoire(plateau, player):
                print("Joueur " + str(player) + " a gagné !")
                fin_partie = True
            elif len(get_colonnes_valides(plateau)) == 0:
                print("Match nul !")
                fin_partie = True
            else:
                player = 3 - player
                currentPlayer = player
        else:
            print("La colonne est pleine. Veuillez choisir une autre colonne.")


def main():
    print("Connect 4")
    print("_________________________")
    print("0: Joueur\n1: Minimax\n2: AlphaBeta\n3: MCTS")
    print("_________________________")
    agent1 = int(input("Type de joueur 1 : "))
    agent2 = int(input("Type de joueur 2 : "))
    layer1 = 0
    layer2 = 0

    if agent1 == 1 or agent1 == 2:
        layer1 = int(input("Profondeur IA 1: "))
        print("0: Agressive |  1: Modérée |  2: Défensive")
        strategy[0] = int(input("Stratégie IA 1: "))
    if agent2 == 1 or agent2 == 2:
        layer2 = int(input("Profondeur IA 2: "))
        print("0: Agressive |  1: Modérée |  2: Défensive")
        strategy[1] = int(input("Stratégie IA 2: "))

    play([agent1, agent2], [layer1, layer2])


# Stratégie - 0: Agressive |  1: Modérée |  2: Défensive
# Premier paramètre
# 0: Joueur
# 1: Minimax
# 2: AlphaBeta
# 3: MCTS
# Deuxième paramètre = nombre de couche

# main()
strategy = [2, 2]
play([0, 3], [3, 1])
