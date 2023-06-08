# Fonction Alpha-Beta
def alphabeta(plateau, profondeur, joueur_max, alpha, beta):
    if profondeur == 0 or verifie_victoire(plateau, 1) or verifie_victoire(plateau, 2) or len(
            get_colonnes_valides(plateau)) == 0:
        # Si la profondeur maximale est atteinte ou si le jeu est terminé, retourner le score de la position
        score = evaluer_position(plateau, joueur_max)
        return score

    if joueur_max == 1:
        # Joueur Max (IA)
        meilleur_score = float("-inf")
        for colonne in get_colonnes_valides(plateau):
            nouveau_plateau = plateau.copy()
            placer_jeton(nouveau_plateau, colonne, joueur_max)
            score = alphabeta(nouveau_plateau, profondeur - 1, 2, alpha, beta)
            meilleur_score = max(meilleur_score, score)
            alpha = max(alpha, meilleur_score)
            if beta <= alpha:
                break
        return meilleur_score

    else:
        # Joueur Min (adversaire)
        meilleur_score = float("inf")
        for colonne in get_colonnes_valides(plateau):
            nouveau_plateau = plateau.copy()
            placer_jeton(nouveau_plateau, colonne, joueur_max)
            score = alphabeta(nouveau_plateau, profondeur - 1, 1, alpha, beta)
            meilleur_score = min(meilleur_score, score)
            beta = min(beta, meilleur_score)
            if beta <= alpha:
                break
        return meilleur_score


# Fonction pour obtenir le meilleur coup pour l'IA
def obtenir_meilleur_coup_alpha_beta(plateau, profondeur):
    colonnes_valides = get_colonnes_valides(plateau)
    meilleur_score = float("-inf")
    meilleur_coup = colonnes_valides[0]
    for colonne in colonnes_valides:
        nouveau_plateau = plateau.copy()
        placer_jeton(nouveau_plateau, colonne, 1)
        score = alphabeta(nouveau_plateau, profondeur - 1, 2, float("-inf"), float("inf"))
        if score > meilleur_score:
            meilleur_score = score
            meilleur_coup = colonne
    return meilleur_coup


# ... Code précédent pour jouer_puissance4() ...

# Lancer le jeu avec l'IA utilisant Alpha-Beta
def jouer_puissance4_avec_IA_alpha_beta():
    lignes = 6
    colonnes = 7
    plateau = creer_plateau(lignes, colonnes)
    joueur = 1
    fin_partie = False

    while not fin_partie:
        afficher_plateau(plateau)

        if joueur == 1:
            colonne = int(input("Joueur " + str(joueur) + ", choisissez une colonne : ")) - 1
        else:
            colonne = obtenir_meilleur_coup_alpha_beta(plateau, 4)  # Profondeur de recherche pour l'IA

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
