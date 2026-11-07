# DMLAKE

Projet perso de bathymétrie : estimer la profondeur des lacs naturels à partir du relief qui les entoure , sans mesure directe dans le lac. Tout le déroulé, jour par jour, avec les erreurs et ce que j'en ai tiré, est dans MONHISTOIRE . Les dates citées dans les commentaires du code renvoient aux entrées de ce journal.

## Les données

Ducôté suisse, la bathymétrie swissBATHY3D et le MNT swissALTI3D (2 m) de swisstopo, pour 9 lacs. Bodensee et le Léman sont exclus, parce que le MNT (modèle numérique de terrain) suisse ne couvre pas la partie étrangère ( car ces lacs se situent à la frontière de deux pays). Côté français, les relevés bathymétriques d'EauFrance Adour-Garonne et le MNT RGE ALTI (5 m) de l'IGN, pour 98 lacs. Tout est téléchargé par les scripts, rien n'est stocké dans le dépôt à part les figures dans outputs/.

## Resultats synthétiques

L'étude retrouve l'allure générale du fond des lacs, avec une erreur d'environ un cinquième de la profondeur maximale du lac. En revanche, il distingue mal les zones profondes et peu profondes d'un lac inconnu. Sur la moitié des lacs, il ne fait pas mieux que si l'on donnait partout la profondeur moyenne du lac. Il donne aussi une profondeur relative, pas des mètres. Il faut donc savoir que les projets de recherches en lien avec cet exercice ont surtout pour but de retrouver la profondeur max des lacs et pas de connaitre leur forme car cela est encore plus dur. 
Cela pourrait probablement être un projet de recherche poussé mais dans cet exercice, on a pu clairement voir la difficulté de l'objectif.


## Relancer le projet

pip install -r requirements.txt
python src/extraire_tous_les_lacs.py
python src/evaluation_finale.py
python src/evaluation_finale.py --test-final


L'extraction est longue, elle télécharge tous les lacs (plusieurs Go). Pour un seul lac : `python src/extract_features.py lacdejoux` ou `python src/extract_features_france.py L80_ASCONIT_2012`. La liste des lacs est dans src/lacs.py.

## Les scripts

- lacs.py : la liste des lacs suisses et des dossiers EauFrance
- fetch_dem.py et fetch_dem_france.py : téléchargement de la bathymétrie et du MNT
- masque_eau.py : le masque d'eau (filtre de Sobel), le même pour les deux pays
- extract_features.py et extract_features_france.py : les variables de relief autour de chaque point (8 directions, 4 distances)
- extraire_tous_les_lacs.py : relance l'extraction de tous les lacs, saute ceux déjà faits
- train_model.py : toutes les expériences, dans l'ordre du journal
- evaluation_finale.py : la config retenue, le leave-one-lake-out sur les lacs de développement, puis le test final (outputs/test_final_r2.png)
