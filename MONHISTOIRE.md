05/07/2026

Bonjour,

Après le projet sur les moteurs de la NASA, je commence un deuxième projet pendant mes vacances, cette fois plus proche de ma formation en géomatique. Comme pour le premier, je raconte ici jour par jour ce que je fais, ce qui marche et ce qui ne marche pas, pour pouvoir retrouver plus tard comment j'ai fait et pourquoi.

Ce nouveau projet est orienté bathymétrie, on va essayer d'estimer la profondeur des lacs naturels à partir de leur morphologie environnante, sans mesure directe. Les lacs naturels n'ont pas d'infrastructure de référence comme les barrages, donc l'idée c'est de prédire la profondeur à partir du relief autour (pente, distance au rivage...).

Pour ce premier jour, le but est de mettre en place le projet et voir si j'arrive à récupérer un jeu de données bathymétriques réel et à le visualiser.

J'ai trouvé deux sources ouvertes intéressantes : swissBATHY3D/swissALTI3D (relevés bathymétriques et MNT suisses) et EauFrance (relevés bathymétriques Adour-Garonne). J'ai testé sur le Lac de Joux, un petit lac suisse, pour aller vite. Le téléchargement et le chargement des points fonctionnent ( environ 9.5 millions de points sur ce lac), et j'ai sorti une première visu du nuage de points coloré par profondeur, la forme du lac se voit bien, ça confirme que les données sont exploitables.

06/07/2026

Avant de pouvoir extraire des features de relief (pente, distance au rivage...), il faut le MNT autour du lac, dans le même référentiel spatial que la bathymétrie. J'ai repéré les cellules de la grille kilométrique couvertes par la bathy du Lac de Joux, ajouté une couronne d'1km pour déborder sur le rivage, et interrogé l'API STAC de swisstopo pour récupérer les tuiles swissALTI3D 2m correspondantes (61 tuiles). Après mosaïquage, j'ai superposé le MNT et le nuage de points bathy pour vérifier l'alignement et le contour du lac colle exactement à la cuvette du relief (altitude MNT ~1004m au niveau des points bathy, contre 971-1041m pour le fond du lac, donc une profondeur max de 32m). Les deux jeux de données sont bien dans le même référentiel (EPSG:2056), on passera à la suite demain.

07/07/2026

Aujourd'hui, on fait une extraction des features de relief. D'abord un masque d'eau (enveloppe convexe des points bathy rasterisée sur la grille du MNT, suffisant pour un lac simple comme Lac de Joux, pas besoin du filtrage Sobel utilisé pour des lacs plus complexes). Puis, pour un échantillon de 2000 points du fond du lac, dans 8 directions (0 à 315°, pas de 45°), recherche du rivage (transition eau à terre dans le masque), puis prolongation du rayon sur terre pour mesurer pente et dénivelé à 4 distances (150/300/600/900m). On fait cela afin de construire les variables d'entrée x de notre futur modèle, pour s'entraîner à la prédiction.
La raison pour laquelle on prolonge le rayon sur terre c'est pour avoir un contexte singulier environnant, par exemple en milieu alpin, on aura un lac avec un fond plus encaissé qu'un lac situé en zone de plaine.
résultat dans : lacdejoux_cross_shore_profiles

09/07/2026

Aujourd'hui, premier modèle de machine learning (`train_model.py`) : je donne au modèle plein de points du lac dont je connais déjà la vraie profondeur mesurée, avec les features de relief extraites le 07/07, et il apprend tout seul la relation entre les deux, pour ensuite deviner la profondeur de points qu'il n'a jamais vus. J'écarte deux colonnes (`angle45` et `angle225`) parce qu'elles ont presque 60% de valeurs manquantes, le lac est allongé pile dans cet axe, du coup les rayons ne trouvent pas la terre à temps dans ces deux directions. J'écarte aussi les flags `_shore_extrapolated` (des indicateurs de fiabilité de la mesure, pas des vraies features de relief). Je garde 75% des points pour entraîner le modèle et 25% de côté pour vérifier s'il devine bien sur des points jamais vus. Le modèle choisi : Random Forest, simple et robuste pour un premier essai, sans réglage compliqué, et adapté ici parce que la relation relief/profondeur n'a pas de raison d'être linéaire.

Premier essai : R²=0.957 (un score entre 0 et 1 qui dit à quel point les prédictions collent aux vraies valeurs, 1 = parfait). Beaucoup trop bon pour un premier essai. En regardant quelle feature comptait le plus, j'ai trouvé le coupable : `angle0_z_DEM_ref` (= profondeur mesurée moins l'altitude du rivage) pesait à elle seule 50.7% de l'importance. Le souci est que l'altitude du rivage change à peine à l'intérieur d'un même lac, donc cette colonne est quasiment une reformulation déguisée de la réponse qu'on essaie de deviner. Le modèle ne devinait pas, il lisait presque directement la solution. Je la retire des features.

Deuxième essai, sans cette triche : RMSE=2.60m (le modèle se trompe en moyenne de 2.60m sur la profondeur), MAE=1.59m (l'erreur "typique"), R²=0.938, toujours très bon, mais honnête cette fois. Ce qui aide le plus à deviner : la distance au rivage dans certaines directions (180°, 270°, 135°).

J'ai séparé mes 75%/25% au hasard, point par point, sur un seul lac (Lac de Joux). Le souci, qui est d'ailleurs compliqué à résoudre : deux points voisins sur le même lac ont presque les mêmes coordonnées, et des features de relief très proches. Du coup mon modèle a peut-être juste "reconnu" des points de test très proches géographiquement de points déjà vus à l'entraînement, plutôt que d'avoir vraiment compris la relation relief et profondeur. Pour être sûr qu'il a appris quelque chose qui généralise, il faudrait le tester sur un lac complètement différent, jamais vu à l'entraînement, ce que je ferai plus tard.

13/07/2026

Repris le point en suspens du 09/07, avec un deuxième lac cette fois : Lungernsee (plus profond que Lac de Joux, 68m contre 31m, mais surface plus petite). Même pipeline relancé dessus sans souci, 2000 points, pas de NaN.

Premier essai en gardant z comme cible, on a un RMSE de 344m, anormal. En fait logique, Lac de Joux est vers 1000m d'altitude et Lungernsee vers 650m, le modèle avait juste appris le niveau du lac d'entraînement, pas une vraie relation relief-profondeur.

Repris avec angle0_z_DEM_ref comme cible au lieu de z. L'erreur redevient raisonnable (30-50m) mais le R² reste négatif dans les deux sens du train/test, donc pire que de prédire la moyenne. Avec seulement 2 lacs le modèle n'a pas de quoi apprendre une relation qui généralise vraiment. Résultat pas terrible mais pas étonnant non plus, il faudra clairement plus de lacs pour voir si ça marche. Je m'arrête là pour aujourd'hui mais on va compléter le test avec davantage d'exemples, les plus exotiques pour entraîner le modèle au mieux.

14/07/2026

Ajouté 3 lacs de plus (aegerisee, baldeggersee, hallwilersee) pour avoir un vrai test à plusieurs lacs plutôt que juste 2. Même pipeline relancé sur chacun, aucun souci au téléchargement/extraction.

En nettoyant les données je me rends compte que l'exclusion angle45/angle225, décidée à l'origine juste à cause de la géométrie de Lac de Joux, ne marche pas du tout pour les autres lacs, il ne reste que 91 points sur 2000 pour aegerisee après dropna, 201 pour hallwilersee, 1054 pour baldeggersee (contre 2000 pour lacdejoux et lungernsee). Chaque lac a sans doute ses propres directions problématiques selon sa forme, pas forcément les mêmes que Lac de Joux.

Test leave-one-lake-out (entraînement sur 4 lacs, test sur le 5e, à tour de rôle), toujours avec angle0_z_DEM_ref comme cible : R²=0.149 pour lacdejoux, -0.751 pour lungernsee, -212 pour aegerisee (mais seulement 91 points de test, donc pas fiable), -0.770 pour baldeggersee, 0.090 pour hallwilersee. Toujours pas de vraie généralisation, un peu mieux que le test à 2 lacs sur certains, pire sur d'autres.

En regardant le taux de NaN par angle sur les 5 lacs, chacun a son propre axe problématique (0°/180° pour hallwilersee, 135°/315° pour aegerisee, etc), lié à sa propre orientation. Pas moyen de choisir un jeu d'angles qui marche pour tout le monde à la fois, il resterait seulement 2 directions sur 8 utilisables partout. Le vrai souci était ailleurs, le rayon de recherche du rivage (fixé à 2500m le 07/07) était juste trop court pour les lacs allongés, peu importe la direction choisie.

Après avoir corrigé en calculant ce rayon dynamiquement par lac (diagonale de l'emprise du lac + marge de 10%) dans extract_features.py, au lieu d'un seuil fixe, refait l'extraction sur les 5 lacs avec ce rayon adaptatif : 0% de NaN partout, sur les 8 angles, plus besoin d'exclure quoi que ce soit.

Relance du leave-one-lake-out avec les jeux de données complets (2000 points par lac au lieu de 91 pour aegerisee) : R²=0.307 pour lacdejoux (était 0.149), -0.159 pour lungernsee (était -0.751), -0.567 pour aegerisee (était -212), 0.140 pour baldeggersee (était -0.770), 0.656 pour hallwilersee (était 0.090). Nette amélioration partout, 3 lacs sur 5 ont maintenant un R² positif. Aegerisee et lungernsee restent négatifs mais beaucoup moins qu'avant. Le vrai problème était bien la couverture des données, pas le modèle.

Pour continuer à tester la généralisation, ajouté les 6 lacs restants du jeu de données swissBATHY3D : bielersee, brienzersee, lacneuchatel, lagomaggiore, bodensee, lacleman. Là aussi découverte importante : bodensee et lacleman sont des lacs transfrontaliers, et swissALTI3D ne couvre que le territoire suisse, après nettoyage il ne reste que 0 point valide sur 2000 pour bodensee, 18 pour lacleman (contre 296 pour lagomaggiore, aussi frontalier avec l'Italie mais moins touché). Exclus ces deux-là du test, gardé lagomaggiore malgré ses 296 points.

Leave-one-lake-out relancé sur ces 9 lacs : résultat globalement pire qu'avec 5, pas mieux. J'ai cherché pourquoi, deux causes plausibles trouvées. En toute honnêteté, cette étape était longue à franchir, je me suis donc aidé de l'IA pour qu'elle m'expose les causes du problème.

Première cause : en regardant l'importance des features, `surface_area` (la surface totale du lac, une seule valeur, la même pour tous les points d'un même lac) écrase tout le reste à plus de 50%. Le problème est que comme cette valeur ne change jamais à l'intérieur d'un même lac, elle finit par se comporter comme un simple "numéro de lac" plutôt qu'une vraie information sur le relief. Le modèle s'appuie dessus pour deviner "quel lac c'est" plutôt que d'apprendre une vraie relation relief et profondeur qui marcherait sur un lac jamais vu.

Deuxième cause, plus grosse : les deux nouveaux lacs les plus profonds (Brienzersee et Bielersee) sont beaucoup plus profonds que les 5 premiers, une erreur de 175m et 126m sur ces deux-là. Le modèle a été entraîné surtout sur des lacs peu profonds (30 à 70m), donc quand on lui demande de deviner un point à 200m de profondeur, il n'a tout simplement jamais vu ça pendant l'entraînement et ne sait pas extrapoler au-delà de ce qu'il connaît. Même limite que celle déjà repérée sur les features de relief, mais cette fois sur l'échelle de profondeur elle-même.

Cela pourrait donc nous pousser à vérifier la singularité de chaque lac avant d'entreprendre des manipulations, ajouter des lacs "au hasard" sans faire attention à leur profondeur ne suffit pas. Deux pistes à creuser une prochaine fois : on peut retirer `surface_area`, ou regrouper les lacs par gamme de profondeur avant d'entraîner. Comme le dernier élément trouvé est personnel je pense que je vais reprendre sur celui-ci.

15/07/2026

Repris sur l'hypothèse de l'échelle de profondeur. Petit souci d'abord : bodensee a fait planter le script de vérification visuelle (mosaïque MNT de 19000x35000 pixels, trop grosse pour matplotlib), corrigé en sous-échantillonnant l'image juste pour l'affichage, l'extraction de features elle-même n'était pas touchée. Comme vu hier, bodensee et lacleman restent exclus : aucun rayon de recherche plus grand n'y changera quoi que ce soit, ce n'est pas un bug mais une vraie limite de couverture de la source de données. On garde donc les 9 autres lacs suisses.

Pour avoir plus de lacs, j'ai codé mon propre pipeline français dans ce repo, sur le même principe que le pipeline suisse : `fetch_dem_france.py` (télécharge la bathymétrie EauFrance et le MNT RGE ALTI de l'IGN autour du lac, via une requête WMS) et `extract_features_france.py` (même méthode cross-shore). Deux différences trouvées en le construisant : le MNT français (RGE ALTI) fait 5m de résolution au lieu de 2m côté suisse, et surtout, `z` côté français est déjà une profondeur mesurée (pas une altitude absolue comme côté suisse), c'est bon signe car ça confirme que le choix de cible `angle0_z_DEM_ref` était le bon réflexe depuis le début. Testé sur 12 lacs (L1 à L92) : 0% de NaN partout, cela semble donc même mieux en termes d'appréhension qualitative de la donnée.

16/07/2026

Leave-one-lake-out relancé sur 21 lacs, on a 7 lacs sur 21 en R² positif. Mais surtout, un résultat qui confirme très clairement l'hypothèse de l'échelle que l'on avait avant : le `L9` (lac français minuscule, profondeur max de 6m) a un R²=-171, alors que son RMSE (13.9m) n'est même pas énorme en absolu, juste que 13.9m d'erreur est absurde pour un lac qui ne fait que 6m de profond. Même phénomène que brienzersee/bielersee (trop profonds) mais à l'autre extrême.

Le cas de L9 donne une idée pour corriger le problème d'échelle que l'on a autant eu : au lieu de faire deviner au modèle la profondeur en mètres (qui varie énormément d'un lac à l'autre, de 6m à 260m), je lui fais deviner un ratio sans unité, profondeur du point / profondeur max de ce lac-là. Un point au fond du lac le plus profond de son lac vaudrait 1, un point tout près du bord vaudrait proche de 0, peu importe que le lac fasse 6m ou 260m de profond, le ratio reste comparable. L'idée est de décorréler "à quel point ce point est profond par rapport à son propre lac" de "quelle est la profondeur totale de ce lac", pour que le modèle n'ait plus à deviner une échelle qu'il n'a jamais vue.

Relance du leave-one-lake-out avec cette cible normalisée à la place de la profondeur brute, nette amélioration, 15 lacs sur 21 ont maintenant un R² positif (contre 7 avant). Le cas L9 (catastrophique avant, R²=-171) remonte à -0.12. Brienzersee (trop profond) passe de -0.83 à +0.07. Ça confirme que la relation "forme du relief autour / position relative en profondeur dans le lac" se généralise plutôt bien d'un lac à l'autre, une fois qu'on retire le problème d'échelle.

20/07/2026

`surface_area` écrase tout et pose souci, surtout cela pose un biais vers les lacs peu profonds. La parade trouvée serait d'utiliser un coefficient isopérimétrique (compacité du lac, indépendant de la taille). On a ajouté ce coefficient isopérimétrique côté suisse. Testé en trois versions : sans rien (R² moyen -0.293), avec `surface_area` en plus (-0.313, pire), isopérimétrique seul sans `surface_area` (-0.184, le meilleur).

Relancé le test complet sur les 21 lacs avec cette config gagnante, cependant seulement 5 lacs sur 21 en R² positif, contre 15 avant. Testé en gardant `surface_area` comme avant mais sur les données françaises actuelles, toujours mauvais. Donc le souci vient des données françaises, pas du choix de feature. On verra comment s'y prendre pour la suite, car cela devient assez compliqué.

Avant, pour les lacs français, on prenait l'enveloppe convexe des points de mesure. Mais sur un lac tout en longueur ou avec des bras qui partent dans plusieurs directions (comme une vallée noyée), cet élastique passe au-dessus de bouts de terre entre les bras du lac. Le modèle croyait que certaines zones de terre étaient de l'eau, ce qui faussait toutes les distances au rivage et les pentes calculées autour. On est passé à une méthode qui regarde directement le relief (le MNT) et détecte les zones plates (l'eau est plate, la terre autour ne l'est pas), beaucoup plus fidèle à la vraie forme du lac, peu importe qu'il soit tordu ou ramifié. C'est donc la méthode du masque de Sobel qui détecte les contours.

Les 4 lacs français qui avaient l'écart le plus flagrant entre l'ancien masque et la vraie forme (L1, L30, L60, L90) sont passés de très négatifs (jusqu'à R²=-1.12 pour L1) à nettement positifs (jusqu'à +0.49). C'était bien le masque le problème, pas les données françaises en elles-mêmes ni le choix de features.

lungernsee, bielersee, lacneuchatel, lagomaggiore et L9 restent négatifs, ce sont exactement les lacs qu'on avait déjà identifiés il y a quelques jours comme ayant un problème différent, leur profondeur sort trop de la plage "normale" des autres lacs (bielersee/lacneuchatel/lagomaggiore sont beaucoup plus profonds que la moyenne, L9 est anormalement peu profond). Le modèle n'a jamais vu d'exemples à cette échelle-là pendant l'entraînement, donc il ne sait pas extrapoler, c'est le problème d'échelle qu'on documente depuis le 16/07, pas un nouveau bug.

01/08/2026

Après une petite pause, on reprend le problème d'échelle d'avant (les lacs hors gamme de profondeur normale restent en R² négatif). Notre but aujourd'hui est de réparer la prédiction en deux étages, un modèle "échelle" (profondeur max du lac, une ligne par lac) et le modèle "forme" (depth_ratio) existant, recombinés ensuite.

Étage 1 avec un Random Forest, échec, catastrophique sur L9 (profondeur prédite 41m contre 5.6m réels, +636%). Cause : un RF ne peut jamais prédire hors de la plage de profondeurs vues à l'entraînement, or L9 est le lac le plus petit, dès qu'on l'exclut, tous les lacs restants font au moins 16m.

Essayé avec une régression log-linéaire à la place (log(surface)->log(profondeur), capable d'extrapoler contrairement au RF) : encore pire (L9 prédit à 100.7m, +1711%). Avec seulement ~20 lacs très hétérogènes (pipelines CH/FR différents, profondeur de 5.6 à 546m soit deux ordres de grandeur), la droite log-log est trop instable en leave-one-lake-out, et une petite erreur de pente devient énorme une fois repassée en mètres.

Autre découverte au passage (colonne "oracle", même modèle de forme mais avec la vraie échelle) : même en donnant la vraie profondeur max, 4 lacs problématiques sur 5 restent en R² négatif. Le problème n'est donc pas seulement l'échelle, la relation forme-relief -> profondeur relative elle-même ne généralise pas vers ces lacs-là. Séparer échelle et forme était une bonne intuition mais ne suffit pas à la quantité de données actuelle (~20 lacs), il faudrait plus de lacs.

03/08/2026 (noté après coup, d'après le commit du jour)

Comme il faut plus de lacs, j'ai élargi la liste des lacs français de 12 à 98 dans `train_model.py`, à partir des dossiers disponibles sur le serveur EauFrance (8 autres exclus parce que leurs données source étaient corrompues). Ça fait 107 lacs en tout avec les 9 suisses. J'en ai profité pour raccourcir les commentaires du code. Les résultats de ce test à 107 lacs, je ne les avais pas notés ce jour-là.
Je dois m'absenter quelques temps donc je reprendrais dès mon retour .

06/09/2026

Bonjour,

Je reprends le projet après trois semaine sans y toucher. En le relisant avec du reculc pour me remettre dans le bain, je vois plusieurs problèmes, surtout dans la façon d'évaluer :
- tous mes choix (le ratio, le coefficient isopérimétrique, le masque de Sobel) ont été faits en regardant le leave-one-lake-out sur les mêmes lacs que ceux qui servent à noter le modèle. Mes scores sont donc trop optimistes, il me faut des lacs jamais regardés, notés une seule fois à la fin.
- la moyenne des R² ne veut pas dire grand-chose quand un lac comme L9 est à -171, elle est écrasée par les extrêmes. Il vaut mieux regarder la médiane, et comparer deux configurations lac par lac.
- la Suisse et la France ne sont pas traitées pareil, enveloppe convexe d'un côté, Sobel de l'autre. Et la cible suisse était z moins l'altitude du premier pixel de terre au nord, qui change d'un point à l'autre (elle contient la hauteur de la berge). En plus, sur un lac sans mesure, ce qui est quand même le but du projet, l'enveloppe des points bathy n'existe pas.

08/09/2026

Donc aujourd'hui, remise à plat des observations que j'ai pu avoir il y a deux jours :
- le masque de Sobel est maintenant dans `masque_eau.py` et sert aux deux pays. Le seuil est en pente (0.04, ce qui revient à l'ancien seuil français de 0.2m pour un pixel de 5m).
- côté suisse, la cible devient z moins le niveau du lac, pris comme la médiane du MNT sur l'eau. Pour le Lac de Joux le niveau vaut 1004.11m.
- en testant je suis tombé sur un problème qui existait depuis le début : une partie des 2000 points trés n'était pas dans l'eau. Pour le Lac de Joux, 10% des points étaient au-dessus du lac (jusqu'à +47m, swissBATHY3D inclut les berges), et pour L80, 418 points sur 2000 étaient hors du masque, avec une distance au rivage d'un pixel dans les 8 directions. Maintenant on ne tire que dans le masque d'eau, et sous le niveau du lac côté suisse.
- la liste des lacs est dans `lacs.py`, le lac se donne en argument au lieu de changer une constante à la main, et `extraire_tous_les_lacs.py` relance tout d'un coup.
- j'ai renommé les fonctions en français, c'est plus clair pour moi.
- pour avoir une vraie note à la fin, j'ai mis de côté la moitié des 86 nouveaux lacs français jamais utilisés pour un choix : 43 lacs, tirés au hasard avec la graine 42. Ils ne servent qu'une fois, à la toute fin (`evaluation_finale.py --test-final`). Les 64 autres lacs servent au développement.
- ajouté un README, et scikit-learn dans requirements.txt, qui manquait.

10/09/2026
Ensuite, extraction complète des 107 lacs. Deux problèmes en route :
- Lac de Neuchâtel : le script est monté à plus de 14 Go de mémoire, parce que tous les points bathy étaient chargés d'un coup. Maintenant on ne lit que 5% des points de chaque tuile, tirés au hasard, ce qui suffit largement pour en garder 2000. Relancé les 9 lacs suisses.
- 22 lacs français avaient un CSV vide, aucun point de mesure ne tombait dans le masque. La règle gardait la plus grande zone plate de l'image, et pour ces lacs ce n'était pas le lac mais une plaine ou un autre plan d'eau (par exemple L15 : 100% de ses points sur une zone plate de 175 ha, alors que la plus grande zone plate faisait 486 ha). C'était donc déjà faux dans le test à 107 lacs du 03/08. Corrigé en gardant la zone plate qui contient le plus de points bathy. Relance des 98 lacs : 0 échec, 0 CSV vide.

L70 ne garde que 19 points dans le masque. En dessous de 200 points le R² ne veut rien dire (comme aegerisee à 91 points le 14/07), donc je l'ignore.

Autre chose remarquée : le coefficient isopérimétrique est faussé par les trous du masque (îles, zones pas plates au milieu du lac), qui rallongent le périmètre. Il y a en médiane 16 trous par lac, soit 7% du périmètre, et jusqu'à 66% pour L4.

13/09/2026

Résultats du leave-one-lake-out sur les 63 lacs de développement (Random Forest, 200 arbres) :

Config                          R² médian   R² > 0    RMSE du ratio
avec coeff isopérimétrique      0.198       40/63     0.220
sans coeff isopérimétrique      0.174       42/63     0.223

Lac par lac, le coefficient améliore 31 lacs et en dégrade 32, avec un écart médian de RMSE nul. Donc le coefficient isopérimétrique, que j'avais choisi le 20/07 sur 9 lacs suisses avec la moyenne des R², n'apporte en fait rien de mesurable. Je le garde pour le test final parce que c'est la config retenue, mais c'est à revoir.

Test final sur les 43 lacs mis de côté, avec le modèle entraîné sur les 63 lacs de développement :

                                  R² médian   R² > 0    RMSE du ratio
développement (un lac de côté)    0.198       40/63     0.220
test final (lacs jamais vus)      0.050       22/43     0.213

Le R² médian tombe à 0.05, donc comme je le craignais, mes scores de développement étaient optimistes. Par contre le RMSE du ratio ne bouge presque pas, l'erreur reste autour de 0.21 sur un ratio entre 0 et 1. Peut-être que la profondeur relative varie moins dans ces lacs-là, ce qui fait baisser le R² à erreur égale, mais je ne l'ai pas vérifié. Les pires lacs : L54 (R²=-2.57), qui avait déjà peu de points sur une zone plate, puis L89 (-1.86), L56 (-1.02), L81 (-1.00).
Il s'agit donc ici de la fin de ce projet, je vous remercie pour votre lecture si vous êtes arrivés jusqu'ici !

Pour le bilan de ce qui a été vu, le modèle retrouve à peu près la forme générale du fond, avec une erreur d'environ 0.21 sur la profondeur relative, mais il explique encore peu les différences de profondeur à l'intérieur d'un lac qu'il n'a jamais vu. Les 43 lacs du test ne doivent plus servir à choisir quoi que ce soit : si je change la config, il faudra mettre d'autres lacs de côté. Je décide de m'arreter à ce stade car dans ce domaine, le problème sur la difficulté de trouver la forme d'un fond et compliqué et bien connu/documenté, d'autrs pistes resteraient envisageables comme combler les trous du masque avant de recalculer le coefficient isopérimétrique, regarder pourquoi L54 et L89 ratent autant, et trouver d'autres variables que le relief autour du lac. Cela pourrait être fait dans l'avenir mais déjà, le processus itératif derrière le projet demeure plus qu'interessant et valorisant pour moi et m'a permis de comprendre les processus et limites de l'apprentissage, et ce qui en dépendaient tout du long.

