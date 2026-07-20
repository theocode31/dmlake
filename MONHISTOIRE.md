05.07.2026
Ce nouveau projet  est orienté bathymétrie, on va essayer d' estimer la profondeur des lacs naturels à partir de leur morphologie environnante, sans mesure directe. Les lacs naturels n'ont pas d'infrastructure de référence comme les barrages, donc l'idée c'est de prédire la profondeur à partir du relief autour (pente, distance au rivage...).

Pour ce premier jour, le but est de mettre en place le projet et voir si j'arrive à récupérer un jeu de données bathymétriques réel et à le visualiser.

J'ai trouvé deux sources ouvertes intéressantes : swissBATHY3D/swissALTI3D (relevés bathymétriques et MNT suisses) et EauFrance (relevés bathymétriques Adour-Garonne). J'ai testé sur le Lac de Joux, un petit lac suisse, pour aller vite. Le téléchargement et le chargement des points fonctionnent (~9.5 millions de points sur ce lac), et j'ai sorti une première visu du nuage de points coloré par profondeur, la forme du lac se voit bien, ça confirme que les données sont exploitables.

06.07.2026

Avant de pouvoir extraire des features de relief (pente, distance au rivage...), il faut le MNT autour du lac, dans le même référentiel spatial que la bathymétrie. J'ai repéré les cellules de la grille kilométrique couvertes par la bathy du Lac de Joux, ajouté une couronne d'1km pour déborder sur le rivage, et interrogé l'API STAC de swisstopo pour récupérer les tuiles swissALTI3D 2m correspondantes (61 tuiles). Après mosaïquage, j'ai superposé le MNT et le nuage de points bathy pour vérifier l'alignement : le contour du lac colle exactement à la cuvette du relief (altitude MNT ~1004m au niveau des points bathy, contre 971-1041m pour le fond du lac, donc une profondeur max 32m . Les deux jeux de données sont bien dans le même référentiel (EPSG:2056), on passera a la suiite demain.

07.07.2026

Aujourd'hui , on fait une extraction des features de relief. D'abord un masque d'eau (enveloppe convexe des points bathy rasterisée sur la grille du MNT, suffisant pour un lac simple comme Lac de Joux, pas besoin du filtrage Sobel utilisé pour des lacs plus complexes). Puis, pour un échantillon de 2000 points du fond du lac, dans 8 directions (0 à 315°, pas de 45°), recherche du rivage (transition eau→terre dans le masque), puis prolongation du rayon sur terre pour mesurer pente et dénivelé à 4 distances (150/300/600/900m). On fait cela afin de construire les vairables d'entrée x de notre futur modele, pour s'entrainern a la prédiction. 
La raison pourlaquelle on prolonge le rayon sur terre c'est pour avoir un contexte singulier environnant par exemple en milieu alpin, on aura un lac avec un fond plus encaissé qu'un lac situé en zone de plaine.
resultat dans :lacdejoux_cross_shore_profiles

09.07.2026
Aujourd'hui, premier modèle de machine learning (`train_model.py`) : je donne au modèle plein de points du lac dont je connais déjà la vraie profondeur mesurée, avec les features de relief extraites au jour 3, et il apprend tout seul la relation entre les deux, pour ensuite deviner la profondeur de points qu'il n'a jamais vus. J'écarte deux colonnes (`angle45` et `angle225`) parce qu'elles ont presque 60% de valeurs manquantes le lac est allongé pile dans cet axe, du coup les rayons ne trouvent pas la terre à temps dans ces deux directions. J'écarte aussi les flags `_shore_extrapolated` (des indicateurs de fiabilité de la mesure, pas des vraies features de relief). Je garde 75% des points pour entraîner le modèle et 25% de côté pour vérifier s'il devine bien sur des points jamais vus. Le modèle choisi : Random Forest  simple et robuste pour un premier essai, sans réglage compliqué, et adapté ici parce que la relation relief/profondeur n'a pas de raison d'être linéaire.

Premier essai : R²=0.957 (un score entre 0 et 1 qui dit à quel point les prédictions collent aux vraies valeurs, 1 = parfait). Beaucoup trop bon pour un premier essai. En regardant quelle feature comptait le plus, j'ai trouvé le coupable : `angle0_z_DEM_ref` (= profondeur mesurée moins l'altitude du rivage) pesait à elle seule 50.7% de l'importance. Le souci est qe l'altitude du rivage change à peine à l'intérieur d'un même lac, donc cette colonne est quasiment une reformulation déguisée de la réponse qu'on essaie de deviner. Le modèle ne devinait pas, il lisait presque directement la solution. Je la retire des features.

Deuxième essai, sans cette triche : RMSE=2.60m (le modèle se trompe en moyenne de 2.60m sur la profondeur), MAE=1.59m (l'erreur "typique"), R²=0.938 , toujours très bon, mais honnête cette fois. Ce qui aide le plus à deviner : la distance au rivage dans certaines directions (180°, 270°, 135°).

J'ai séparé mes 75%/25% au hasard, point par point, sur un seul lac (Lac de Joux). Le souci est qui est d'ailleurs compliqué à resoudre deux points voisins sur le même lac ont presque les mêmes coordonnées, et des features de relief très proches. Du coup mon modèle a peut-être juste "reconnu" des points de test très proches géographiquement de points déjà vus à l'entraînement, plutôt que d'avoir vraiment compris la relation reliefe tprofondeur. Pour être sûr qu'il a appris quelque chose qui généralise, il faudrait le tester sur un lac complètement différent, jamais vu à l'entraînement, ce que je ferais plus tard.

13.07.2026

Repris le point en suspens du jour 4, avec un deuxième lac cette fois : Lungernsee (plus profond que Lac de Joux, ~68m contre 31m, mais surface plus petite). Même pipeline relancé dessus sans souci, 2000 points, pas de NaN.

Premier essai en gardant z comme cible , on a RMSE de 344m, anormal. En fait logique, Lac de Joux est vers 1000m d'altitude et Lungernsee vers 650m, le modèle avait juste appris le niveau du lac d'entraînement, pas une vraie relation relief-profondeur.

Repris avec angle0_z_DEM_ref comme cible  au lieu de z. L'erreur redevient raisonnable (30-50m) mais le R² reste négatif dans les deux sens du train/test, donc pire que de prédire la moyenne. Avec seulement 2 lacs le modèle n'a pas de quoi apprendre une relation qui généralise vraiment. Résultat pas terrible mais pas étonnant non plus, il faudra clairement plus de lacs pour voir si ça marche. Je m'arrête là pour aujourd'hui mais on va completer le test avec davantage d'exemples, les plus exotiques pour entrainer le mdoele au mieux.

14.07.2026

Ajouté 3 lacs de plus (aegerisee, baldeggersee, hallwilersee) pour avoir un vrai test a plusieurs lacs plutot que juste 2. Meme pipeline relancé sur chacun, aucun souci au telechargement/extraction.

En nettoyant les donnees je me rends compte que l'exclusion angle45/angle225, decidee a l'origine juste a cause de la geometrie de Lac de Joux, ne marche pas du tout pour les autres lacs : il ne reste que 91 points sur 2000 pour aegerisee apres dropna, 201 pour hallwilersee, 1054 pour baldeggersee (contre 2000 pour lacdejoux et lungernsee). Chaque lac a sans doute ses propres directions problematiques selon sa forme, pas forcement les memes que Lac de Joux.

Test leave-one-lake-out (entrainement sur 4 lacs, test sur le 5e, a tour de role), toujours avec angle0_z_DEM_ref comme cible : R²=0.149 pour lacdejoux, -0.751 pour lungernsee, -212 pour aegerisee (mais seulement 91 points de test, donc pas fiable), -0.770 pour baldeggersee, 0.090 pour hallwilersee. Toujours pas de vraie generalisation, un peu mieux que le test a 2 lacs sur certains, pire sur d'autres.


En regardant le taux de NaN par angle sur les 5 lacs, chacun a son propre axe problematique (0°/180° pour hallwilersee, 135°/315° pour aegerisee, etc), lie a sa propre orientation. Pas moyen de choisir un jeu d'angles qui marche pour tout le monde a la fois, il resterait seulement 2 directions sur 8 utilisables partout. Le vrai souci etait ailleurs , le rayon de recherche du rivage (fixe a 2500m au jour 3) etait juste trop court pour les lacs allonges, peu importe la direction choisie.

Après avoir corriger en calculant ce rayon dynamiquement par lac (diagonale de l'emprise du lac + marge de 10%) dans extract_features.py, au lieu d'un seuil fixe. Refait l'extraction sur les 5 lacs avec ce rayon adaptatif : 0% de NaN partout, sur les 8 angles, plus besoin d'exclure quoi que ce soit.

Relance du leave-one-lake-out avec les jeux de donnees complets (2000 points par lac au lieu de 91 pour aegerisee) : R²=0.307 pour lacdejoux (etait 0.149), -0.159 pour lungernsee (etait -0.751), -0.567 pour aegerisee (etait -212), 0.140 pour baldeggersee (etait -0.770), 0.656 pour hallwilersee (etait 0.090). Nette amelioration partout, 3 lacs sur 5 ont maintenant un R² positif. Aegerisee et lungernsee restent negatifs mais beaucoup moins qu'avant. Le vrai probleme etait bien la couverture des donnees, pas le modele.

Pour continuer a tester la generalisation, ajoute les 6 lacs restants du jeu de donnees swissBATHY3D : bielersee, brienzersee, lacneuchatel, lagomaggiore, bodensee, lacleman. La aussi decouverte importante : bodensee et lacleman sont des lacs transfrontaliers, et swissALTI3D ne couvre que le territoire suisse , apres nettoyage il ne reste que 0 point valide sur 2000 pour bodensee, 18 pour lacleman (contre 296 pour lagomaggiore, aussi frontalier avec l'Italie mais moins touche). Exclus ces deux-la du test, gardes lagomaggiore malgre ses 296 points.

Leave-one-lake-out relance sur ces 9 lacs : résultat globalement pire qu'avec 5, pas mieux. J'ai cherché pourquoi, deux causes trouvées plausible, en toute honneteté, cette étape était longue à franchir, je me suis donc aidé de l'ia pour qu'elle m'expose les causes du probleme:.

Première cause : en regardant l'importance des features, `surface_area` (la surface totale du lac, une seule valeur, la même pour tous les points d'un même lac) écrase tout le reste à plus de 50%. Le problème est comme cette valeur ne change jamais à l'intérieur d'un même lac, elle finit par se comporter comme un simple "numéro de lac" plutôt qu'une vraie information sur le relief. Le modèle s'appuie dessus pour deviner "quel lac c'est" plutôt que d'apprendre une vraie relation relief 'et profondeur qui marcherait sur un lac jamais vu.

Deuxième cause, plus grosse : les deux nouveaux lacs les plus profonds (Brienzersee et Bielersee) sont beaucoup plus profonds que les 5 premiers,  une erreur de 175m et 126m sur ces deux-là, le modèle a été entraîné surtout sur des lacs peu profonds (30 à 70m), donc quand on lui demande de deviner un point à 200m de profondeur, il n'a tout simplement jamais vu ça pendant l'entraînement et ne sait pas extrapoler au-delà de ce qu'il connaît, même limite que celle déjà repérée sur les features de relief, mais cette fois sur l'échelle de profondeur elle-même.

Cela pourrait donc nous pousser à verifier la singuralité de chaque lac afin d'entreprendre des manipulatiosn, ajouter des lacs "au hasard" sans faire attention à leur profondeur ne suffit pas, deux pistes à creuser une prochaine fois, on peut retirer `surface_area`, ou regrouper les lacs par gamme de profondeur avant d'entraîner. Comme le dernier élement trouvé est personnel je pense que je vais reprendre sur celui ci.

15.07.2026

Repris sur l'hypothese de l'echelle de profondeur. D'abord elargi le jeu de lacs suisses au maximum disponible , les 6 lacs restants (bielersee, brienzersee, lacneuchatel, lagomaggiore, bodensee, lacleman). Bodensee a fait planter le script de verification visuelle (mosaique MNT de 19000x35000 pixels, trop grosse pour matplotlib), corrige en sous-echantillonnant l'image juste pour l'affichage, l'extraction de features elle-meme n'etait pas touchee.

Sur ces 6 lacs, decouverte d'une limite differente de tout ce qu'on avait vu, bodensee et lacleman sont transfrontaliers, et swissALTI3D (le MNT) ne couvre que le territoire suisse. Apres nettoyage il ne reste que 0 point valide sur 2000 pour bodensee (perimetre presque entierement allemand/autrichien), 18 pour lacleman (frontiere francaise), contre 296 pour lagomaggiore (frontiere italienne, moins touche). Aucun rayon de recherche plus grand n'y changera quoi que ce soit, ce n'est pas un bug mais une vraie limite de couverture de la source de donnees. Exclus bodensee et lacleman, garde les 9 autres lacs suisses.

Pour avoir plus de lacs, j'ai codé mon propre pipeline français dans ce repo, sur le même principe que le pipeline suisse : `fetch_dem_france.py` (télécharge la bathymétrie EauFrance et le MNT RGE ALTI de l'IGN autour du lac, via une requête WMS) et `extract_features_france.py` (même méthode cross-shore). Deux différences trouvées en le construisant, le MNT français (RGE ALTI) fait 5m de résolution au lieu de 2m côté suisse , et surtout, `z` côté français est déjà une profondeur mesurée (pas une altitude absolue comme côté suisse), c'est bon signecar ça confirme que le choix de cible `angle0_z_DEM_ref` était le bon réflexe depuis le début. Testé sur 12 lacs (L1 à L92) : 0% de NaN partout, cela semble donc même mieux en terme d'apprehension qualiative de la donnée.

16.07.2026

Leave-one-lake-out relancé sur 21 lacs , on a 7 lacs sur 21 en R² positif. Mais surtout, un resultat qui confirme tres clairement l'hypothese de l'echelle que l'on avait avant : le `L9` (lac francais minuscule, profondeur max de 6m) a un R²=-171, alors que son RMSE (13.9m) n'est meme pas enorme en absolu, juste que 13.9m d'erreur est absurde pour un lac qui ne fait que 6m de profond. Meme phenomene que brienzersee/bielersee (trop profonds) mais a l'autre extreme.

Le cas de L9 donne une idée pour corriger le problème d'échelle que l'on a autant eu, au lieu de faire deviner au modèle la profondeur en mètres (qui varie énormément d'un lac à l'autre, de 6m à 260m), je lui fais deviner un ratio sans unité avec profondeur du point / profondeur max de ce lac-là. Un point au fond du lac le plus profond de son lac vaudrait 1, un point tout près du bord vaudrait proche de 0, peu importe que le lac fasse 6m ou 260m de profond, le ratio reste comparable. L'idée est de décorréler "à quel point ce point est profond par rapport à son propre lac" de "quelle est la profondeur totale de ce lac", pour que le modèle n'ait plus à deviner une échelle qu'il n'a jamais vue.

Relance du leave-one-lake-out avec cette cible normalisée à la place de la profondeur brute, nette amélioration, 15 lacs sur 21 ont maintenant un R² positif (contre 7 avant). Le cas L9 (catastrophique avant, R²=-171) remonte à -0.12. Brienzersee (trop profond) passe de -0.83 à +0.07. Ça confirme que la relation "forme du relief autour de la position relative en profondeur dans le lac" se généralise plutôt bien d'un lac à l'autre, une fois qu'on retire le problème d'échelle.

20.07.2026
surface_area  écrase tout et pose souci, surtout cela pose biais vers les lacs peu profonds. La parade trouvée serait d'utiliser un coefficient isopérimétrique (compacité du lac, indépendant de la taille). On a ajouté ce coefficient isopérimétrique côté suisse. Testé en trois versions : sans rien (R² moyen -0.293), avec `surface_area` en plus (-0.313, pire), isopérimétrique seul sans `surface_area` (-0.184, le meilleur). 

Relancé le test complet sur les 21 lacs avec cette config gagnanten cependant, seulement 5 lacs sur 21 en R² positif, contre 15 avant. Testé en gardant `surface_area` comme avant mais sur les données françaises actuelles, toujours mauvais. Donc le souci vient des données françaises, pas du choix de feature. On verra comment s'y prendre pour la suite, car cela devient assez compliqué.


Avant, pour les lacs français, on prenait l'enveloppe convexe des points de mesure. Mais ur un lac tout en longueur ou avec des bras qui partent dans plusieurs directions (comme une vallée noyée), cet élastique passe au-dessus de bouts de terre entre les bras du lac. Le modèle croyait que certaines zones de terre étaient de l'eau, ce qui faussait toutes les distances au rivage et les pentes calculées autour. On est passé à une méthode qui regarde directement le relief (le MNT) et détecte les zones plates (l'eau est plate, la terre autour ne l'est pas), beaucoup plus fidèle à la vraie forme du lac, peu importe qu'il soit tordu ou ramifié. C'est donc la méthode du masque de Sobel qui detecte les contours

Les 4 lacs français qui avaient l'écart le plus flagrant entre l'ancien masque et la vraie forme (L1, L30, L60, L90) sont passés de très négatifs (jusqu'à R²=-1.12 pour L1) à nettement positifs (jusqu'à +0.49). C'était bien le masque le problème, pas les données françaises en elles-mêmes ni le choix de features.

lungernsee, bielersee, lacneuchatel, lagomaggiore et L9 restent négatifs ce sont exactement les lacs qu'on avait déjà identifiés il y a quelques jours comme ayant un problème différent, leur profondeur sort trop de la plage "normale" des autres lacs (bielersee/lacneuchatel/lagomaggiore sont beaucoup plus profonds que la moyenne, L9 est anormalement peu profond). Le modèle n'a jamais vu d'exemples à cette échelle-là pendant l'entraînement, donc il ne sait pas extrapoler, c'est le problème d'échelle qu'on documente depuis le jour 16, pas un nouveau bug.








