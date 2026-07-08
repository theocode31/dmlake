05.07.2026
Ce nouveau projet  est orienté bathymétrie, on va essayer d' estimer la profondeur des lacs naturels à partir de leur morphologie environnante, sans mesure directe. Les lacs naturels n'ont pas d'infrastructure de référence comme les barrages, donc l'idée c'est de prédire la profondeur à partir du relief autour (pente, distance au rivage...).

Pour ce premier jour, le but est de mettre en place le projet et voir si j'arrive à récupérer un jeu de données bathymétriques réel et à le visualiser.

J'ai trouvé deux sources ouvertes intéressantes : swissBATHY3D/swissALTI3D (relevés bathymétriques et MNT suisses) et EauFrance (relevés bathymétriques Adour-Garonne). J'ai testé sur le Lac de Joux, un petit lac suisse, pour aller vite. Le téléchargement et le chargement des points fonctionnent (~9.5 millions de points sur ce lac), et j'ai sorti une première visu du nuage de points coloré par profondeur, la forme du lac se voit bien, ça confirme que les données sont exploitables.

06.07.2026

Avant de pouvoir extraire des features de relief (pente, distance au rivage...), il faut le MNT autour du lac, dans le même référentiel spatial que la bathymétrie. J'ai repéré les cellules de la grille kilométrique couvertes par la bathy du Lac de Joux, ajouté une couronne d'1km pour déborder sur le rivage, et interrogé l'API STAC de swisstopo pour récupérer les tuiles swissALTI3D 2m correspondantes (61 tuiles). Après mosaïquage, j'ai superposé le MNT et le nuage de points bathy pour vérifier l'alignement : le contour du lac colle exactement à la cuvette du relief (altitude MNT ~1004m au niveau des points bathy, contre 971-1041m pour le fond du lac, donc une profondeur max ~32m cohérente avec les ~31m connus). Les deux jeux de données sont bien dans le même référentiel (EPSG:2056), on passera a la suiite demain.

07.07.2026

Aujourd'hui , on fait une extraction des features de relief. D'abord un masque d'eau (enveloppe convexe des points bathy rasterisée sur la grille du MNT, suffisant pour un lac simple comme Lac de Joux, pas besoin du filtrage Sobel utilisé pour des lacs plus complexes). Puis, pour un échantillon de 2000 points du fond du lac, dans 8 directions (0 à 315°, pas de 45°), recherche du rivage (transition eau→terre dans le masque), puis prolongation du rayon sur terre pour mesurer pente et dénivelé à 4 distances (150/300/600/900m). resultat dans :lacdejoux_cross_shore_profiles
