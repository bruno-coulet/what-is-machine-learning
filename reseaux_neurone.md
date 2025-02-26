# Les réseaux neuronaux artificiels

[retour au readMe](README.md)

peuvent être classés en fonction de la manière dont les données circulent du nœud d'entrée au nœud de sortie. Voici quelques exemples :

## Réseaux neuronaux à action directe
Les réseaux neuronaux à action directe (feedforward) traitent les données dans une seule direction, du nœud d'entrée au nœud de sortie. Chaque nœud d'une couche est connecté à chaque nœud de la couche suivante. Un réseau à action directe utilise un processus rétroactif pour améliorer les prédictions au fil du temps.

## Algorithme de rétropropagation
Les réseaux neuronaux artificiels apprennent en permanence en utilisant des boucles de rétroaction corrective pour améliorer leur analytique prédictive. En termes simples, vous pouvez imaginer que les données circulent du nœud d'entrée au nœud de sortie par plusieurs chemins différents dans le réseau neuronal. Un seul chemin est le chemin correct qui relie le nœud d'entrée au nœud de sortie correct. Pour trouver ce chemin, le réseau neuronal utilise une boucle de rétroaction, qui fonctionne comme suit :

1. Chaque nœud fait une supposition sur le prochain nœud du chemin.
2. Il vérifie si la supposition était correcte. Les nœuds attribuent des valeurs de poids plus élevées aux chemins qui mènent à un plus grand nombre de suppositions correctes et des valeurs de poids plus faibles aux chemins de nœuds qui mènent à des suppositions incorrectes.
3. Pour le point de données suivant, les noeuds effectuent une nouvelle prédiction en utilisant les chemins de poids plus élevé, puis répètent l'étape 1.

## Réseaux neuronaux convolutifs
Les couches cachées des réseaux neuronaux convolutifs exécutent des fonctions mathématiques spécifiques, comme la synthèse ou le filtrage, appelées convolutions. Ils sont très utiles pour la classification des images, car ils peuvent extraire des caractéristiques pertinentes des images qui sont utiles pour la reconnaissance et la classification des images. La nouvelle forme est plus facile à traiter sans perdre les caractéristiques qui sont essentielles pour faire une bonne prédiction. Chaque couche cachée extrait et traite différentes caractéristiques de l'image, comme les bords, la couleur et la profondeur.

[retour au readMe](README.md)
