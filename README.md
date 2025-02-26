# what-is-machine-learning

## Contexte du projet :
Travail personnel de recherches et de documentation pour la définition des éléments suivants :


illustrés et sourcés

## A. La science des données
![data_science](img/dataScience.png)
Ou comment générer du sens à partir de données.
Considérée comme un nom alternatif pour les statistiques dans les années 60, la science des données devient une discipline issue de l'informatique à la fin des années 90.
Elle s'articule autour de : 
- La conception.
- La collecte.
- L'analyse de données.

L'objet de cette science est l'étudie et l'analyse des données afin d'en extraire des informations pertinentes pour les entreprises.
Elle adopte une approche pluridisciplinaire, mêlant des concepts et des méthodes issus des mathématiques, des statistiques, de l'intelligence artificielle et de l'informatique. L'objectif est d'examiner de vastes ensembles de données pour répondre à des questions clés telles que : Que s'est-il passé ? Pourquoi cela s'est-il produit ? Que va-t-il se passer ? Et quelles actions peut-on entreprendre sur la base de ces résultats ?

4 formes d'analyse principale se dégagent :
1. **Analyse descriptive**
visualisation de données
2. **Analyse diagnostique**
exploration, transformation, corrélation...
3. **Analyse prédictive**
machine learning, la prédiction, la comparaison de modèles et la modélisation prédictive.
4. **Analyse prescriptive**
suite logique et proactive de l'analyse prédictive

## B. Apprentissage automatique
L'art de programmer des ordinateurs de sorte qu'ils puissent apprendre à partir de données

On considère qu'un ordinateur "apprend" s'il améliore sa **performance** lors de l'exécution d'une **tâche** au fur et à mesure de son **expérience**.

##  Apprentissage profond
![deep learning](img/deep_learning.png)
Procédé d’apprentissage automatique utilisant des [réseaux de neurones](reseaux_neurone.md) composé de nombreuses couches cachées et des algorithmes avec de très nombreux paramètres.
Ce procédé requière une grande quantité de données afin d’être entraîné.
![resaue de neurones](img/reseauNeurones.png)
A permis des progrès importants et rapides :
- analyse du signal sonore, reconnaissance faciale
- analyse du signal visuel, reconnaissance vocale
- le traitement automatisé du langage

Le développement de l'apprentissage profond à été rendu possible par des investissements privés et publics importants, notamment de la part des GAFAM, durant les années 2000.

## Apprentissage supervisé
Les données d'entrainement fournies à l'algorithme comportent des **étiquettes** qui indiquent le résultat voulu.
L'objectif de l'algorithme est d'apprendre à faire correspondre les entrées aux sorties afin de pouvoir prédire l'étiquette correcte pour de nouvelles données jamais vues.

![apprentissage supervisé](img/apprentissage_supervise.png)

## Apprentissage non supervisé
Les données d'entrainement ne comportent **pas d'étiquettes**.
L'algorithme n'a donc pas d'informations sur le résultat attendu.
Il explore les données pour y découvrir des structures, des motifs ou des relations cachées sans qu'une sortie spécifique ne lui soit fournie.
Il tente d'organiser les données selon leurs similarités ou différences.

|                 | Apprentissage supervisé            | Apprentissage non supervisé           |
|-----------------------|------------------------------------|--------------------------------------|
| Données               | Étiquetées                         | Non étiquetées                      |
| Objectif              | Prédire une sortie                 | Découvrir des structures            |
| Types de problèmes    | Classification, régression         | Clustering, réduction de dimensionnalité |
| Exemples typiques     | Détection de spam, prévision des ventes | Segmentation de clients, détection d'anomalies |

## Classification / Régression
En apprentissage automatique, on distingue les problèmes de régression des problèmes de classification.

**En règle général :**
 
 |Classification|Regression|
 |--------------|----------|
 | prédiction d'une **variable qualitative**| prédiction d'une **variable quantitative**|



## Classification supervisée

La classification supervisée est une tâche d’apprentissage supervisé où le modèle apprend à attribuer des catégories ou des labels à de nouvelles données en se basant sur des exemples d’entraînement étiquetés. Par exemple, classer des emails comme spam ou non-spam.

1. **On dispose d'articles déjà classés en rubrique :**
économie, politique, sport, culture...

2. **On veut classer un nouvel article, lui attribuer une étiquette** 

![Classification supervisée](img/classification_supervise.png)

Définition des règles permettant de classer des objets dans des classes.

Ces règles se basent sur des variables qualitatives ou quantitatives des
objets à classer.

Les méthodes s'étendent souvent à des variables Y quantitatives (régression).

Un échantillon dont le classement est connu est utilisé pour l'apprentissage des règles de classement.

Il faut ensuite étudier la fiabilité de ces règles pour les comparer et les appliquer.
Evaluer les cas de sous apprentissage ou de sur apprentissage (complexité du modèle). 

On utilise souvent un deuxième échantillon indépendant, dit de validation ou de test.

https://math.univ-angers.fr/~labatte/enseignement%20UFR/master%20MIM/classificationsupervisee.pdf


## Classification non supervisée
1. **On dispose d'éléments non classés :**
mots d'un texte
![Classification non supervisée](img/classification_non_supervise1.png)
2. **On veut les regrouper en classes :**
si deux mots ont la même étiquette, ils sont en rapport avec une
même thématique...
![Classification non supervisée 2](img/classification_non_supervise2.png)
![Classification non supervisée3](img/classification_non_supervise3.png)

## Régression

En mathématiques, la régression recouvre plusieurs méthodes d’analyse statistique permettant d’approcher une variable à partir d’autres qui lui sont corrélées.

## Validation croisée
Dans un projet de Machine Learnig, il faut séparer les données :

1. un jeux de données pour **entrainer** le modèle 
2. un jeux de données pour **tester** le modèle entrainé
3. un dernier jeux pour **valider** le modèle sur de nouvelles données

**<font color="orange">La validation croisée</font>** consiste en l'utilisation alternative et conjointe des jeux d'entrainement et de test.

**<font color="orange">La validation croisée</font>** se compose de plusieurs étapes :
1. division du jeux de donné en K sous-ensembles
2. entrainement puis évaluation du modèle K fois
    - en changeant de combinaison jeux d'entrainement / jeux d'évaluation à chaque itération
3. compare les résultats

Ainsi, toutes les tranches de donnée sont alternativement réservèes aux test.
Au final, toutes les données ont servies à l'entrainement et au test.
Cela permet d'obtenir une estimation plus stable des performances.


### Validation croisée en images

les k-1 premières tranches sont utilisé pour l'entrainement
![train_4](img/cross_validation_train_4.png)
la tranche k est utiliée pour le test
![test_4](img/cross_validation_test_4.png)
on note les résultats
![track_4](img/cross_validation_track_4.png)
puis c'est la tranche k-1 qui est réservée pour le test, on note les résultats
![track_3](img/cross_validation_track_3.png)

Et ainsi de suite pour chaque tranches jusquà la 1ère tranche
![](img/cross_validation_test2-.png)
 puis on compile les résultats.

L'idéal étant de faire une **validation croisée avec différent modèles** afin de les comparer :
- Logistic regression
- support vector machines
- k-nearest neighbors
- etc...
![track_2](img/cross_validation_comparaison.png)

## Données d’entraînement, les données de test et/ ou de validation
Dans un projet de Machine Learnig, il faut séparer les données :

1. un jeux de données d'**entrainement** pour ajuster le modèle aux données. **apprendre à partir des données.**
2. un jeux de données de **test** pour évaluer les performances du modèle entrainé. **évaluer la qualité de l'apprentissage.**
3. un dernier jeux de **validation** pour optimiser le modèle et prévenir le surajustement.  **affiner et optimiser le modèle.**
   (avec de nouvelles données)
 
Ces trois étapes – entraînement, test et validation – sont essentielles pour garantir que le modèle est fiable et performant avant son déploiement.



### Entrainement

Après les phase de collecte, de nettoyage et de préparation des données :
- recherche de correlations entre les variables
- gestion des variables quantitative (stratification, normalisation, ...)
- gestion des variables qualitative (encodage, onehot encoding, ...)
- combinaison de variables (création de nouvelles caractéristiques, feature engineering).

Vient la phase de l'**entrainement du modèle**.
supervisé ou non supervisé selon que les données contiennent ou non des étiquettes (labels)
Il permet d'ajuster le modèle choisi aux données dans le but de faire des prédictions ou de la classification sur de nouvelles données

**Problématique :**

**- Surajustement** (overfitting) : le modèle apprend trop bien les détails et le bruit des données d'entraînement, ce qui nuit à sa capacité à généraliser.
**- Sous-ajustement** (underfitting) : le modèle est trop simple et ne capte pas la structure sous-jacente des données.

### test
La phase de test consiste à évaluer les performances du modèle sur un ensemble de données qui n'a pas été utilisé pendant l'entraînement.
Cela permet d'obtenir une estimation objective de la capacité du modèle à généraliser ses prédictions sur des données inconnues.

**Évaluation des performances :**
On compare les prédictions du modèle avec les valeurs du jeu de test.

**Métriques courantes pour évaluer les performances :**
- Précision (Accuracy) : proportion de prédictions correctes.
- Rappel (Recall) : capacité du modèle à identifier les éléments positifs.
- F1-score : moyenne harmonique entre la précision et le rappel.
- Erreur quadratique moyenne (MSE) : utilisée pour les modèles de régression.

**Importance du test :**
- Identifier les biais et faiblesses du modèle 
- Vérifier sa capacité à généraliser.

### validation
Après l'entraînement et le test, la validation est une étape cruciale.
Elle vise à affiner le modèle et à s'assurer qu'il fonctionne correctement dans des conditions réelles.

- Optimisation des hyperparamètres du modèle pour améliorer ses performances.
- détecter d'éventuels problèmes
- Détecter le surajustement : un modèle trop complexe mémorise les données au lieu d'apprendre leurs tendances.
- Détecter le sous-ajustement : un modèle trop simple passe à côté des structures importantes.
- Optimiser les performances : tester différentes configurations pour maximiser les résultats.
- Généralisation : le jeu de validation permet d'estimer comment le modèle se comportera sur des données réelles et non vues auparavant.



## Corrélation linéaire (de Pearson) entre deux variables
## Fonction de coût
## Descente de gradient
