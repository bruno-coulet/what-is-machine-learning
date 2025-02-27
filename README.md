# what-is-machine-learning

## Contexte du projet :
Travail personnel de recherches et de documentation pour la définition des éléments suivants :


illustrés et sourcés



## Index

1. [Science des données](#science-des-données)
2. [Apprentissage automatique ](#apprentissage-automatique)
3. [Apprentissage supervisé](#apprentissage-supervisé)
4. [Apprentissage non supervisé](#apprentissage-non-supervisé)
5. [Classification supervisée](#classification-supervisée)
6. [Classification non supervisée](#classification-non-supervisée)
7. [Régression](#régression)
8. [Validation croisée](#validation-croisée)
9. [Données d'entraînement, de test, de validation](#données-dentraînement-de-test-de-validation)
10. [Corrélation linéaire (de pearson) entre deux variables](#corrélation-linéaire-de-pearson-entre-deux-variables)
11. [Fonction de coût](#fonction-de-coût)
12. [Descente de gradient](#descente-de-gradient)


## A. La science des données
Ou comment générer du sens à partir de données.
![data_science](img/dataScience.png)
Considérée comme un nom alternatif pour les statistiques dans les années 60, la science des données devient une discipline issue de l'informatique à la fin des années 90.
Elle s'articule autour de : 
- La conception.
- La collecte.
- L'analyse de données.

L'objet de cette science est l'étude et l'analyse des données afin d'en extraire des informations pertinentes pour les entreprises.

Elle adopte une approche pluridisciplinaire, mêlant des concepts et des méthodes issus des mathématiques, des statistiques, de l'intelligence artificielle et de l'informatique.
L'objectif est d'examiner de vastes ensembles de données pour répondre à des questions clés telles que :
- Que s'est-il passé ?
- Pourquoi cela s'est-il produit ?
- Que va-t-il se passer ?
- Quelles actions peut-on entreprendre sur la base de ces résultats ?

4 formes d'analyse principale se dégagent :
1. **Analyse descriptive**
visualisation de données
2. **Analyse diagnostique**
exploration, transformation, corrélation...
3. **Analyse prédictive**
machine learning, la prédiction, la comparaison de modèles et la modélisation prédictive.
4. **Analyse prescriptive**
suite logique et proactive de l'analyse prédictive


[Retour à l'index](#index)

## Apprentissage automatique
L'art de programmer des ordinateurs de sorte qu'ils puissent apprendre à partir de données

On considère qu'un ordinateur "apprend" s'il améliore sa **[performance](#fonction-de-coût)** lors de l'exécution d'une **tâche** au fur et à mesure de son **expérience**.

Il existe 2 grandes familles d'apprentissage automatique :
### La classification
prédire une catégorie ou une étiquette à partir des caractéristiques des données d'entrée.

**exemple :**  filtre de spam à partir d'e-mail accompagnés de leur classe (normal/spam) 

### La prédiction
prédit une valeur numérique **cible (target)** à partir des valeurs **caractéristiques(feature)** d'attributs ou de variables d'une observation

**exemple :** prédire le prix d'une voiture en fonction de son age, de son kilométrage, etc...

### Terminologie :


**variable explicative :** caractéristique
**caractéristique :** un attribut et sa valeur (ex: kilométrage = 58 000 km)
**variable à expliquer :** étiquette
**attribut :** type de donnée (ex: kilométrage)

[Retour à l'index](#index)
<br>

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

[Retour à l'index](#index)
<br>

## Apprentissage supervisé
Les données d'entrainement fournies à l'algorithme comportent des **étiquettes** qui indiquent le résultat voulu.
L'objectif de l'algorithme est d'apprendre à faire correspondre les entrées aux sorties afin de pouvoir prédire l'étiquette correcte pour de nouvelles données jamais vues.

![apprentissage supervisé](img/apprentissage_supervise.png)

[Retour à l'index](#index)


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


[Retour à l'index](#index)

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



[Retour à l'index](#index)

## Classification non supervisée
1. **On dispose d'éléments non classés :**
mots d'un texte
![Classification non supervisée](img/classification_non_supervise1.png)
2. **On veut les regrouper en classes :**
si deux mots ont la même étiquette, ils sont en rapport avec une
même thématique...
![Classification non supervisée 2](img/classification_non_supervise2.png)
![Classification non supervisée3](img/classification_non_supervise3.png)


[Retour à l'index](#index)

## Régression

En mathématiques, la régression recouvre plusieurs méthodes d’analyse statistique permettant d’approcher une variable à partir d’autres qui lui sont corrélées.

#### Régression linéaire
méthode simple qui modélise la relation entre une variable dépendante et une ou plusieurs variables indépendantes par une droite.
#### Régression polynomiale
permet de s'ajuster à des jeux de données non linéaires en introduisant des puissances des variables indépendantes. Elle utilise plus de paramètres, ce qui la rend plus flexible, mais également plus sujette au surajustement (overfitting).
#### Régression logistique
utilisée pour des problèmes de classification binaire, elle permet de prédire la probabilité d'appartenance à une classe en utilisant une fonction logistique (sigmoïde).
#### Régression softmax
une généralisation de la régression logistique utilisée pour les problèmes de classification multiclasse, qui permet de prédire les probabilités d'appartenance à plusieurs classes.

## Validation croisée
Dans un projet de Machine Learnig, il faut séparer les données :

- un jeux de données pour **entrainer** le modèle 
- un jeux de données pour **tester** le modèle entrainé
- un dernier jeux pour **valider** le modèle sur de nouvelles données

**<font color="orange">La validation croisée</font>**
Consiste en l'utilisation alternative et conjointe des jeux d'entrainement et de test.
Cela implique d'entrainer/tester le modèle plusieurs fois :

1. division du jeux de donné en K sous-ensembles
<br>
2. entrainement puis évaluation du modèle K fois
    - en changeant de combinaison jeux d'entrainement / jeux d'évaluation à chaque itération
<br>
3. compare les résultats
<br>

Ainsi, toutes les tranches de donnée sont alternativement réservèes aux test.
Au final, toutes les données ont servies à l'entrainement et au test.
Cela permet d'obtenir une estimation plus stable des performances.
Nécessite l'utilisation d'une [fonction de fitness](#fonction-de-coût)



[Retour à l'index](#index)
<br>
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


[Retour à l'index](#index)

## Données d'entraînement, de test, de validation
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

[Retour à l'index](#index)
<br>
## Corrélation linéaire (de Pearson) entre deux variables

Corrélation linéaire (de Pearson) entre deux variables
La corrélation linéaire de Pearson mesure l'intensité et le sens de la relation linéaire entre deux variables quantitatives.

Elle est définie par un coefficient, noté **r**, avec une **valeur comprise entre -1 et 1 :**
| coefficient r|correlation|signification|
|--|-------|-----|
|**1**|Corrélation **positive forte**| lorsque l'une des variables augmente, l'autre augmente aussi|
|**- 1**|Corrélation **négative forte**| lorsque l'une des variables augmente, l'autre diminue|
|**0**|**Aucune corrélation linéaire**| les variables ne présentent pas de relation linéaire claire|

La formule du coefficient de Pearson est donnée par :
​$$ r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}} $$


- **r** : le coefficient de corrélation linéaire de Pearson  
- **\( x_i \)** et **\( y_i \)** : les valeurs des deux variables étudiées  
- **\( \bar{x} \)** et **\( \bar{y} \)** : les moyennes des variables \( x \) et \( y \)  
- Le numérateur mesure la covariance entre \( x \) et \( y \)  
- Le dénominateur normalise cette covariance par le produit des écarts-types des deux variables  


**<font color="orange">Attention</font> : une corrélation forte ne signifie pas nécessairement une relation causale**

[Retour à l'index](#index)
<br>
## Fonction de coût

Mesure de performance qui permet de savoir si le modèle est bien parametré :
| fonction|Valeur si le modèle est bon|
|--------------|-------------|
|de **fitness** (d'adaptation) | élevée|
| de **coût** | faible|

La **<font color="orange">fonction de coût</font>** quantifie l'**écart entre les prédictions du modèle et les valeurs réelles**.
L'objectif lors de l'entraînement est de minimiser cette fonction pour améliorer la précision du modèle.

Différents types de fonctions de coût existent selon le problème traité :

- Erreur quadratique moyenne (MSE) pour les problèmes de régression
- Entropie croisée pour les problèmes de classification
- Hinge loss pour les SVM (machines à vecteurs de support)

Un modèle bien paramétré aura donc une fonction de coût faible et, inversement, une fonction de fitness élevée, indiquant une bonne capacité du modèle à généraliser sur des données non vues.

[Retour à l'index](#index)

## Descente de gradient
Ou **Gradient Descent** en anglais **GD**

Méthode d'entrainement d'un modèle de regression linéaire par **optimisation itérative**
Consiste enune modification graduelle du paramêtre du modèle pour diminuer la fonction de coût sur le jeu d'entrainement

Il existe plusieurs variantes :
- Descente de gradient groupée (batch)
- Descente de gradient par mini-lots
- Descente de gradient stochastique

[Retour à l'index](#index)
