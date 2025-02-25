# what-is-machine-learning

Recherches, documentées, illustrées et sourcées des éléments suivants :

Définition des éléments de
veille ci-dessous.
Cette documentation se fera sous la forme de votre choix (Markdown, document PDF)

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
![deep learning](img/deepLearning.png)
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
 
**Classification** : prédiction d'une variable qualitative
**Regression** : prédiction d'une variable quantitative.

## Classification supervisée

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
## Données d’entraînement, les données de test et/ ou de validation
## Corrélation linéaire (de Pearson) entre deux variables
## Fonction de coût
## Descente de gradient


contexte du projet :


