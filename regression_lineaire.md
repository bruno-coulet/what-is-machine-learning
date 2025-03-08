[Retour au README / regression](README.md#regression-lineaire)

# Régression linéaire
méthode simple qui modélise la relation entre une variable dépendante (en sortie) et une ou plusieurs variables indépendantes (en entrée) par une droite.

En apprentissage supervisé, la régression linéaire vise à approximer une relation linéaire entre les variables explicatives 
$𝑋$ et la variable cible $𝑦$.

**prédiction =  somme pondérée des variables d'entrée plus une constante**

Peut s'écrire sous plusieurs formes:
- [scalaire](#Forme-scalaire)
- [pondérée](#Forme-somme-pondérée)
- [vectorielle](#Forme-vectorielle)
- [matricielle](#Forme-matricielle)

## Regression Linéaire - Entrainement

Entrainer un modèle consiste à définir ses paramètres de telle sorte qu'ils s'ajustent au mieux au jeu d'entrainement

Mesures courantes pour indiquer si un modèle de Regression s'ajuste bien ou pas aux donnée d'entrainement :

- RMSE racine carrée des erreurs quadratique moyenne -(root mean square error)
- MSE erreur quadratique moyenne - (mean square error)


Pour entrainer un modèle de régression linéaire, il faut donc trouver le vecteur θ qui minimise la RMSE

En pratique, il est plus simple de minimiser la MSE que la RMSE

La qualité de cette approximation est mesurée à l'aide de la fonction de coût, qui quantifie l'erreur entre les prédictions du modèle et les vraies valeurs.

### Fonction de coût MSE pour le modèle de régression linéaire


$$
MSE(X, h_\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( \theta^T x^{(i)} - y^{(i)} \right)^2
$$

$(X, h_\theta)$ pour montrer que le modèle est paramétré par le vecteur $\theta$<br><br>
Peut s'écrire de plusieurs manières :<br>
$MSE(X, h_\theta)$<br>$MSE(X, h)$ comme pour la RMSE ci-dessous<br>$MSE(\theta)$ pour simplifier

|symbole|signification|
|:--:|:--------|
|$m$ | nombre total d'exemples dans l'ensemble d'entraînement.|
|$𝑥(𝑖)$  | vecteur des caractéristiques de l'exemple 𝑖|
|$𝜃$ | vecteur des paramètres du modèle.|
|$𝜃𝑇𝑥(𝑖)$ <font color = "orange">ou</font> $h (x^{(i)})$ | prédiction du modèle pour $𝑥(𝑖)$|
|$y(i)$|valeur réelle associée à $𝑥_i$|
|$ (\theta^T x^{(i)} - y^{(i)})^2 $|erreur quadratique pour un exemple donné|

<font color = "orange">Le MSE pénalise les grandes erreurs plus fortement que les petites erreurs, ce qui le rend sensible aux valeurs aberrantes (outliers).</font>

### Fonction de coût MSE pour le modèle de régression linéaire

$$ RMSE(X, h) = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left( h (x^{(i)}) - y^{(i)} \right)^2} $$



### Equation normale - solution analytique
formule mathématique qui donne lavaleur minimal de $\theta$
$$ \hat{\theta} = (X^T X)^{-1} X^T y $$

| Symbole                            | signification |
|----------------------------------------|---------|
| $ \hat{\theta}$ | valeur du vecteur $\theta$ qui minimise la fonction de coût |
| $ y$ | vecteur des valeurs cible $y^{(1)}$ à $y^{(m)}$ |

| Description                            | Formule |
|----------------------------------------|---------|
| **Fonction de coût MSE** pour la régression linéaire | $ MSE(X, h_\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( \theta^T x^{(i)} - y^{(i)} \right)^2 $ |
| **Fonction de coût RMSE** (Racine de MSE) | $ RMSE(X, h_\theta) = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left( \theta^T x^{(i)} - y^{(i)} \right)^2} $ |
| **Équation normale** pour $\hat{\theta}$ | $ \hat{\theta} = (X^T X)^{-1} X^T y $ |


### Entrainement - Méthode analytique
Calcul les valeurs des paramètres du modèle qui donnent le meilleur résultat sur le jeu d'entrainement (qui minimise la fonction de coût)

### Entrainement - Descente de gradient
ou Gradient Descent en anglais Optimisation itérative qui modifie graduellement les paramètres du modèle pour minimiser la fonction de coût sur le jeu d'entrainement Converge au final vers le même jeu de paramètres que la méthode analytique



---

##### Forme scalaire

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$$

<p>

|symbole|signification|
|:--:|:--:|
|$\hat{y}$ | prédiction|
|$\theta_0$ | biais (intercept)|
|$\theta_i$ | paramètres du modèle|
|$x_i$ | variables d'entrée|
|$n$ | nombre de variables|

</p>

---

##### Forme somme pondérée

$$\hat{y} = \sum_{i=0}^{n} \theta_i x_i$$

En régression linéaire,
on écrit souvent la somme en commençant par 0 (i=0)
avec la convention que $x_0 = 1$
pour inclure le biais $\theta_0$ dans la somme.


**biais, constante, intercept** - sont 3 termes qui désignent la même chose



|symbole|signification|
|:--:|:--:|
|$\hat{y}$ | valeur prédite|
|$\theta_i$ | i-ème paramètre du modèle|
|$x_i$ | i-ème variable|

---

##### Forme vectorielle

On introduit le vecteur colonne des variables d'entrée, avec $𝑥0=1$:

- $𝑋=\begin{pmatrix} x_0 \\ x_1 \\ \vdots \\ x_n \end{pmatrix}$

Et le vecteur colonne des paramètres du modèle :

- $\theta=\begin{pmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_n \end{pmatrix}$


Le produit scalaire entre ces deux vecteurs donne :

$$\hat{y}=𝜃_0⋅𝑥_0+𝜃_1⋅𝑥_1+⋯+𝜃_𝑛⋅𝑥_𝑛​$$


---

##### Forme matricielle

$$\hat{y} =θ^TX$$

|symbole|signification|
|:--:|:--:|
|$X$ | vecteur des valeurs d'une observation de $x_0$ à $x_n$, matrice de taille (n+1) x 1|
|$θ^T$ | transposés se $\theta$, matrice de taille 1 x (n+1)|



[Retour à la regression](README.md#regression-lineaire)


