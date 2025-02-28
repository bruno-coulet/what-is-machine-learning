[Retour à la regression](README.md#regression-lineaire)

#### Régression linéaire
méthode simple qui modélise la relation entre une variable dépendante (en sortie) et une ou plusieurs variables indépendantes (en entrée) par une droite.

**prédiction =  somme pondérée des variables d'entrée plus une constante**

Peut s'écrire sous plusieurs formes:
- scalaire
- pondérée
- vectorielle
- matricielle

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


