# Cours de Machine Learning (ML) de A à Z

> Architecture suivie (comme sur ton schéma) :
>
> 1. Positionnement du Machine Learning  
> 2. **Apprentissage supervisé** → Classification, Régression, Détection d’objets  
> 3. **Apprentissage non supervisé** → Clustering, Réduction de dimension, Modèles génératifs (VAE, GAN, Diffusion, Transformers/LLM)  
> 4. **Apprentissage par renforcement** → Jeux, Robotique, Optimisation  
> 5. Place de l’IA générative dans le paysage ML

---

## 0) Pré-requis indispensables

### Mathématiques
- Algèbre linéaire : vecteurs, matrices, produit scalaire, valeurs propres.
- Probabilités/statistiques : loi normale, espérance, variance, Bayes.
- Calcul différentiel : dérivées, gradient, descente de gradient.

### Programmation
- Python : `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
- Bonus deep learning : `pytorch` ou `tensorflow`.

### Notions de base
- Dataset = données d’entrée.
- Feature (variable explicative) = colonne d’entrée.
- Label/target = ce qu’on veut prédire.
- Modèle = fonction qui apprend le lien entre entrées/sorties.

---

## 1) Positionnement dans le Machine Learning

Le **Machine Learning** est une sous-partie de l’IA :
- IA (grand domaine)
- ML (apprentissage à partir de données)
- DL (deep learning, réseaux de neurones profonds)
- IA générative (générer texte/images/audio/code)

Un pipeline ML standard :
1. Définir le problème métier.
2. Collecter et nettoyer les données.
3. Split train/validation/test.
4. Choisir un modèle.
5. Entraîner + tuner les hyperparamètres.
6. Évaluer (métriques adaptées).
7. Déployer + monitorer.

---

## 2) Apprentissage supervisé

Le modèle apprend avec des exemples **(X, y)**, où `y` est connu.

## 2.1 Classification

### Objectif
Prédire une **classe** (spam/pas spam, fraude/non fraude, chat/chien).

### Algorithmes classiques
- Régression logistique
- Arbres de décision
- Random Forest
- SVM
- Réseaux de neurones (MLP, CNN)

### Métriques
- Accuracy
- Precision / Recall / F1
- AUC-ROC
- Matrice de confusion

### Exemple concret : détection de spam
- Entrée : texte d’email vectorisé (TF-IDF).
- Sortie : `spam` ou `ham`.

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

texts = ["gagnez de l'argent", "réunion demain 10h", "offre gratuite", "compte-rendu projet"]
labels = [1, 0, 1, 0]  # 1=spam, 0=non spam

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.5, random_state=42)
vec = TfidfVectorizer()
X_train_v = vec.fit_transform(X_train)
X_test_v = vec.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train_v, y_train)
preds = clf.predict(X_test_v)
print(classification_report(y_test, preds))
```

---

## 2.2 Régression

### Objectif
Prédire une **valeur continue** (prix, température, ventes).

### Algorithmes
- Régression linéaire
- Ridge / Lasso
- Random Forest Regressor
- XGBoost/LightGBM
- Réseaux de neurones

### Métriques
- MAE
- MSE
- RMSE
- R²

### Exemple concret : prédiction de prix immobilier

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))
```

---

## 2.3 Détection d’objets

### Objectif
Identifier **quoi** (classe) et **où** (boîte englobante) dans une image.

### Modèles populaires
- YOLO (temps réel)
- Faster R-CNN (précis)
- SSD, RetinaNet

### Métriques
- mAP (mean Average Precision)
- IoU (Intersection over Union)
- Recall à différents seuils

### Exemple d’usage
- Caméras industrielles (détection défauts)
- Voiture autonome (piétons, panneaux)
- Retail (inventaire visuel)

---

## 3) Apprentissage non supervisé

Pas de label `y`. Le but est de découvrir la structure cachée des données.

## 3.1 Clustering

### Objectif
Regrouper automatiquement des observations similaires.

### Algorithmes
- K-Means
- DBSCAN
- Agglomerative Clustering

### Métriques
- Silhouette Score
- Davies-Bouldin
- Calinski-Harabasz

### Exemple : segmentation clients
- Variables : fréquence d’achat, panier moyen, récence.
- Résultat : groupes de clients (fidèles, occasionnels, premium).

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[2, 30], [3, 35], [20, 300], [22, 280], [10, 120]])
X = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
print(clusters)
```

---

## 3.2 Réduction de dimension

### Objectif
Passer de nombreuses features à un espace plus petit en conservant l’essentiel.

### Méthodes
- PCA (linéaire)
- t-SNE (visualisation)
- UMAP (structure locale + globale)
- Autoencoders

### Cas d’usage
- Visualiser des données complexes en 2D/3D
- Réduire le bruit
- Accélérer certains modèles

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)
X_pca = PCA(n_components=2).fit_transform(X)
print(X_pca.shape)  # (n_samples, 2)
```

---

## 3.3 Modèles génératifs

Objectif : apprendre la distribution des données pour **générer** de nouveaux exemples plausibles.

### 3.3.1 VAE (Variational Autoencoder)
- Encodeur : transforme entrée → espace latent probabiliste.
- Décodeur : latent → reconstruction.
- Pertes : reconstruction + divergence KL.
- Usages : génération d’images, interpolation latente, débruitage.

### 3.3.2 GAN (Generative Adversarial Networks)
- Générateur vs Discriminateur en compétition.
- Le générateur apprend à tromper le discriminateur.
- Usages : images réalistes, super-résolution, data augmentation.
- Limite : entraînement instable, mode collapse.

### 3.3.3 Modèles de diffusion
- Ajout progressif de bruit puis apprentissage du débruitage inverse.
- Très performants en génération d’images.
- Usages : text-to-image, inpainting, stylisation.

### 3.3.4 Transformers et LLM
- Basés sur le mécanisme d’attention.
- Pré-entraînement sur grands corpus + fine-tuning/instruction tuning.
- Usages : résumé, chat, extraction d’info, génération de code.

### Différence principale
- VAE/GAN/Diffusion : surtout vision/audio (mais pas uniquement).
- Transformers/LLM : très dominants en NLP, multimodal en expansion.

---

## 4) Apprentissage par renforcement (Reinforcement Learning)

Un agent interagit avec un environnement pour maximiser une récompense cumulée.

### Concepts clés
- État `s`
- Action `a`
- Récompense `r`
- Politique `π(a|s)`
- Valeur d’état / Q-value

### Familles d’algorithmes
- Value-based : Q-Learning, DQN
- Policy-based : REINFORCE
- Actor-Critic : A2C, PPO, SAC

## 4.1 Jeux
- Atari, échecs, Go, stratégie temps réel.
- Exemple connu : AlphaGo (combinant RL + recherche).

## 4.2 Robotique
- Contrôle de bras robotisés
- Locomotion
- Manipulation d’objets

## 4.3 Optimisation
- Gestion de ressources
- Allocation dynamique
- Trading algorithmique (avec fortes précautions)

### Mini exemple Q-learning (table)
```python
import numpy as np

Q = np.zeros((5, 2))  # 5 états, 2 actions
alpha, gamma, epsilon = 0.1, 0.99, 0.1

# boucle simplifiée
for episode in range(100):
    s = np.random.randint(0, 5)
    for _ in range(20):
        if np.random.rand() < epsilon:
            a = np.random.randint(0, 2)
        else:
            a = np.argmax(Q[s])

        s_next = np.random.randint(0, 5)
        r = 1 if s_next == 4 else 0
        Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
        s = s_next
```

---

## 5) L’IA générative dans le paysage du ML

Comme dans ton schéma, l’IA générative se positionne principalement dans :
- **Non supervisé / auto-supervisé** (apprentissage de représentations et distributions)
- Avec des ponts vers le supervisé (fine-tuning, instruction datasets)
- Et parfois RL (RLHF pour aligner certains LLM)

En pratique, un produit GenAI moderne combine souvent :
1. Pré-entraînement auto-supervisé (Transformer)
2. Supervised fine-tuning
3. Alignement via préférences humaines (souvent RL)
4. Déploiement avec garde-fous (safety, monitoring)

---

## 6) MLOps : passer du prototype à la production

### Étapes
1. Versionner données et modèles (DVC/MLflow).
2. Pipelines reproductibles (train/eval/deploy).
3. Monitoring post-déploiement : drift, performance, latence.
4. Ré-entraînement périodique.

### Risques à surveiller
- Data leakage
- Biais et équité
- Surapprentissage (overfitting)
- Décalage de distribution (data drift)

---

## 7) Roadmap d’apprentissage (12 semaines)

### S1-S2 : Fondations
- Python data stack + stats + algèbre linéaire.

### S3-S4 : Supervisé classique
- Classification/régression avec scikit-learn.
- Feature engineering + validation croisée.

### S5-S6 : Non supervisé
- Clustering + PCA + visualisation.

### S7-S8 : Deep Learning
- MLP, CNN, entraînement GPU, régularisation.

### S9-S10 : Génératif
- Autoencoders, GAN, diffusion, introduction LLM.

### S11 : Reinforcement Learning
- MDP, Q-learning, PPO concepts.

### S12 : Projet de bout en bout
- Dataset réel, API de prédiction, tableau de bord métriques.

---

## 8) Projet fil rouge (très recommandé)

### Sujet
Prédiction + segmentation + génération de rapports automatiques pour un e-commerce.

### Modules
1. **Régression** : prédire le CA hebdo.
2. **Classification** : prédire churn client.
3. **Clustering** : segmenter les clients.
4. **GenAI (LLM)** : générer un résumé business hebdomadaire.
5. **MLOps** : pipeline CI/CD + monitoring.

### Livrables
- Notebook exploration
- Script d’entraînement
- API FastAPI
- Dashboard (Streamlit)
- Rapport technique + README

---

## 9) Bonnes pratiques universelles

- Commencer simple (baseline) avant modèle complexe.
- Toujours avoir un jeu de test isolé.
- Comparer avec des métriques alignées business.
- Vérifier robustesse (données manquantes, outliers).
- Documenter hypothèses et limites.

---

## 10) Résumé final (vision d’ensemble)

- **Supervisé** : tu as des labels → classification, régression, détection d’objets.
- **Non supervisé** : pas de labels → clustering, réduction de dimension, génératif.
- **Renforcement** : interaction + récompense → jeux, robotique, optimisation.
- **IA générative** : branche majeure moderne, surtout via diffusion et transformers.

Si tu veux, je peux te préparer la **version 2** de ce cours avec :
1. des TP guidés (exercice + correction),
2. un niveau “débutant / intermédiaire / avancé”,
3. et un plan orienté “trouver un stage/job ML”.
