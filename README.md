# Logistic Regression from Scratch

Cette implémentation utilise la **Descente de Gradient** pour minimiser la **Log Loss** (Cross-Entropy). Contrairement au SVM, ce modèle prédit une probabilité, ce qui le rend idéal pour des tâches où la confiance de la prédiction est aussi importante que la classe elle-même.

## 🧠 Concepts Clés
Le modèle utilise la fonction sigmoïde pour mapper les prédictions entre 0 et 1 :
$$P(y=1|x) = \sigma(w \cdot x + b)$$

La mise à jour des poids lors de la descente de gradient est définie par :
$$w = w - \alpha \cdot \left[ \frac{1}{n} \sum (\sigma(w \cdot x_i + b) - y_i) \cdot x_i \right]$$

## 🛠 Intégration Scikit-Learn
Comme pour le SVM, ce modèle est encapsulé pour respecter l'API `scikit-learn` :

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.logistic_regression import CustomLogisticRegression

model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', CustomLogisticRegression(learning_rate=0.1, n_iterations=1000))
])