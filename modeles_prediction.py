import pandas as pd

data = pd.read_csv("DATA/ANNONCES_CLEAN.csv")
data = data[data["type_bien"] == "Appartements"]
data = data.dropna(subset=["surface", "pieces", "prix"])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# --------------------------------------------------------
# Premier modèle de prédiction : Régression linéaire
# --------------------------------------------------------

y = data["prix"]
X = data[["surface", "pieces", "latitude", "longitude"]]

# standardisation + régression linéaire
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]
)

# séparation train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# l'entrainement
pipeline.fit(X_train, y_train)

# la prédiction
y_pred = pipeline.predict(X_test)

# l'évaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² : {r2:.3f}")
print(f"RMSE : {rmse:.0f} €")

# --------------------------------------------------
# Deuxième modèle de prédiction : Random Forest
# --------------------------------------------------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Modèle Random Forest simple
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Entraînement
rf_model.fit(X_train, y_train)

# Prédiction
y_pred_rf = rf_model.predict(X_test)

# Évaluation
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"Random Forest - R² : {r2_rf:.3f}")
print(f"Random Forest - RMSE : {rmse_rf:.0f} €")


# --------------------------------------------------
# Résumé des performances des modèles
# --------------------------------------------------

resultats = pd.DataFrame({
    "Modèle": ["Régression linéaire", "Random Forest"],
    "R² (test)": [0.228, 0.400],
    "RMSE (€)": [168344, 148355]
})
print(resultats)
