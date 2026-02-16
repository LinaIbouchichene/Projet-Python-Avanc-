import pandas as pd
import re
from geopy.geocoders import Nominatim
import time
import math

df = pd.read_csv("ANNONCES_DETAILLEES.csv")

# ---- Extraction des nombres ----
def extraire_numero(valeur):
    if pd.isna(valeur):
        return None
    
    chiffres = re.findall(r'\d+', str(valeur))
    if len(chiffres) == 0:
        return None
    
    return int("".join(chiffres))

# --- Supprimer les doublons ----
df = df.drop_duplicates()


# --- Nettoyer les doublons ---
df["prix"] = (
    df["prix"].astype(str)
    .str.replace("€", "", regex=False)
    .str.replace(" ", "")
    .str.replace("n.c.", "")      
)
# --- Convertir les vides en None ---
df["prix"] = df["prix"].replace("", None) 
# --- Conversion SANS ERREUR ---
df["prix"] = df["prix"].astype(float)      


# --- Nettoyer la Colonne Surface ---
df["surface"] = df["surface"].astype(str)
df["surface"] = df["surface"].str.replace("m²", "", regex=False)
df["surface"] = df["surface"].str.replace(" ", "")
df["surface"] = df["surface"].str.replace("n.c.", "", regex=False)
df["surface"] = df["surface"].str.split("/").str[0]
df["surface"] = df["surface"].str.replace("\u202f", "")
df["surface"] = df["surface"].str.replace("\xa0", "")
df["surface"] = df["surface"].replace("", None)
df["surface"] = df["surface"].astype(float)


# --- Nettoyage de la colonne Type_bien ---
df["type_bien"] = df["type_bien"].str.replace(">", "", regex=False).str.strip()
df["type_bien"] = df["type_bien"].apply(
    lambda x: x.split()[0] if isinstance(x, str) else x
)
# --- Nettoyage de la colonne Code_postal ---
df["code_postal"] = df["code_postal"].astype("Int64")

# --- Calcul du Prix au m2 ---
df["prix_m2"] = (df["prix"] / df["surface"]).round()

# --- Nettoyage de la colonne Pièces ---
df["pieces"] = df["pieces"].astype(str).str.extract(r"(\d+)")
df["pieces"] = df["pieces"].astype("Int64")

# --- Nettoyage des nombres décimaux---
df["prix"] = df["prix"].astype("Int64")
df["surface"] = df["surface"].astype("Int64")
df["prix_m2"] = df["prix_m2"].astype("Int64")

# --- Geolocalisation des villes ---
print("Géolocalisation des villes…")

df["ville"] = df["ville"].str.replace(r"Paris.*", "Paris", regex=True)

geolocator = Nominatim(user_agent="clean_script")
coord_villes = {}

villes = df["ville"].dropna().unique()

for v in villes:
    try:
        loc = geolocator.geocode(v)
        if loc:
            coord_villes[v] = (loc.latitude, loc.longitude)
        else:
            coord_villes[v] = (None, None)
        
        print(f"{v} -> {coord_villes[v]}")
        time.sleep(1)
    except:
        coord_villes[v] = (None, None)

# --- Ajout des coordonnées ---
df["latitude"] = df["ville"].apply(lambda x: coord_villes.get(x, (None, None))[0])
df["longitude"] = df["ville"].apply(lambda x: coord_villes.get(x, (None, None))[1])

# --- Sauvegarde ---
df.to_csv("DATA/ANNONCES_CLEAN.csv", index=False)

print("Nettoyage terminé ! Fichier créé : ANNONCES_CLEAN.csv")
 
# --- Vérification des valeurs manquantes ---
print("\nValeurs manquantes par colonne :")
print(df.isna().sum())
print(df[df["ville"].isna()])
