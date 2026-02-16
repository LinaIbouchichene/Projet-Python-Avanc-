#  Analyse du marché immobilier en France
**Web scraping & data analysis avec Python**
---

## ** À propos du projet**
Ce projet consiste à analyser le marché immobilier français à partir de données collectées automatiquement sur des sites d’annonces immobilières.  
L’objectif est de comprendre comment le **prix au mètre carré** varie selon différents critères : **localisation**, **surface** et **type de bien**.  

Le projet inclut :  
- Collecte des annonces immobilières (scraping ou API)  
- Nettoyage et structuration des données  
- Analyses statistiques et visualisations  
- Création d’un tableau de bord interactif avec Streamlit  
---

## ** Objectifs**
- Extraire automatiquement les informations clés des annonces immobilières : prix, surface, type de bien, nombre de pièces, localisation.  
- Nettoyer et structurer les données pour les rendre exploitables.  
- Étudier les tendances :  
  - Prix moyen et médian par ville  
  - Rapport prix / m²  
  - Corrélation entre surface et prix  
  - Visualisation des tendances par région ou type de bien  
- Créer une interface interactive pour explorer les données et les visualisations.  


## ** Pipeline du projet**
1. **Scraping** : extraction du titre, prix, surface, nombre de pièces, type de bien et localisation des annonces.  
2. **Nettoyage et structuration** : suppression des doublons, normalisation des formats, extraction des valeurs numériques et géocodage des adresses manquantes.  
3. **Analyse statistique** : calcul du prix moyen et médian, corrélations, distribution des prix par ville ou type de bien.  
4. **Visualisation cartographique** : carte interactive avec `folium` pour localiser les biens et afficher les prix.  
5. **Tableau de bord interactif** : filtrage par ville, prix, surface et nombre de pièces, affichage des graphiques et cartes dynamiques.


## ** Contenu du dépôt**
- `SRC/` : scripts Python pour le scraping, nettoyage, analyse et dashboard  
- `NOTEBOOKS/` : notebook d’exploration et tests 
- `DATA/` : Bases de données extraites  
- `README.md` : ce fichier  
- `RAPPORT/` : rapport final du projet  

## ** Exemples de visualisations**
- Histogramme du prix au m² par ville  
- Carte interactive des annonces géolocalisées  
- Évolution temporelle du prix moyen  
- Nuage de points surface vs prix
- 
## ** Options avancées**
- Actualisation automatique des données via `schedule` ou `cron`  
- Modèle de prédiction du prix d’un bien (régression linéaire)  
- Déploiement du dashboard sur Streamlit Cloud ou Render

---

> Ce projet est un exemple pédagogique montrant le pipeline complet de collecte, traitement et analyse des données immobilières avec Python.
