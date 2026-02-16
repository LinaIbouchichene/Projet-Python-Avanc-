# 📊 Analyse du marché immobilier en France
**Web scraping & data analysis avec Python**
---

## ** Présentation du projet**
Ce projet vise à analyser le marché immobilier français à partir de données collectées automatiquement sur plusieurs plateformes d’annonces immobilières.  
À l’aide de techniques de **web scraping**, de **nettoyage de données**, d’**analyse statistique** et de **visualisation**, l’objectif est de comprendre comment le **prix au mètre carré** varie selon différents critères (localisation, surface, type de bien).

Ce projet s’inscrit dans le cadre d’un **projet académique de niveau Master 1**.

---

## ** Objectifs**
- Collecter automatiquement des annonces immobilières depuis plusieurs sites web  
- Nettoyer et structurer les données (prix, surface, localisation, type de bien, nombre de pièces, etc.)  
- Analyser les tendances du marché immobilier :
  - prix moyens et médians
  - variation selon la localisation
  - rapport prix / m²
  - évolution dans le temps
- Visualiser les résultats à l’aide de graphiques et d’un **tableau de bord interactif**

---

## ** Problématique**
**Comment le prix au mètre carré varie-t-il en fonction de la localisation, de la surface et du type de bien immobilier en France ?**

---

## ** Sources de données**
Les données sont collectées via web scraping (ou API lorsqu’elle est disponible) à partir des plateformes suivantes :  

- Leboncoin Immobilier  
- SeLoger  
- Logic-Immo  
- Bien’ici  




## ** Pipeline du projet**

### **1️⃣ Scraping des données**
- Récupération des pages HTML avec `requests`
- Parsing du contenu avec `BeautifulSoup`
- Extraction des informations suivantes :
  - Titre et description
  - Prix
  - Surface (m²)
  - Nombre de pièces
  - Adresse / ville / code postal
  - Type de bien (maison, appartement, studio…)

---

### **2️⃣ Nettoyage et structuration**
- Suppression des doublons
- Normalisation des formats (prix, surface, prix/m²)
- Extraction des valeurs numériques via expressions régulières
- Complétion des localisations manquantes par géocodage automatique (Nominatim)

---

### **3️⃣ Analyse statistique**
- Calcul du prix moyen et médian au m²
- Comparaison des prix selon les villes et régions
- Étude de la corrélation entre surface et prix
- Création d’histogrammes et de boxplots

---

### **4️ Visualisation cartographique**
- Création d’une carte interactive avec `folium`
- Chaque bien est représenté par un point géolocalisé
- Affichage des informations clés au survol

---

### **5️ Tableau de bord interactif**
Le tableau de bord permet à l’utilisateur de :
- Sélectionner une ville ou une région
- Filtrer par surface, prix ou nombre de pièces
- Visualiser dynamiquement les graphiques et la carte

---

## ** Options avancées**
- Actualisation automatique des données (cron, `schedule`)
- Modèle de prédiction des prix (régression linéaire)
- Mise en ligne du dashboard (Streamlit Cloud, Render)

