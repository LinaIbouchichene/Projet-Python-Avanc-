import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random


# --------------------------------------------------------------
# 1. Partie 1 : Scrapper les pages web selon les villes et pages 
# --------------------------------------------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9"
}

def get_page(city_code, page_number):
    url = f"https://www.etreproprio.com/annonces/{city_code}-r0.odd.g{page_number}"
    response = requests.get(url, headers=HEADERS, timeout=10)
    return BeautifulSoup(response.content, "lxml")

def extract_info(card):
    
    # ---- Titre ----
    title = card.select_one(".ep-title")
    title = title.get_text(strip=True) if title else None

    # ---- Villes ----
    city = card.select_one(".ep-city")
    city = city.get_text(strip=True) if city else None

    # ---- Prix ----
    price = card.select_one(".ep-price")
    price = price.get_text(strip=True) if price else None

    # ---- Surface ---- 
    surface = card.select_one(".ep-area")
    surface = surface.get_text(strip=True) if surface else None

    # ---- URL ----
    link_tag = card.find_parent("a")
    link = link_tag["href"] if link_tag else None
    if link and not link.startswith("https"):
        link = "https://www.etreproprio.com" + link

    return {
        "titre": title,
        "ville": city,
        "prix": price,
        "surface": surface,
        "url": link
    }


villes = ["thflcpo.lc94080", #Vincennes 
          "thflcpo.lc69266", #Villeurbanne
          "thflcpo.lc75056", #Paris
          "thflcpo.lc31555", #Toulouse
          "thflcpo.lc06088", #Nice
          "thflcpo.lc44109", #Nantes 
          "thflcpo.lc34172", #Montpellier
          "thflcpo.lc67482", #Strasbourg
          "thflcpo.lc59350", #Lille
          "thflcpo.lc37261", #Tours
          "thflcpo.lc33063", #Bordeaux
          "thflcpo.lc35238", #Rennes
          "thflcpo.lc51454", #Reims
          "thflcpo.lc76351", #Le Havre
          "thflcpo.lc42218", #Saint-étienne
          "thflcpo.lc83137", #Toulon
          "thflcpo.lc38185", #Grenoble
          "thflcpo.lc21231", #Dijon
          "thflcpo.lc49007", #Angers
          "thflcpo.lc30189", #Nimes
          "thflcpo.lc63113", #Clermont-Ferrand
          "thflcpo.lc72181", #Le Mans
          "thflcpo.lc13001", #Aix en provence
          "thflcpo.lc29019", #Brest
          "thflcpo.lc80021", #Amiens
          "thflcpo.lc87085", #Limoges
          "thflcpo.lc74010", #Annecy
          "thflcpo.lc57463", #Metz
          "thflcpo.lc66136", #Perpignan
          "thflcpo.lc45234", #Orléans
          "thflcpo.lc76540", #Rouen
          "thflcpo.lc14118", #Caen
          "thflcpo.lc68224", #Mulhouse
          "thflcpo.lc54395", #Nancy
          "thflcpo.lc97411", #Saint Denis
          "thflcpo.lc84007", #Avignon
          "thflcpo.lc93048", #Montreuil
          "thflcpo.lc17300", #La Rochelle
          "thflcpo.lc64445", #Pau
          "thflcpo.lc86194", #Poitiers
          "thflcpo.lc28085", #Chartres
          ]

data = []

print("\n===== DÉBUT DU SCRAPING (41 VILLES) =====\n")

for city_code in villes:
    print(f"\n------------------------------")
    print(f"   Ville : {city_code}")
    print(f"------------------------------")

    for page in range(1, 7):
        print(f"➡️  Page {page}")

        soup = get_page(city_code, page)
        cards = soup.select(".card-cla-search")

        if not cards:
            print(" Aucune annonce -> arrêt pour cette ville.")
            break

        for card in cards:
            info = extract_info(card)
            info["code_ville"] = city_code
            info["page"] = page
            data.append(info)

        time.sleep(random.uniform(1, 2))

df = pd.DataFrame(data)
df.to_csv("DATA/ANNONCES_RAW.csv", index=False)

print("\n===== SCRAPING TERMINÉ =====")
print(f"Total annonces récupérées : {len(df)}")

# -----------------------------------------------
# 2. Partie 2 : Scrapper les annonces une par une 
# -----------------------------------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9"
}

INPUT_FILE = "DATA/ANNONCES_RAW.csv"
OUTPUT_FILE = "DATA/ANNONCES_DETAILLEES.csv"

def scrape_detail_page(url):

    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.content, "lxml")

    except:
        print(f" Erreur de connexion pour : {url}")
        return None

    # ---- Titre ----
    titre = soup.select_one("h1")
    titre = titre.get_text(strip=True) if titre else None


    # --- Ville et Code Postal ----
    loc_tag = soup.select_one(".ep-loc")

    ville = None
    code_postal = None

    if loc_tag:
    
        texte = loc_tag.get_text(" ", strip=True)
        texte = texte.replace("—", "").strip()
        elements = texte.split()
        code_postal = elements[-1]
        ville = " ".join(elements[:-1])

    # ---- Type du bien ----
    type_tag = soup.select_one("div.ep-breadcrumb-cla-dir ol li:nth-child(2)")
    type_bien = type_tag.get_text(strip=True) if type_tag else None

    # ---- Nombre de pièces ----
    pieces_tag = soup.select_one(".ep-room")
    pieces = pieces_tag.get_text(strip=True) if pieces_tag else None

    # ---- Surface ---- 
    surface_tag = soup.select_one(".ep-area")
    surface = surface_tag.get_text(strip=True) if surface_tag else None

    # ---- Prix -----
    prix = soup.select_one(".ep-price")
    prix = prix.get_text(strip=True) if prix else None

    return {
        "titre": titre,
        "ville": ville,
        "code_postal": code_postal,
        "prix": prix,
        "surface": surface,
        "type_bien": type_bien,
        "pieces": pieces,
        "url": url
    }

df = pd.read_csv(INPUT_FILE)
urls = df["url"].unique()

print(f" Nombre d'annonces à scraper : {len(urls)}")

# Création du CSV final
pd.DataFrame(columns=[
    "titre","ville","code_postal","prix",
    "surface", "type_bien", "pieces", "url"
]).to_csv(OUTPUT_FILE, index=False)


for i, url in enumerate(urls):

    print(f"\n({i+1}/{len(urls)}) Scraping : {url}")

    data = scrape_detail_page(url)

    if data:
        pd.DataFrame([data]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        print("Données ajoutées")

    pause = random.uniform(1.2, 2.8)
    time.sleep(pause)

    if i % 20 == 0 and i != 0:
        print("⏸ Pause longue de 10 secondes…")
        time.sleep(10)


print("\n Scraping détaillé terminé !")
print(f" Données enregistrées dans : {OUTPUT_FILE}")





