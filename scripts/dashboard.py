import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import folium
from folium.features import DivIcon
from folium.plugins import MarkerCluster, HeatMap
import hashlib
import math

DATA_PATH = "DATA/ANNONCES_CLEAN.csv"
# Polygones grossiers (lat, lon) pour limiter les pins à la France métropolitaine + Corse
FR_MAINLAND_POLY = [
    (-5.2, 48.9),
    (-4.8, 47.8),
    (-1.8, 43.5),
    (-1.3, 42.7),
    (3.0, 42.5),
    (4.5, 42.7),
    (5.5, 42.9),
    (7.0, 43.1),
    (7.6, 43.7),
    (7.8, 45.0),
    (7.8, 49.0),
    (8.3, 50.1),
    (2.7, 51.1),
    (1.5, 50.5),
    (0.2, 50.6),
    (-1.9, 49.8),
    (-4.8, 48.7),
]
FR_CORSICA_POLY = [
    (8.5, 42.2),
    (8.6, 41.4),
    (9.4, 41.3),
    (9.6, 42.0),
    (9.1, 42.6),
]
ARR_CENTROIDS = {
    75001: (48.8625, 2.3446),
    75002: (48.8687, 2.3431),
    75003: (48.8629, 2.3601),
    75004: (48.8543, 2.3550),
    75005: (48.8442, 2.3512),
    75006: (48.8519, 2.3325),
    75007: (48.8583, 2.3123),
    75008: (48.8721, 2.3095),
    75009: (48.8756, 2.3386),
    75010: (48.8753, 2.3605),
    75011: (48.8603, 2.3788),
    75012: (48.8360, 2.4065),
    75013: (48.8284, 2.3561),
    75014: (48.8329, 2.3230),
    75015: (48.8414, 2.2922),
    75016: (48.8580, 2.2708),
    75017: (48.8875, 2.3071),
    75018: (48.8920, 2.3453),
    75019: (48.8897, 2.3843),
    75020: (48.8646, 2.4070),
}


@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Charge les données nettoyées et prépare quelques colonnes utiles."""
    df = pd.read_csv(path)

    df["prix"] = df["prix"].astype(float)
    df["surface"] = df["surface"].astype(float)
    df["pieces"] = df["pieces"].astype("Int64")
    df["code_postal"] = df["code_postal"].astype("Int64")

    # Ne conserve que les appartements (mot-clé explicite) et recalcule le prix au m²
    df = df[df["type_bien"].str.contains("appartements", case=False, na=False)]
    df["prix_m2"] = df["prix"] / df["surface"]

    # Départements (proxy simple pour la notion de région)
    dept = df["code_postal"]
    df["departement"] = (
        dept.astype(str).str.zfill(5).str[:2].where(dept.notna(), None)
    )
    # Ajoute des coordonnées jitterées stables pour disperser les pins autour du centre-ville
    df = jitter_coordinates(df, radius_m=5_000)
    return df


def filter_data(
    df: pd.DataFrame,
    city: str,
    departement: str,
    price_range: tuple[float, float],
    surface_range: tuple[float, float],
    pieces_selection: list[int],
    include_missing_price_surface: bool = False,
) -> pd.DataFrame:
    """Applique les filtres sélectionnés par l'utilisateur."""
    mask = pd.Series(True, index=df.index)

    if city != "Toutes":
        mask &= df["ville"] == city

    if departement != "Tous":
        mask &= df["departement"] == departement

    if include_missing_price_surface:
        mask &= df["prix"].between(price_range[0], price_range[1], inclusive="both") | df[
            "prix"
        ].isna()
        mask &= df["surface"].between(
            surface_range[0], surface_range[1], inclusive="both"
        ) | df["surface"].isna()
    else:
        mask &= df["prix"].between(price_range[0], price_range[1], inclusive="both")
        mask &= df["surface"].between(
            surface_range[0], surface_range[1], inclusive="both"
        )

    if pieces_selection:
        mask &= df["pieces"].isin(pieces_selection)

    return df[mask].copy()


def limit_ads_by_city(df: pd.DataFrame, max_ads_per_city: int = 100) -> pd.DataFrame:
    """Limite à N annonces par ville (priorité aux plus chères dans chaque ville)."""
    return (
        df.sort_values("prix", ascending=False)
        .groupby("ville", group_keys=False)
        .head(max_ads_per_city)
    )


def limit_ads_by_city_map(
    df: pd.DataFrame, city_caps: dict[str, int], default_cap: int = 50
) -> pd.DataFrame:
    """Limite par ville selon un barème fourni."""
    caps = city_caps or {}
    return (
        df.sort_values("prix", ascending=False)
        .groupby("ville", group_keys=False)
        .apply(lambda g: g.head(caps.get(g.name, default_cap)))
    )


def format_price(val: float) -> str:
    return f"{val:,.0f} €".replace(",", " ") if pd.notna(val) else "Prix n.c."


def price_bucket(val: float) -> str:
    """Retourne un code couleur en fonction du prix pour un repère lisible sur la carte."""
    if pd.isna(val):
        return "#94a3b8"  # gris
    if val < 150_000:
        return "#10b981"  # vert
    if val < 300_000:
        return "#0ea5e9"  # bleu
    if val < 500_000:
        return "#f59e0b"  # orange
    return "#ef4444"      # rouge


def _stable_random_pair(seed: str) -> tuple[float, float]:
    """Renvoie deux nombres pseudo-aléatoires stables dans [0,1)."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    r1 = int.from_bytes(digest[:8], "big") / 2**64
    r2 = int.from_bytes(digest[8:16], "big") / 2**64
    return r1, r2


def _point_in_polygon(lon: float, lat: float, poly: list[tuple[float, float]]) -> bool:
    """Test point-in-polygon (algorithme ray casting)."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        lon1, lat1 = poly[i]
        lon2, lat2 = poly[(i + 1) % n]
        intersects = ((lat1 > lat) != (lat2 > lat)) and (
            lon < (lon2 - lon1) * (lat - lat1) / (lat2 - lat1 + 1e-12) + lon1
        )
        if intersects:
            inside = not inside
    return inside


def _in_france(lat: float, lon: float) -> bool:
    """Retourne True si le point est dans un des polygones France + Corse."""
    return _point_in_polygon(lon, lat, FR_MAINLAND_POLY) or _point_in_polygon(
        lon, lat, FR_CORSICA_POLY
    )


def jitter_coordinates(df: pd.DataFrame, radius_m: float = 5_000) -> pd.DataFrame:
    """
    Disperse chaque annonce dans un rayon donné (mètres) de façon stable
    en fonction de la ville pour éviter les pins superposés.
    """
    out = df.copy()
    out[["latitude", "longitude"]] = out[["latitude", "longitude"]].apply(
        pd.to_numeric, errors="coerce"
    )

    # Utilise la moyenne par ville pour combler les lat/lon manquantes.
    city_means = (
        out.groupby("ville")[["latitude", "longitude"]]
        .transform(lambda g: g.fillna(g.mean()))
    )
    out[["latitude", "longitude"]] = out[["latitude", "longitude"]].fillna(city_means)
    if out["latitude"].isna().any() or out["longitude"].isna().any():
        # Fallback très grossier : moyenne globale pour éviter de perdre ces lignes.
        out["latitude"] = out["latitude"].fillna(out["latitude"].mean())
        out["longitude"] = out["longitude"].fillna(out["longitude"].mean())

    meters_per_deg_lat = 111_320  # approx

    def jitter_row(row: pd.Series) -> pd.Series:
        base_lat, base_lon = row["latitude"], row["longitude"]
        radius_local = radius_m

        # Pour Paris, ancre sur le centre de l'arrondissement si connu
        cp = row["code_postal"]
        if row.get("ville") == "Paris" and pd.notna(cp):
            cp_int = int(cp)
            if cp_int in ARR_CENTROIDS:
                base_lat, base_lon = ARR_CENTROIDS[cp_int]
                radius_local = 800  # rayon réduit pour rester dans l'arrondissement

        if pd.isna(base_lat) or pd.isna(base_lon):
            return pd.Series({"latitude_jittered": None, "longitude_jittered": None})

        seed = f"{row['ville']}-{row['titre']}-{row['prix']}-{row['surface']}"
        r1, r2 = _stable_random_pair(seed)

        distance_m = math.sqrt(r1) * radius_local  # uniform dans le disque
        angle = 2 * math.pi * r2

        def offset(dist_m: float, ang: float) -> tuple[float, float]:
            delta_lat = (dist_m * math.cos(ang)) / meters_per_deg_lat
            meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(base_lat))
            if abs(meters_per_deg_lon) < 1e-6:
                meters_per_deg_lon = meters_per_deg_lat
            delta_lon = (dist_m * math.sin(ang)) / meters_per_deg_lon
            return base_lat + delta_lat, base_lon + delta_lon

        # Essaie plusieurs fois en réduisant le rayon pour rester dans la France.
        for scale in [1.0, 0.7, 0.5, 0.3, 0.0]:
            dist_try = distance_m * scale
            lat_j, lon_j = offset(dist_try, angle)
            if _in_france(lat_j, lon_j):
                return pd.Series(
                    {"latitude_jittered": lat_j, "longitude_jittered": lon_j}
                )

        # Fallback : pas trouvé, on reste sur la coordonnée de base.
        return pd.Series(
            {"latitude_jittered": base_lat, "longitude_jittered": base_lon}
        )

    out[["latitude_jittered", "longitude_jittered"]] = out.apply(
        jitter_row, axis=1
    )
    return out


def make_map(df: pd.DataFrame) -> folium.Map | None:
    """Construit une carte Folium compacte avec clustering des pins."""
    geo_df = df.dropna(subset=["latitude_jittered", "longitude_jittered"])
    if geo_df.empty:
        return None

    center = [
        geo_df["latitude_jittered"].mean(),
        geo_df["longitude_jittered"].mean(),
    ]
    fmap = folium.Map(
        location=center or [48.8566, 2.3522], zoom_start=6, tiles="CartoDB positron"
    )
    cluster = MarkerCluster().add_to(fmap)

    for _, row in geo_df.iterrows():
        prix_txt = format_price(row["prix"])
        prix_m2_txt = (
            f"{row['prix_m2']:,.0f} €/m²".replace(",", " ")
            if pd.notna(row["prix_m2"])
            else "Prix/m² n.c."
        )

        label_color = price_bucket(row["prix"])
        label_html = (
            f"<div style='background:{label_color};"
            f"color:white;padding:4px 8px;border-radius:6px;"
            f"font-weight:700;font-size:12px;white-space:nowrap;"
            f"box-shadow:0 2px 6px rgba(0,0,0,0.25);'>"
            f"{prix_txt}</div>"
        )

        tooltip = f"{prix_txt} — {row['titre'] or 'Titre indisponible'}"
        popup = folium.Popup(
            html=(
                f"<b>{row['titre']}</b><br>"
                f"{row['ville'] or ''} {row['code_postal'] or ''}<br>"
                f"{prix_txt} · {row['surface'] or 'n.c.'} m²<br>"
                f"Type : {row['type_bien'] or 'n.c.'}<br>"
                f"{prix_m2_txt}<br>"
                f"<a href='{row['url']}' target='_blank'>Voir l'annonce</a>"
            ),
            max_width=320,
        )

        folium.Marker(
            location=[row["latitude_jittered"], row["longitude_jittered"]],
            icon=DivIcon(html=label_html),
            tooltip=tooltip,
            popup=popup,
        ).add_to(cluster)

    # Ajuste le zoom aux données si possible
    bounds = [
        [geo_df["latitude_jittered"].min(), geo_df["longitude_jittered"].min()],
        [geo_df["latitude_jittered"].max(), geo_df["longitude_jittered"].max()],
    ]
    fmap.fit_bounds(bounds, padding=(10, 10))

    return fmap


def make_heatmap(df: pd.DataFrame) -> folium.Map | None:
    """Heatmap densité annonces par ville."""
    geo_df = df.dropna(subset=["latitude_jittered", "longitude_jittered"])
    if geo_df.empty:
        return None

    grouped = (
        geo_df.groupby("ville")
        .agg(
            lat=("latitude_jittered", "mean"),
            lon=("longitude_jittered", "mean"),
            weight=("ville", "size"),
        )
        .reset_index()
    )

    fmap = folium.Map(location=[46.6, 2.6], zoom_start=5.5, tiles="CartoDB positron")
    heat_data = grouped[["lat", "lon", "weight"]].values.tolist()
    HeatMap(
        heat_data,
        radius=25,
        blur=30,
        max_zoom=6,
        min_opacity=0.2,
    ).add_to(fmap)

    bounds = [
        [grouped["lat"].min(), grouped["lon"].min()],
        [grouped["lat"].max(), grouped["lon"].max()],
    ]
    fmap.fit_bounds(bounds, padding=(10, 10))
    return fmap


def _interpolate_color(ratio: float) -> str:
    """Interpole entre bleu (#3b82f6) et rouge (#ef4444)."""
    ratio = max(0.0, min(1.0, ratio))
    c1 = (0x3B, 0x82, 0xF6)
    c2 = (0xEF, 0x44, 0x44)
    r = int(c1[0] + (c2[0] - c1[0]) * ratio)
    g = int(c1[1] + (c2[1] - c1[1]) * ratio)
    b = int(c1[2] + (c2[2] - c1[2]) * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def make_bubble_map(df: pd.DataFrame) -> folium.Map | None:
    """Carte bulles par ville : taille = nb annonces, couleur = prix médian/m²."""
    geo_df = df.dropna(subset=["latitude_jittered", "longitude_jittered", "ville"])
    if geo_df.empty:
        return None

    grouped = (
        geo_df.groupby("ville")
        .agg(
            lat=("latitude_jittered", "mean"),
            lon=("longitude_jittered", "mean"),
            prix_m2=("prix_m2", "median"),
            weight=("ville", "size"),
            url=("url", "first"),
        )
        .reset_index()
    )

    if grouped["prix_m2"].dropna().empty:
        return None

    p_min, p_max = grouped["prix_m2"].min(), grouped["prix_m2"].max()
    fmap = folium.Map(location=[46.6, 2.6], zoom_start=5.5, tiles="CartoDB positron")

    for _, row in grouped.iterrows():
        prix = row["prix_m2"]
        if pd.isna(prix):
            continue
        ratio = 0 if p_max == p_min else (prix - p_min) / (p_max - p_min)
        color = _interpolate_color(ratio)
        # Rayon linéaire (échelle douce) : 50 -> ~18, 120 -> ~42
        radius = max(6, min(45, 0.35 * row["weight"]))
        popup_html = (
            f"<b>{row['ville']}</b><br>"
            f"Prix médian : {prix:,.0f} €/m²<br>"
            f"Annonces : {row['weight']}<br>"
            f"<a href='{row['url']}' target='_blank'>Voir une annonce</a>"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.65,
            weight=1,
            popup=popup_html,
            tooltip=f"{row['ville']} · {prix:,.0f} €/m² · {row['weight']} annonces",
        ).add_to(fmap)

    bounds = [
        [grouped["lat"].min(), grouped["lon"].min()],
        [grouped["lat"].max(), grouped["lon"].max()],
    ]
    fmap.fit_bounds(bounds, padding=(10, 10))

    legend_html = """
    <div style="
        position: absolute;
        top: 12px;
        left: 12px;
        z-index: 9999;
        background: white;
        padding: 10px 12px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        font-size: 12px;
        line-height: 1.4;
        border: 1px solid #e5e7eb;
        pointer-events: auto;
    ">
      <div style="font-weight: 700; margin-bottom: 8px;">Légende</div>
      <div style="margin-bottom: 6px;">
        Couleur : prix médian au m²
        <div style="
            width: 160px;
            height: 10px;
            margin-top: 6px;
            background: linear-gradient(90deg, #3b82f6 0%, #ef4444 100%);
            border-radius: 6px;
            border: 1px solid #e5e7eb;
        "></div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:#475569;">Bas</span>
            <span style="color:#475569;">Haut</span>
        </div>
      </div>
      <div>
        Taille : nombre d'annonces (échelle log)
        <div style="display:flex;align-items:center;gap:10px;margin-top:6px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#94a3b8;"></div><span style="color:#475569;">Peu</span>
            <div style="width:20px;height:20px;border-radius:50%;background:#94a3b8;"></div><span style="color:#475569;">Plus</span>
            <div style="width:40px;height:40px;border-radius:50%;background:#94a3b8;"></div><span style="color:#475569;">Beaucoup</span>
        </div>
      </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    return fmap


def price_distribution_chart(df: pd.DataFrame):
    data = df["prix_m2"].dropna()
    if data.empty:
        st.info("Pas assez de données pour l'histogramme.")
        return
    low, high = data.quantile([0.01, 0.99])
    clipped = data.clip(lower=low, upper=high)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(clipped, bins=30, color="#2563eb", edgecolor="black")
    ax.set_xlabel("Prix au m² (€)")
    ax.set_ylabel("Nombre d'annonces")
    ax.set_title("Distribution du prix au m²")
    st.pyplot(fig)


def scatter_surface_price(df: pd.DataFrame):
    data = df.dropna(subset=["surface", "prix"])
    if data.empty:
        st.info("Pas de données suffisantes pour afficher la corrélation.")
        return
    surf_low, surf_high = data["surface"].quantile([0.01, 0.99])
    price_low, price_high = data["prix"].quantile([0.01, 0.99])
    clipped = data[
        data["surface"].between(surf_low, surf_high)
        & data["prix"].between(price_low, price_high)
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(clipped["surface"], clipped["prix"], alpha=0.4, s=30, color="#10b981")
    ax.set_xlabel("Surface (m²)")
    ax.set_ylabel("Prix (€)")
    ax.set_title("Corrélation surface / prix")
    st.pyplot(fig)


def price_per_piece_boxplot(df: pd.DataFrame):
    """Boxplot du prix au m² par nombre de pièces."""
    df_clean = df.dropna(subset=["prix_m2", "pieces"])
    if df_clean.empty:
        st.info("Pas assez de données pour le boxplot prix/m² par pièces.")
        return
    low, high = df_clean["prix_m2"].quantile([0.01, 0.99])
    df_clean = df_clean[df_clean["prix_m2"].between(low, high)]
    grouped = [grp["prix_m2"].values for _, grp in df_clean.groupby("pieces")]
    labels = [str(int(p)) for p in sorted(df_clean["pieces"].unique())]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(grouped, labels=labels, patch_artist=True)
    ax.set_xlabel("Nombre de pièces")
    ax.set_ylabel("Prix au m² (€)")
    ax.set_title("Prix au m² par nombre de pièces")
    st.pyplot(fig)


def top_cities_bar(df: pd.DataFrame, top_n: int = 10):
    """Bar chart des villes triées par prix médian au m²."""
    df_city = df.dropna(subset=["prix_m2", "ville"])
    if df_city.empty:
        st.info("Pas assez de données pour le classement des villes.")
        return
    stats = (
        df_city.groupby("ville")["prix_m2"]
        .median()
        .sort_values(ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    stats.plot(kind="bar", color="#6366f1", ax=ax)
    ax.set_ylabel("Prix médian au m² (€)")
    ax.set_title(f"Top {top_n} villes par prix médian au m²")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Immobilier – Dashboard", layout="wide")
    st.title("Tableau de bord immobilier")
    st.caption("Navigation avec les flèches : Carte ↔ Tableau de bord interactif.")

    df = load_data()
    if df.empty:
        st.error("Aucune donnée disponible.")
        return

    pages = ["Carte interactive", "Tableau de bord interactif"]
    if "page_idx" not in st.session_state:
        st.session_state["page_idx"] = 0

    # --- Navigation par flèches ---
    col_nav = st.columns([1, 6, 1])
    with col_nav[0]:
        if st.button("◀", disabled=st.session_state["page_idx"] == 0):
            st.session_state["page_idx"] -= 1
    with col_nav[2]:
        if st.button("▶", disabled=st.session_state["page_idx"] == len(pages) - 1):
            st.session_state["page_idx"] += 1
    st.markdown(f"**Page : {pages[st.session_state['page_idx']]}**")

    # Page 1 : carte France sans filtres (carte clusters + bulles)
    if pages[st.session_state["page_idx"]] == "Carte interactive":
        st.subheader("Carte France : clusters (caps par ville)")
        city_caps = {
            "Paris": 120,
            "Lyon": 120,
            "Marseille": 120,
            "Bordeaux": 100,
            "Toulouse": 100,
            "Lille": 100,
        }
        base_df = limit_ads_by_city_map(df, city_caps, default_cap=50)

        fmap_cluster = make_map(base_df)
        if fmap_cluster:
            st.components.v1.html(fmap_cluster._repr_html_(), height=700)
        else:
            st.info("Aucune coordonnée disponible pour la carte clusters.")

        st.markdown("Carte France : bulles (taille = nb annonces, couleur = prix médian/m²)")
        fmap_bubble = make_bubble_map(base_df)
        if fmap_bubble:
            st.components.v1.html(fmap_bubble._repr_html_(), height=700)
        else:
            st.info("Aucune coordonnée disponible pour la carte bulles.")
        return

    # --- Filtres (pour le tableau de bord) ---
    st.sidebar.header("Filtres")
    villes = ["Toutes"] + sorted(v for v in df["ville"].dropna().unique())
    depts = ["Tous"] + sorted(v for v in df["departement"].dropna().unique())

    selected_city = st.sidebar.selectbox("Ville", villes, index=0)
    selected_dept = st.sidebar.selectbox("Département (proxy région)", depts)

    base_df = df
    if base_df.empty:
        st.warning("Aucune annonce pour ces types de biens.")
        return

    price_min, price_max = (
        int(base_df["prix"].min(skipna=True)),
        int(base_df["prix"].max(skipna=True)),
    )
    surface_min, surface_max = (
        float(base_df["surface"].min(skipna=True)),
        float(base_df["surface"].max(skipna=True)),
    )

    price_range = st.sidebar.slider(
        "Prix (€)",
        min_value=price_min,
        max_value=price_max,
        value=(price_min, price_max),
        step=10_000,
    )

    surface_range = st.sidebar.slider(
        "Surface (m²)",
        min_value=float(surface_min),
        max_value=float(surface_max),
        value=(float(surface_min), float(surface_max)),
        step=5.0,
    )

    pieces_options = sorted(int(p) for p in base_df["pieces"].dropna().unique())
    pieces_selection = st.sidebar.multiselect("Nombre de pièces", pieces_options)

    filtered_for_count = filter_data(
        base_df,
        city=selected_city,
        departement=selected_dept,
        price_range=price_range,
        surface_range=surface_range,
        pieces_selection=pieces_selection,
        include_missing_price_surface=True,
    )

    filtered_raw = filter_data(
        base_df,
        city=selected_city,
        departement=selected_dept,
        price_range=price_range,
        surface_range=surface_range,
        pieces_selection=pieces_selection,
    )
    city_caps = {
        "Paris": 120,
        "Lyon": 120,
        "Marseille": 120,
        "Bordeaux": 100,
        "Toulouse": 100,
        "Lille": 100,
    }
    filtered = limit_ads_by_city_map(filtered_raw, city_caps, default_cap=50)

    if pages[st.session_state["page_idx"]] == "Carte interactive":
        st.subheader("Carte interactive (pins avec prix au survol)")
        fmap = make_map(filtered)
        if fmap:
            st.components.v1.html(fmap._repr_html_(), height=700)
        else:
            st.info("Aucune coordonnée disponible pour tracer la carte.")

    if pages[st.session_state["page_idx"]] == "Tableau de bord interactif":
        st.subheader("Données filtrées")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Annonces", len(filtered_for_count))
        col_b.metric("Prix moyen (€)", f"{filtered['prix'].mean():,.0f}".replace(",", " "))
        col_c.metric(
            "Prix moyen au m² (€)",
            f"{filtered['prix_m2'].mean():,.0f}".replace(",", " "),
            )
        col_d.metric(
            "Surface médiane (m²)",
            f"{filtered['surface'].median():,.0f}".replace(",", " "),
        )

        st.dataframe(
            filtered[
                [
                    "titre",
                    "ville",
                    "code_postal",
                    "prix",
                    "surface",
                    "pieces",
                    "type_bien",
                    "prix_m2",
                ]
            ].reset_index(drop=True),
            use_container_width=True,
            height=400,
        )

        col1, col2 = st.columns(2)
        with col1:
            if not filtered["prix_m2"].dropna().empty:
                price_distribution_chart(filtered)
            else:
                st.info("Pas de données suffisantes pour afficher l'histogramme.")
        with col2:
            if not filtered[["surface", "prix"]].dropna().empty:
                scatter_surface_price(filtered.dropna(subset=["surface", "prix"]))
            else:
                st.info("Pas de données suffisantes pour afficher la corrélation.")

        col3, col4 = st.columns(2)
        with col3:
            price_per_piece_boxplot(filtered)
        with col4:
            top_cities_bar(filtered)

        st.subheader("Carte interactive")
        fmap = make_map(filtered)
        if fmap:
            st.components.v1.html(fmap._repr_html_(), height=700)
        else:
            st.info("Aucune coordonnée disponible pour tracer la carte.")

    st.markdown(
        "Astuce : lancer l'app avec `streamlit run Dashboard.py` depuis le répertoire du projet."
    )


if __name__ == "__main__":
    main()
