import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("DATA/ANNONCES_CLEAN.csv")
os.makedirs("figures", exist_ok=True)

sns.set_theme(
    style="whitegrid",
    palette="deep",
    font_scale=1.1
)
# --------------------------------------------------
# 1. Partie 1 : On séléctionne seulement les appartements 
# --------------------------------------------------
df_appart = df[df["type_bien"].str.contains("appartements", case=False, na=False)]
df_appart["prix_m2"] = df_appart["prix"] / df_appart["surface"]

print("Nombre total d'annonces :", len(df))
print("Nombre d'appartements :", len(df_appart))
print("Nombre de villes :", df_appart["ville"].nunique())

print("\n===== NOMBRE D'APPARTEMENTS PAR VILLE =====")
print(df_appart["ville"].value_counts())

# --------------------------------------------------
# 2. Partie 2 : Calcul moyenne & médiane du prix au m²
# --------------------------------------------------
moyenne_m2 = df_appart["prix_m2"].mean()
mediane_m2 = df_appart["prix_m2"].median()

print("\n===== STATISTIQUES PRIX AU M² =====")
print(f"Prix moyen au m² : {moyenne_m2:.0f} €")
print(f"Prix médian au m² : {mediane_m2:.0f} €")

# --------------------------------------------------
# 3. Partie 3 : Répartition des prix selon les villes 
# --------------------------------------------------
prix_ville = (
    df_appart.groupby("ville")["prix_m2"].mean().sort_values(ascending=False)
)
print("\n===== PRIX AU M2 MOYEN PAR VILLE (Appartements) =====")
print(prix_ville.head(42))

plt.figure(figsize=(12, 8))
sns.barplot(
    x=prix_ville.head(42).values,
    y=prix_ville.head(42).index
)
plt.title("Prix moyen au m² des appartements par ville") 
plt.xlabel("Prix moyen au m² (€)")                        
plt.ylabel("Ville")
plt.tight_layout()
plt.savefig("figures/prix_moyen_m2_par_ville.png", dpi=300, bbox_inches='tight')
plt.show()


# --------------------------------------------------
# 4. Partie 4 : Corrélation entre surface et prix
# --------------------------------------------------
corr = df_appart["surface"].corr(df_appart["prix"])
print("\n===== CORRÉLATION =====")
print(f"Corrélation surface/prix : {corr:.3f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="surface",
    y="prix",
    data=df_appart,
    alpha=0.4,
    s=40
)
plt.xlim(10, 150)
plt.ylim(50000, 800000)
plt.title("Corrélation entre surface et prix")
plt.xlabel("Surface (m²)")
plt.ylabel("Prix (€)")
plt.tight_layout()
plt.savefig("figures/correlation_surface_prix.png", dpi=300, bbox_inches='tight')
plt.show()

# 2ème visualisation : évolution des prix moyens en fonction de la surface
df_grouped = (
    df_appart
    .groupby("surface", as_index=False)
    .agg(prix_moyen=("prix", "mean"))
    .sort_values("surface")
)

plt.figure(figsize=(10, 5))

plt.plot(
    df_grouped["surface"],
    df_grouped["prix_moyen"],
    marker="o",
    linestyle="-"
)

plt.title("Évolution des prix moyens en fonction de la surface")
plt.xlabel("Surface (m²)")
plt.ylabel("Prix moyen (€)")
plt.xlim(0, 350)
plt.tight_layout()
plt.savefig("figures/evolution_prix_m2_surface.png", dpi=300, bbox_inches='tight')
plt.show()

# 3ème visualisation : évolution du prix moyen au m² en fonction de la surface
df_grouped = (
    df_appart
    .groupby("surface", as_index=False)
    .agg(prix_m2_moyen=("prix_m2", "mean"))   
    .sort_values("surface")
)

plt.figure(figsize=(10, 5))
plt.plot(
    df_grouped["surface"],
    df_grouped["prix_m2_moyen"],
    marker="o",
    linestyle="-"
)

plt.title("Évolution du prix moyen au m² en fonction de la surface")  # 🔧 MODIF
plt.xlabel("Surface (m²)")
plt.ylabel("Prix moyen au m² (€)")                                   # 🔧 MODIF
plt.xlim(0, 350)
plt.tight_layout()
plt.savefig("figures/evolution_prix_m2_surface.png", dpi=300, bbox_inches="tight")
plt.show()
prix_m2_pieces = (
    df_appart
    .groupby("pieces")["prix_m2"]
    .mean()
    .reset_index()
)

# 4ème visualisation : prix moyen au m² selon le nombre de pièces
df_pieces = df_appart[df_appart["pieces"].between(1, 6)].copy()
df_pieces["pieces"] = df_pieces["pieces"].astype(int)

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_pieces,
    x="pieces",
    y="prix_m2"
)

plt.title("Distribution du prix au m² selon le nombre de pièces")
plt.xlabel("Nombre de pièces")
plt.ylabel("Prix au m² (€)")

plt.tight_layout()
plt.savefig("figures/boxplot_prix_m2_par_pieces.png", dpi=300, bbox_inches="tight")
plt.show()


# --------------------------------------------------
# 5. Partie 5 : Histogrammes et Boxplots
# --------------------------------------------------

# 5.1 Histogramme général du prix au m² par ville
plt.figure(figsize=(10, 6))
sns.histplot(
    df_appart["prix_m2"], 
    bins=30,
    edgecolor="black"
)
plt.xlim(0, 10000)
plt.title("Distribution du prix au m²")
plt.xlabel("Prix au m² (€)")
plt.ylabel("Nombre d'annonces")
plt.tight_layout()
plt.savefig("figures/histogramme_prix_m2.png", dpi=300, bbox_inches="tight")
plt.show()

# 5.2 Boxplot du prix au m² par ville
plt.figure(figsize=(24, 18))
sns.boxplot(
    data=df_appart,
    y="ville",
    x="prix_m2",
    order=sorted(df_appart["ville"].unique())
)
plt.xlim(1000, 15000)
plt.title("Boxplot du prix au m² par ville")
plt.xlabel("Prix au m² (€)")
plt.ylabel("Villes")
plt.tight_layout()
plt.savefig("figures/boxplot_prix_m2.png", dpi=300, bbox_inches='tight')
plt.show()


# 5.3 Les 10 villes les plus chères
villes_cheres = (
    df_appart.groupby("ville")["prix_m2"]
    .median()
    .sort_values(ascending=False)
    .head(10)
    .index
)

plt.figure(figsize=(14, 8))
sns.boxplot(
    x="ville",
    y="prix_m2",
    data=df_appart[df_appart["ville"].isin(villes_cheres)]
)
plt.xticks(rotation=45)
plt.title("10 villes les plus chères")
plt.tight_layout()
plt.savefig("figures/boxplot_10_villes_cheres.png", dpi=300, bbox_inches='tight')
plt.show()

# 5.4 Pairplot des variables numériques
df_pair = df_appart[["prix", "surface", "prix_m2", "pieces"]].dropna()
df_pair = df_pair.sample(min(len(df_pair), 400), random_state=1)

sns.pairplot(
    df_pair,
    diag_kind="kde",
    corner=True,
    plot_kws={"alpha": 0.5, "s": 40}
)
plt.suptitle("Pairplot – Relations entre variables", y=1.02)
plt.savefig("figures/pairplot_variables.png", dpi=300, bbox_inches='tight')
plt.show()


# --------------------------------------------------
print("\nAnalyse des appartements terminée !")

