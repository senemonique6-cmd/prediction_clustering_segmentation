import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="DBSCAN - Graines de blé",
    page_icon="🌾",
    layout="wide"
)


st.title("🌾 Segmentation des graines de blé")
st.subheader("Analyse par DBSCAN")


st.write(
    """
    Cette application utilise le modèle DBSCAN entraîné
    sur les données des graines de blé afin d'identifier
    les différents groupes et les observations considérées
    comme des anomalies.
    """
)


# =====================================================
# CHARGEMENT DU MODELE
# =====================================================

@st.cache_resource
def load_model():

    return joblib.load("dbscan_wheat.pkl")


@st.cache_data
def load_data():

    return joblib.load("wheat_data.pkl")


try:

    dbscan_model = load_model()
    x = load_data()

    st.success("✅ Modèle DBSCAN et données chargés avec succès.")

except Exception as e:

    st.error(
        f"❌ Erreur lors du chargement des fichiers : {e}"
    )

    st.stop()


# =====================================================
# CONVERSION DES DONNEES
# =====================================================

if isinstance(x, np.ndarray):

    features = [
        "area A",
        "perimeter",
        "compactness",
        "length of kernel",
        "width of kernel",
        "asymmetry coefficient",
        "length of kernel groove"
    ]

    x = pd.DataFrame(
        x,
        columns=features
    )


# =====================================================
# INFORMATIONS SUR LE MODELE
# =====================================================

st.sidebar.header("⚙️ Paramètres DBSCAN")

st.sidebar.write(
    f"**eps :** {dbscan_model.eps}"
)

st.sidebar.write(
    f"**min_samples :** {dbscan_model.min_samples}"
)

st.sidebar.write(
    f"**Nombre de variables :** {x.shape[1]}"
)

st.sidebar.write(
    f"**Nombre d'observations :** {x.shape[0]}"
)


# =====================================================
# APERCU DES DONNEES
# =====================================================

st.header("📋 Données des graines")

st.dataframe(
    x.head(10),
    use_container_width=True
)


# =====================================================
# CLUSTERING
# =====================================================

st.header("🔍 Analyse DBSCAN")


# On récupère les labels du modèle déjà entraîné
if hasattr(dbscan_model, "labels_"):

    labels = dbscan_model.labels_

else:

    labels = dbscan_model.fit_predict(
        x.values
    )


# Vérification
if len(labels) != len(x):

    st.error(
        "Le nombre de labels ne correspond pas "
        "au nombre d'observations."
    )

    st.stop()


# Ajouter les clusters
results_df = x.copy()

results_df["Cluster"] = labels


# =====================================================
# STATISTIQUES
# =====================================================

number_observations = len(labels)

number_clusters = len(
    set(labels)
) - (1 if -1 in labels else 0)


number_anomalies = int(
    np.sum(labels == -1)
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "🌾 Nombre de graines",
        number_observations
    )


with col2:

    st.metric(
        "🔵 Nombre de clusters",
        number_clusters
    )


with col3:

    st.metric(
        "⚠️ Anomalies",
        number_anomalies
    )


# =====================================================
# REPARTITION DES CLUSTERS
# =====================================================

st.header("📊 Répartition des clusters")


cluster_counts = (
    pd.Series(labels)
    .value_counts()
    .sort_index()
)


cluster_counts_df = pd.DataFrame({

    "Cluster":
        cluster_counts.index.astype(str),

    "Nombre de graines":
        cluster_counts.values

})


st.dataframe(
    cluster_counts_df,
    use_container_width=True,
    hide_index=True
)


# =====================================================
# GRAPHIQUE DES CLUSTERS
# =====================================================

fig_bar = px.bar(

    cluster_counts_df,

    x="Cluster",

    y="Nombre de graines",

    text="Nombre de graines",

    title="Répartition des graines par cluster"

)


st.plotly_chart(
    fig_bar,
    use_container_width=True
)


# =====================================================
# PCA
# =====================================================

st.header("📍 Visualisation des clusters")


pca = PCA(
    n_components=2
)


x_pca = pca.fit_transform(
    x.values
)


pca_df = pd.DataFrame({

    "PC 1":
        x_pca[:, 0],

    "PC 2":
        x_pca[:, 1],

    "Cluster":
        labels.astype(str)

})


fig_pca = px.scatter(

    pca_df,

    x="PC 1",

    y="PC 2",

    color="Cluster",

    title="Visualisation des clusters DBSCAN",

    hover_data=[
        "Cluster"
    ]

)


st.plotly_chart(
    fig_pca,
    use_container_width=True
)


# =====================================================
# ANOMALIES
# =====================================================

st.header("⚠️ Anomalies détectées")


anomalies_df = results_df[
    results_df["Cluster"] == -1
]


if len(anomalies_df) > 0:

    st.warning(
        f"{len(anomalies_df)} anomalie(s) détectée(s)."
    )

    st.dataframe(
        anomalies_df,
        use_container_width=True
    )

else:

    st.success(
        "✅ Aucune anomalie détectée."
    )


# =====================================================
# RESULTATS COMPLETS
# =====================================================

st.header("📋 Résultats complets")


st.dataframe(
    results_df,
    use_container_width=True
)


# =====================================================
# TELECHARGEMENT
# =====================================================

csv_results = results_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(

    label="⬇️ Télécharger les résultats",

    data=csv_results,

    file_name="resultats_dbscan_wheat.csv",

    mime="text/csv"

)


# =====================================================
# INFORMATIONS
# =====================================================

st.info(
    """
    ℹ️ Dans DBSCAN, les observations appartenant au cluster
    -1 sont considérées comme des points aberrants ou des anomalies.
    """
)
