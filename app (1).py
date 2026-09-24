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


# =====================================================
# TITRE
# =====================================================

st.title("🌾 Segmentation des graines de blé avec DBSCAN")

st.write(
    """
    Cette application présente les résultats du modèle
    DBSCAN entraîné sur les données des graines de blé.
    """
)


# =====================================================
# CHARGEMENT DU MODELE
# =====================================================

@st.cache_resource
def load_model():

    return joblib.load("dbscan_wheat.pkl")


try:

    dbscan_model = load_model()

    st.success("✅ Modèle DBSCAN chargé avec succès !")

except Exception as e:

    st.error(
        f"❌ Erreur lors du chargement du modèle : {e}"
    )

    st.stop()


# =====================================================
# INFORMATIONS DU MODELE
# =====================================================

st.header("⚙️ Paramètres du modèle")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "eps",
        dbscan_model.eps
    )

with col2:

    st.metric(
        "min_samples",
        dbscan_model.min_samples
    )


# =====================================================
# LABELS DBSCAN
# =====================================================

if hasattr(dbscan_model, "labels_"):

    labels = dbscan_model.labels_

else:

    st.warning(
        "Le modèle ne contient pas les labels d'entraînement."
    )

    st.stop()


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


st.header("📊 Résultats du clustering")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "🌾 Observations",
        number_observations
    )


with col2:

    st.metric(
        "🔵 Clusters",
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

st.header("📈 Répartition des clusters")


cluster_counts = (
    pd.Series(labels)
    .value_counts()
    .sort_index()
)


results = pd.DataFrame({

    "Cluster":
        cluster_counts.index,

    "Nombre d'observations":
        cluster_counts.values

})


st.dataframe(
    results,
    use_container_width=True,
    hide_index=True
)


# =====================================================
# GRAPHIQUE
# =====================================================

st.bar_chart(
    results.set_index("Cluster")
)


# =====================================================
# ANOMALIES
# =====================================================

st.header("⚠️ Détection des anomalies")


if number_anomalies > 0:

    st.warning(
        f"{number_anomalies} observation(s) ont été "
        "identifiées comme anomalies par DBSCAN."
    )

else:

    st.success(
        "✅ Aucune anomalie détectée."
    )


# =====================================================
# LABELS
# =====================================================

st.header("🔍 Labels attribués par DBSCAN")


labels_df = pd.DataFrame({

    "Observation":
        range(1, len(labels) + 1),

    "Cluster":
        labels

})


st.dataframe(
    labels_df,
    use_container_width=True,
    hide_index=True
)


# =====================================================
# INTERPRETATION
# =====================================================

st.header("ℹ️ Interprétation")

st.write(
    """
    Dans DBSCAN :

    • Les valeurs positives correspondent aux différents clusters.

    • La valeur -1 correspond aux observations considérées
      comme des anomalies ou des points aberrants.

    • Le paramètre eps contrôle le voisinage utilisé par DBSCAN.

    • Le paramètre min_samples définit le nombre minimum
      de points nécessaires pour former une région dense.
    """
)
