import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances


# ==========================================
# CONFIGURATION DE L'APPLICATION
# ==========================================

st.set_page_config(
    page_title="Wheat Seeds - DBSCAN",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Segmentation des graines de blé")

st.write(
    "Application de segmentation des graines de blé "
    "basée sur l'algorithme DBSCAN."
)

st.info(
    "DBSCAN ne possède pas de méthode predict() native. "
    "L'affectation d'une nouvelle graine est donc réalisée "
    "approximativement à partir du point central DBSCAN le plus proche."
)


# ==========================================
# CHARGEMENT DU MODELE
# ==========================================

@st.cache_resource
def load_model():

    return joblib.load("dbscan_wheat.pkl")


try:

    dbscan_model = load_model()

    st.success(
        "✅ Modèle DBSCAN chargé avec succès !"
    )

except Exception as e:

    st.error(
        f"❌ Erreur lors du chargement du modèle : {e}"
    )

    st.stop()


# ==========================================
# INFORMATIONS SUR LE MODELE
# ==========================================

eps = dbscan_model.eps
min_samples = dbscan_model.min_samples

labels = dbscan_model.labels_


nombre_clusters = (
    len(set(labels))
    - (1 if -1 in labels else 0)
)

nombre_anomalies = list(labels).count(-1)


st.sidebar.header("⚙️ Paramètres du modèle")

st.sidebar.write(
    f"**Epsilon (eps) :** {eps}"
)

st.sidebar.write(
    f"**Min samples :** {min_samples}"
)

st.sidebar.write(
    f"**Nombre de clusters :** {nombre_clusters}"
)

st.sidebar.write(
    f"**Nombre d'anomalies :** {nombre_anomalies}"
)


# ==========================================
# NOMS DES VARIABLES
# ==========================================

features = [

    "area A",

    "perimeter",

    "compactness",

    "length of kernel",

    "width of kernel",

    "asymmetry coefficient",

    "length of kernel groove"

]


# ==========================================
# SAISIE DES DONNEES
# ==========================================

st.header("🌾 Informations de la nouvelle graine")

st.write(
    "Saisissez les caractéristiques de la graine "
    "afin d'estimer son groupe de segmentation."
)


with st.form("wheat_form"):

    col1, col2 = st.columns(2)


    # ==========================================
    # COLONNE 1
    # ==========================================

    with col1:

        area = st.number_input(

            "Area A",

            min_value=0.0,

            value=15.26,

            step=0.01,

            format="%.4f"

        )


        perimeter = st.number_input(

            "Perimeter",

            min_value=0.0,

            value=14.84,

            step=0.01,

            format="%.4f"

        )


        compactness = st.number_input(

            "Compactness",

            min_value=0.0,

            value=0.8710,

            step=0.001,

            format="%.4f"

        )


        length_kernel = st.number_input(

            "Length of kernel",

            min_value=0.0,

            value=5.763,

            step=0.001,

            format="%.4f"

        )


    # ==========================================
    # COLONNE 2
    # ==========================================

    with col2:

        width_kernel = st.number_input(

            "Width of kernel",

            min_value=0.0,

            value=3.312,

            step=0.001,

            format="%.4f"

        )


        asymmetry = st.number_input(

            "Asymmetry coefficient",

            min_value=0.0,

            value=2.221,

            step=0.001,

            format="%.4f"

        )


        kernel_groove = st.number_input(

            "Length of kernel groove",

            min_value=0.0,

            value=5.220,

            step=0.001,

            format="%.4f"

        )


    submitted = st.form_submit_button(

        "🔍 Identifier le cluster",

        use_container_width=True

    )


# ==========================================
# AFFECTATION APPROXIMATIVE AVEC DBSCAN
# ==========================================

if submitted:


    # ==========================================
    # CREATION DU VECTEUR
    # ==========================================

    seed_data = np.array([[

        area,

        perimeter,

        compactness,

        length_kernel,

        width_kernel,

        asymmetry,

        kernel_groove

    ]])


    # ==========================================
    # VERIFICATION
    # ==========================================

    if seed_data.shape[1] != len(features):

        st.error(
            "❌ Le nombre de variables est incorrect."
        )

        st.stop()


    # ==========================================
    # VERIFICATION DES POINTS CENTRAUX
    # ==========================================

    if not hasattr(dbscan_model, "components_"):

        st.error(

            "❌ Le modèle ne contient pas les points "
            "centraux nécessaires à l'affectation."

        )

        st.stop()


    core_samples = dbscan_model.components_


    core_labels = dbscan_model.labels_[

        dbscan_model.core_sample_indices_

    ]


    # ==========================================
    # NORMALISATION
    # ==========================================

    seed_normalized = normalize(
        seed_data
    )


    # ==========================================
    # DISTANCES
    # ==========================================

    distances = pairwise_distances(

        seed_normalized,

        core_samples,

        metric="euclidean"

    )[0]


    # ==========================================
    # POINT CENTRAL LE PLUS PROCHE
    # ==========================================

    nearest_index = np.argmin(
        distances
    )


    nearest_distance = distances[
        nearest_index
    ]


    predicted_cluster = core_labels[
        nearest_index
    ]


    # ==========================================
    # RESULTAT
    # ==========================================

    st.header("📌 Résultat de la segmentation")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(

            "Cluster estimé",

            str(predicted_cluster)

        )


    with col2:

        st.metric(

            "Distance minimale",

            f"{nearest_distance:.4f}"

        )


    with col3:

        st.metric(

            "Seuil eps",

            f"{eps:.4f}"

        )


    # ==========================================
    # INTERPRETATION
    # ==========================================

    if nearest_distance <= eps:

        if predicted_cluster == -1:

            st.warning(

                "⚠️ La graine est associée à une "
                "zone considérée comme une anomalie "
                "par DBSCAN."

            )

        else:

            st.success(

                f"✅ La graine est affectée "
                f"approximativement au cluster "
                f"{predicted_cluster}."

            )

    else:

        st.warning(

            "⚠️ La graine est considérée comme "
            "un point isolé ou une anomalie : "
            "sa distance dépasse eps."

        )

        predicted_cluster = -1


    # ==========================================
    # AFFICHAGE DES DONNEES
    # ==========================================

    st.subheader("📋 Caractéristiques de la graine")


    seed_df = pd.DataFrame(

        seed_data,

        columns=features

    )


    st.dataframe(

        seed_df,

        use_container_width=True

    )


    # ==========================================
    # INFORMATIONS SUR L'AFFECTATION
    # ==========================================

    st.subheader(
        "📊 Informations sur l'affectation"
    )


    st.write(

        f"Distance au point central le plus proche : "
        f"**{nearest_distance:.4f}**"

    )


    st.write(

        f"Valeur du seuil eps : "
        f"**{eps:.4f}**"

    )


    if nearest_distance <= eps:

        st.write(

            "La distance est inférieure ou égale à eps. "
            "La graine se trouve dans le voisinage "
            "d'un point central DBSCAN."

        )

    else:

        st.write(

            "La distance est supérieure à eps. "
            "La graine est considérée comme une "
            "anomalie potentielle."

        )
