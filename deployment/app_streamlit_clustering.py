from pathlib import Path

import pandas as pd
import streamlit as st
import joblib


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Deployment K-Means Clustering",
    layout="wide"
)


# =========================================================
# PATH FILE DEPLOYMENT
# =========================================================
# BASE_DIR = folder tempat file app_streamlit_clustering.py berada.
# Ini penting agar aplikasi aman saat dijalankan di lokal maupun Streamlit Cloud.
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model_kmeans_clustering_k4.pkl"
PREPROCESSOR_PATH = BASE_DIR / "preprocessor_clustering.pkl"
INTERPRETATION_PATH = BASE_DIR / "bab5_cluster_interpretation_summary.csv"
NUMERIC_PROFILE_PATH = BASE_DIR / "bab5_numeric_profile_summary.csv"


# =========================================================
# FUNGSI LOAD FILE
# =========================================================
@st.cache_resource
def load_artifacts():
    required_files = {
        "Model K-Means": MODEL_PATH,
        "Preprocessor": PREPROCESSOR_PATH,
        "Interpretasi Cluster": INTERPRETATION_PATH,
        "Profil Numerik Cluster": NUMERIC_PROFILE_PATH,
    }

    missing_files = []
    for nama, path in required_files.items():
        if not path.exists():
            missing_files.append(f"{nama}: {path}")

    if missing_files:
        st.error("Beberapa file deployment tidak ditemukan.")
        st.write("Pastikan file berikut berada di folder yang sama dengan `app_streamlit_clustering.py`:")
        for item in missing_files:
            st.code(item)
        st.stop()

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    interpretation = pd.read_csv(INTERPRETATION_PATH)
    numeric_profile = pd.read_csv(NUMERIC_PROFILE_PATH)

    # Pastikan kolom Cluster bertipe integer agar filter cluster aman
    if "Cluster" in interpretation.columns:
        interpretation["Cluster"] = interpretation["Cluster"].astype(int)

    if "Cluster" in numeric_profile.columns:
        numeric_profile["Cluster"] = numeric_profile["Cluster"].astype(int)

    return model, preprocessor, interpretation, numeric_profile


model, preprocessor, interpretation, numeric_profile = load_artifacts()


# =========================================================
# TAMPILAN UTAMA
# =========================================================
st.title("Deployment Segmentasi Pola Transaksi Pelanggan")

st.write(
    "Aplikasi sederhana ini menggunakan model K-Means K=4 untuk menentukan cluster transaksi "
    "berdasarkan karakteristik pelanggan dan nilai pengeluaran."
)

st.info(
    "Alur aplikasi: input transaksi baru → preprocessing → model K-Means → hasil cluster → interpretasi dan rekomendasi."
)


# =========================================================
# INPUT DATA BARU
# =========================================================
st.subheader("Input Data Transaksi Baru")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=15,
        max_value=78,
        value=35
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Unknown"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single"]
    )

with col2:
    employees_status = st.selectbox(
        "Employees Status",
        ["Employees", "Self-Employed", "Unemployment", "Workers", "Unknown"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        ["PayPal", "Card", "Other"]
    )

    referral = st.selectbox(
        "Referral",
        [0.0, 1.0],
        format_func=lambda x: "Ya" if x == 1.0 else "Tidak"
    )

with col3:
    amount_spent = st.number_input(
        "Amount Spent",
        min_value=0.0,
        max_value=5000.0,
        value=1000.0,
        step=10.0
    )


# =========================================================
# MEMBENTUK DATAFRAME INPUT
# =========================================================
input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Marital_status": marital_status,
    "Employees_status": employees_status,
    "Payment_method": payment_method,
    "Referral": referral,
    "Amount_spent": amount_spent
}])


with st.expander("Lihat data input mentah"):
    st.dataframe(input_data, use_container_width=True)


# =========================================================
# PREDIKSI CLUSTER
# =========================================================
if st.button("Prediksi Cluster"):
    try:
        # Input mentah diproses dulu oleh preprocessor
        input_prepared_array = preprocessor.transform(input_data)

        # Jika hasil transform berbentuk sparse matrix, ubah ke array biasa
        if hasattr(input_prepared_array, "toarray"):
            input_prepared_array = input_prepared_array.toarray()

        # Ambil nama fitur hasil preprocessing
        feature_names = preprocessor.get_feature_names_out()

        input_prepared = pd.DataFrame(
            input_prepared_array,
            columns=feature_names
        )

        # Model K-Means menentukan cluster
        cluster = int(model.predict(input_prepared)[0])

        st.success(f"Transaksi masuk ke Cluster {cluster}")

        # Ambil interpretasi cluster
        hasil = interpretation[interpretation["Cluster"] == cluster]
        profil = numeric_profile[numeric_profile["Cluster"] == cluster]

        # =================================================
        # TAMPILKAN INTERPRETASI
        # =================================================
        if not hasil.empty:
            st.subheader("Interpretasi Cluster")

            nama_profil = hasil.iloc[0].get("Nama Profil", "-")
            karakteristik = hasil.iloc[0].get("Karakteristik Utama", "-")
            rekomendasi = hasil.iloc[0].get("Rekomendasi Awal", "-")

            st.write("**Nama Profil:**", nama_profil)
            st.write("**Karakteristik Utama:**", karakteristik)
            st.write("**Rekomendasi Awal:**", rekomendasi)
        else:
            st.warning("Interpretasi untuk cluster ini tidak ditemukan pada file interpretasi.")

        # =================================================
        # TAMPILKAN PROFIL NUMERIK
        # =================================================
        if not profil.empty:
            st.subheader("Profil Numerik Cluster")
            st.dataframe(profil, use_container_width=True)
        else:
            st.warning("Profil numerik untuk cluster ini tidak ditemukan.")

        # =================================================
        # OPSIONAL: LIHAT HASIL PREPROCESSING
        # =================================================
        with st.expander("Lihat hasil preprocessing input"):
            st.write(
                "Data di bawah ini adalah input setelah diubah oleh preprocessor. "
                "Format inilah yang dibaca oleh model K-Means."
            )
            st.dataframe(input_prepared, use_container_width=True)

    except Exception as e:
        st.error("Terjadi error saat melakukan prediksi cluster.")
        st.write("Detail error:")
        st.exception(e)


# =========================================================
# RINGKASAN SEMUA CLUSTER
# =========================================================
st.subheader("Ringkasan Profil Semua Cluster")
st.dataframe(interpretation, use_container_width=True)

st.subheader("Profil Numerik Semua Cluster")
st.dataframe(numeric_profile, use_container_width=True)


# =========================================================
# CATATAN PENJELASAN
# =========================================================
with st.expander("Penjelasan Alur Deployment"):
    st.write(
        """
        1. User memasukkan data transaksi baru.
        2. Input masih berbentuk data mentah, seperti Gender, Payment Method, dan Amount Spent.
        3. Preprocessor mengubah input tersebut menjadi format numerik yang sama seperti data training.
        4. Model K-Means menentukan cluster berdasarkan input yang sudah diproses.
        5. Nomor cluster dicocokkan dengan tabel interpretasi.
        6. Aplikasi menampilkan nama profil, karakteristik utama, dan rekomendasi bisnis.
        """
    )