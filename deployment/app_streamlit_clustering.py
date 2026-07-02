import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Deployment K-Means Clustering", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model_kmeans_clustering_k4.pkl")
    preprocessor = joblib.load("preprocessor_clustering.pkl")
    interpretation = pd.read_csv("bab5_cluster_interpretation_summary.csv")
    numeric_profile = pd.read_csv("bab5_numeric_profile_summary.csv")
    return model, preprocessor, interpretation, numeric_profile

model, preprocessor, interpretation, numeric_profile = load_artifacts()

st.title("Deployment Segmentasi Pola Transaksi Pelanggan")
st.write(
    "Aplikasi sederhana ini menggunakan model K-Means K=4 untuk menentukan cluster transaksi "
    "berdasarkan karakteristik pelanggan dan nilai pengeluaran."
)

st.subheader("Input Data Transaksi Baru")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=15, max_value=78, value=35)
    gender = st.selectbox("Gender", ["Female", "Male", "Unknown"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])

with col2:
    employees_status = st.selectbox(
        "Employees Status",
        ["Employees", "Self-Employed", "Unemployment", "Workers", "Unknown"]
    )
    payment_method = st.selectbox("Payment Method", ["PayPal", "Card", "Other"])
    referral = st.selectbox("Referral", [0.0, 1.0], format_func=lambda x: "Ya" if x == 1.0 else "Tidak")

with col3:
    amount_spent = st.number_input("Amount Spent", min_value=0.0, max_value=5000.0, value=1000.0, step=10.0)

input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Marital_status": marital_status,
    "Employees_status": employees_status,
    "Payment_method": payment_method,
    "Referral": referral,
    "Amount_spent": amount_spent
}])

if st.button("Prediksi Cluster"):
    input_prepared_array = preprocessor.transform(input_data)
    input_prepared = pd.DataFrame(input_prepared_array, columns=preprocessor.get_feature_names_out())
    cluster = int(model.predict(input_prepared)[0])

    st.success(f"Transaksi masuk ke Cluster {cluster}")

    hasil = interpretation[interpretation["Cluster"] == cluster]
    profil = numeric_profile[numeric_profile["Cluster"] == cluster]

    if not hasil.empty:
        st.subheader("Interpretasi Cluster")
        st.write("**Nama Profil:**", hasil.iloc[0]["Nama Profil"])
        st.write("**Karakteristik Utama:**", hasil.iloc[0]["Karakteristik Utama"])
        st.write("**Rekomendasi Awal:**", hasil.iloc[0]["Rekomendasi Awal"])

    if not profil.empty:
        st.subheader("Profil Numerik Cluster")
        st.dataframe(profil, use_container_width=True)

st.subheader("Ringkasan Profil Semua Cluster")
st.dataframe(interpretation, use_container_width=True)
