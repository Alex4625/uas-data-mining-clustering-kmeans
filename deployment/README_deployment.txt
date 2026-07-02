Panduan menjalankan deployment Streamlit:

1. Pastikan file berikut berada dalam folder yang sama:
   - app_streamlit_clustering.py
   - model_kmeans_clustering_k4.pkl
   - preprocessor_clustering.pkl
   - bab5_cluster_interpretation_summary.csv
   - bab5_numeric_profile_summary.csv
   - requirements_deployment.txt

2. Install library:
   pip install -r requirements_deployment.txt

3. Jalankan aplikasi:
   streamlit run app_streamlit_clustering.py

Aplikasi akan menerima input transaksi baru, memproses input dengan preprocessing yang sama seperti BAB III, lalu memprediksi cluster menggunakan model K-Means K=4 dari BAB IV.
