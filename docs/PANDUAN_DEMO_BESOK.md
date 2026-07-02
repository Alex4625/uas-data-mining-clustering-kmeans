# Panduan Demo Pelatihan dan Deployment

## Tujuan Demo
Demo menunjukkan proyek berjalan end-to-end dari data mentah, preprocessing, training model, evaluasi, sampai deployment aplikasi.

## Alur Demo Singkat

### 1. Tunjukkan Repository
Buka repository GitHub dan jelaskan struktur folder:
- laporan
- notebooks
- data
- outputs
- figures
- deployment

### 2. Demo Pelatihan Model di Colab
Buka notebook final:
`notebooks/Notebook_Bukti_Final_UAS_Clustering_BAB2_BAB6.ipynb`

Jelaskan urutan proses:
1. Dataset dibaca.
2. Data understanding dilakukan.
3. Data preparation dilakukan: missing value, konsistensi kategori, encoding, standardisasi.
4. K-Means dilatih.
5. Jumlah cluster diuji dengan Elbow, Silhouette, dan Davies-Bouldin.
6. Model akhir menggunakan K = 4.
7. Model dan preprocessor disimpan.

Output yang perlu ditunjukkan:
- jumlah data akhir setelah preprocessing
- grafik Elbow/Silhouette/Davies-Bouldin
- metrik model akhir
- file model `model_kmeans_clustering_k4.pkl`
- file preprocessor `preprocessor_clustering.pkl`

### 3. Demo Deployment Streamlit
Masuk ke folder deployment, lalu jalankan:

```bash
pip install -r requirements_deployment.txt
streamlit run app_streamlit_clustering.py
```

Masukkan contoh input:
- Age: 35
- Gender: Female
- Marital Status: Married
- Employees Status: Employees
- Payment Method: PayPal
- Referral: Ya
- Amount Spent: 1000

Tunjukkan hasil:
- cluster prediksi
- nama profil cluster
- karakteristik utama
- rekomendasi awal

### 4. Kalimat Penjelasan Deployment
Input baru tidak langsung masuk ke model. Input diproses dulu oleh `preprocessor_clustering.pkl` agar bentuknya sama seperti data training, lalu model `model_kmeans_clustering_k4.pkl` menentukan cluster. Setelah cluster diperoleh, aplikasi mengambil interpretasi dari file `bab5_cluster_interpretation_summary.csv`.

### 5. Pertanyaan yang Mungkin Ditanyakan Dosen

**Kenapa memakai K-Means?**
Karena dataset berukuran besar, fitur sudah dinumerikkan dan distandardisasi, serta K-Means efisien untuk segmentasi berbasis jarak.

**Kenapa K = 4?**
Karena hasil pengujian menunjukkan K=4 memberikan segmentasi yang lebih seimbang untuk interpretasi bisnis, dengan pertimbangan Elbow, Silhouette, dan Davies-Bouldin.

**Kenapa tidak ada train-test split?**
Karena tugas clustering bersifat unsupervised, tidak memiliki target/label, sehingga evaluasi menggunakan metrik internal seperti Silhouette, Davies-Bouldin, dan Inertia.

**Kenapa perlu preprocessor?**
Karena input baru masih berupa data mentah, sedangkan model dilatih menggunakan data yang sudah encoding dan standardisasi.

**Apa output bisnisnya?**
Outputnya adalah segmentasi transaksi pelanggan dan rekomendasi strategi pemasaran per cluster.
