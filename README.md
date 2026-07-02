# Segmentasi Pola Transaksi Pelanggan Toko Online Menggunakan K-Means Clustering

Repository ini berisi proyek UAS Data Analytics / Data Mining berbasis CRISP-DM untuk melakukan clustering pada dataset transaksi pelanggan toko online.

## Anggota Kelompok
1. Alexander Noventino Lambut - 2401010802
2. I Gede Agus Arta Pratama Putra - 2401010252
3. Rivaldi Kinaryoadi - 2401010246

## Metode
Metode utama yang digunakan adalah **K-Means Clustering** dengan jumlah cluster akhir **K = 4**. Proyek ini bersifat unsupervised, sehingga tidak menggunakan target, pelabelan, atau pembagian data latih/uji.

## Struktur Repository

```text
laporan/      : laporan final BAB I sampai BAB VI
notebooks/    : notebook bukti proses BAB II sampai BAB VI
data/         : dataset mentah
outputs/      : file hasil preprocessing, modeling, dan evaluation
figures/      : gambar visualisasi untuk laporan
deployment/   : aplikasi Streamlit, model, preprocessor, dan file pendukung
docs/         : panduan demo dan tempat screenshot
```

## Cara Menjalankan Notebook di Google Colab
1. Buka Google Colab.
2. Upload file `notebooks/Notebook_Bukti_Final_UAS_Clustering_BAB2_BAB6.ipynb`.
3. Upload dataset `data/customer_spending_1M_2018_2025.csv` ke folder kerja Colab.
4. Jalankan semua cell dari atas ke bawah.
5. Notebook akan menghasilkan file preprocessing, model, evaluasi, visualisasi, dan deployment.

## Cara Menjalankan Deployment Streamlit
Masuk ke folder `deployment`, lalu jalankan:

```bash
pip install -r requirements_deployment.txt
streamlit run app_streamlit_clustering.py
```

Pastikan file berikut berada dalam folder yang sama dengan aplikasi Streamlit:
- `app_streamlit_clustering.py`
- `model_kmeans_clustering_k4.pkl`
- `preprocessor_clustering.pkl`
- `bab5_cluster_interpretation_summary.csv`
- `bab5_numeric_profile_summary.csv`

## Output Utama
- `data_clustering_prepared.csv` : data hasil preprocessing untuk modeling.
- `data_clustering_profile.csv` : data bersih untuk profiling cluster.
- `model_kmeans_clustering_k4.pkl` : model K-Means final.
- `preprocessor_clustering.pkl` : preprocessor untuk input baru pada deployment.
- `data_clustering_profile_with_cluster.csv` : data profiling yang sudah memiliki kolom cluster.
- `bab5_cluster_interpretation_summary.csv` : interpretasi dan rekomendasi awal tiap cluster.

## Link Repository GitHub
Isi link repository setelah upload:

```text
https://github.com/username/uas-data-mining-clustering-kmeans
```
