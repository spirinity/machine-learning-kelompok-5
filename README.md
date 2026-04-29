# Segmentasi Pelanggan Digital Marketing Menggunakan K-Means Clustering

README ini merangkum isi notebook 3_Preliminary_Result_Kelompok5.ipynb yang berisi hasil awal (preliminary results) segmentasi pelanggan berbasis K-Means, dengan validasi Decision Tree, merujuk pada paper Wang (2025).

## Ringkasan Notebook

Notebook ini mengimplementasikan pipeline berikut:

1. Import library dan konfigurasi lingkungan
2. Memuat dataset UCI Online Retail
3. Feature engineering 11 fitur RFM-like (Var1-Var11)
4. Z-score normalization dan analisis korelasi
5. PCA (6 komponen, >= 90% variance explained)
6. K-Means baseline (K=6) dengan elbow + silhouette
7. Validasi segmentasi menggunakan Decision Tree (train/test split + CV)
8. Tabel komparasi K-Means vs Decision Tree

## Referensi Paper

Wang, G. (2025). Customer segmentation in the digital marketing using a Q-learning based differential evolution algorithm integrated with K-means clustering. PLOS ONE, 20(2): e0318519. https://doi.org/10.1371/journal.pone.0318519

## Kebutuhan

- Python 3.x
- Library utama: pandas, numpy, matplotlib, seaborn, scikit-learn, ucimlrepo

Contoh instalasi (di notebook):

```
pip install ucimlrepo
```

## Cara Menjalankan

1. Buka notebook 3_Preliminary_Result_Kelompok5.ipynb.
2. Jalankan sel secara berurutan dari atas ke bawah.
3. Pastikan koneksi internet aktif untuk mengunduh dataset UCI.

## Output Utama

Notebook menghasilkan beberapa visualisasi dan metrik evaluasi, termasuk:

- Elbow curve dan silhouette score
- Distribusi kluster (pie/bar)
- Proyeksi PCA dan scatter plot kluster
- Profil kluster (radar chart)
- Evaluasi Decision Tree (accuracy, confusion matrix, feature importance)

## 👥 Anggota Kelompok

| No  | Nama                        | NIM      |
| --- | --------------------------- | -------- |
| 1   | Aisyah Wilda Fauziah Amanda | 11231005 |
| 2   | Galuh Juliviana Romanita    | 11231027 |
| 3   | Mahardika Arka              | 11231037 |
| 4   | Muhammad Shadiq Al-Fatiy    | 11231065 |
| 5   | Olivia Dafina               | 11231077 |
