# 🛍️ Segmentasi Pelanggan E-Commerce Menggunakan K-Means Clustering & Decision Tree

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas)
![Dataset](https://img.shields.io/badge/Dataset-Zenodo%2014614253-brightgreen?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab)

**Proyek Machine Learning — Kelompok 5**

Reimplementasi paper *"Customer segmentation in the digital marketing using a Q-learning based differential evolution algorithm integrated with K-means clustering"* (PLoS ONE, 2025)

</div>

---

## 👥 Anggota Kelompok

| No | Nama | NIM |
|:--:|------|:---:|
| 1 | Aisyah Wilda Fauziah Amanda | 11231005 |
| 2 | Galuh Juliviana Romanita | 11231027 |
| 3 | Mahardika Arka | 11231037 |
| 4 | Muhammad Shadiq Al-Fatiy | 11231065 |
| 5 | Olivia Dafina | 11231077 |

---

## 📋 Deskripsi Proyek

Proyek ini merupakan reimplementasi dari paper:

> **"Customer segmentation in the digital marketing using a Q-learning based differential evolution algorithm integrated with K-means clustering"**
> Guanqun Wang, *PLoS ONE 20(2): e0318519*, February 2025
> 🔗 [DOI: 10.1371/journal.pone.0318519](https://doi.org/10.1371/journal.pone.0318519)

### Ide Utama

Pipeline segmentasi pelanggan ini bekerja dalam dua tahap utama:

```
TAHAP 1 — SEGMENTASI (Unsupervised):
  Data transaksi mentah
    → Feature Engineering RFM (11 fitur: Var1–Var11)
    → Z-Score Normalization
    → PCA: 11 fitur → 6 komponen utama (≥90% variance)
    → K-Means Clustering (K=6, ditentukan via Elbow Method)
  Output: 6 segmen pelanggan dengan profil bisnis berbeda

TAHAP 2 — VALIDASI (Supervised):
  Label kluster K-Means → input Decision Tree
    → Train/Test Split (80:20, stratified)
    → 5-Fold Cross Validation (hyperparameter tuning)
    → Evaluasi akurasi test set
  Output: Akurasi >95% → bukti segmentasi K-Means valid & konsisten
```

> ✅ Segmentasi berbasis perilaku transaksi nyata (RFM)
> ✅ Reduksi dimensi dengan PCA mempertahankan 92,43% informasi
> ✅ Validasi silang dengan model supervised (Decision Tree)
> ✅ Akurasi validasi mencapai **98,73%** pada test set

---

## 🗂️ Dataset

| Properti | Detail |
|----------|--------|
| **Nama** | Customer Segmentation (UCI Online Retail) |
| **Sumber** | [Zenodo — DOI: 10.5281/zenodo.14614252](https://zenodo.org/records/14614253) |
| **Jumlah Transaksi** | 541.909 baris |
| **Fitur Asli** | 8 (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country) |
| **Fitur Hasil Engineering** | 11 fitur RFM-based (Var1–Var11) |
| **Pelanggan Unik** | ~4.300 (setelah preprocessing) |
| **Format** | `.csv` |

### Deskripsi 11 Fitur (Var1–Var11)

| No | Fitur | Deskripsi |
|----|-------|-----------|
| 1 | Var1 | Jumlah hari sejak transaksi terakhir (*Recency*) |
| 2 | Var2 | Total jumlah transaksi (*Frequency*) |
| 3 | Var3 | Total jumlah produk yang dibeli |
| 4 | Var4 | Total pengeluaran (*Monetary*) |
| 5 | Var5 | Rata-rata biaya per transaksi |
| 6 | Var6 | Jumlah tipe produk yang dibeli |
| 7 | Var7 | Rata-rata selang hari antar pembelian |
| 8 | Var8 | Estimasi hari pembelian berikutnya |
| 9 | Var9 | Asal negara UK (binary: 1=UK, 0=non-UK) |
| 10 | Var10 | Frekuensi pembatalan transaksi |
| 11 | Var11 | Rata-rata pengeluaran per bulan |

---

## 🧠 Arsitektur Pipeline

### Tahap 1 — K-Means Clustering

```
┌─────────────────────────────────────────────────────┐
│  INPUT: 541.909 baris transaksi mentah              │
│                                                     │
│  1. Data Cleaning                                   │
│     - Hapus Quantity ≤ 0, UnitPrice ≤ 0            │
│     - Hapus baris tanpa CustomerID                  │
│     - Isi NaN Var7 & Var10 dengan 0                 │
│     - Clipping outlier (persentil ke-99)            │
│                                                     │
│  2. Feature Engineering (RFM-based)                 │
│     - Agregasi per CustomerID → 11 fitur (Var1–11) │
│     - Binary Encoding: Country → Var9 (UK flag)    │
│                                                     │
│  3. Normalisasi: Z-Score (StandardScaler)           │
│     - Mean ≈ 0, Std ≈ 1 per fitur                  │
│                                                     │
│  4. PCA: 11 dimensi → 6 komponen utama             │
│     - Cumulative variance ≥ 92,43%                  │
│                                                     │
│  5. K-Means Clustering (K=6)                        │
│     - Elbow Method → K=6 optimal                   │
│     - Init: k-means++, n_init=20                    │
│                                                     │
│  OUTPUT: 6 label segmen pelanggan (C1–C6)          │
└─────────────────────────────────────────────────────┘
```

### Tahap 2 — Decision Tree (Validasi)

```
┌─────────────────────────────────────────────────────┐
│  INPUT: 6 komponen PCA + label K-Means              │
│                                                     │
│  1. Train/Test Split: 80% train, 20% test           │
│     - Stratified → distribusi kelas proporsional   │
│                                                     │
│  2. Hyperparameter Tuning                           │
│     - Stratified 5-Fold CV                         │
│     - Grid search max_depth: 2–20                  │
│     - max_depth optimal = 13                        │
│                                                     │
│  3. Evaluasi Test Set                               │
│     - Accuracy, Precision, Recall, F1-Score        │
│     - Confusion Matrix                             │
│                                                     │
│  OUTPUT: Akurasi 98,73% → segmentasi valid ✅       │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Hasil & Evaluasi

### K-Means Clustering (K=6)

| Metrik Evaluasi | Nilai | Keterangan |
|-----------------|-------|------------|
| Jumlah Cluster (K) | **6** | Ditentukan via Elbow Method |
| SSE / Inertia | **14.878,27** | Total jarak kuadrat ke centroid |
| Silhouette Score | **0,4340** | Skala 0–1, lebih tinggi = lebih baik |
| Davies-Bouldin Index | **0,7325** | Lebih rendah = lebih baik |
| Cluster terbesar | **C1 (57,7%)** | Segmen pelanggan dominan |

### Profil 6 Segmen Pelanggan

| Kluster | Proporsi | Profil Bisnis | Strategi Marketing |
|---------|----------|---------------|--------------------|
| C1 | ~57,7% | Price-sensitive, volume tinggi | Promosi diskon & bundling |
| C2 | ~9,7% | Premium, pengeluaran tinggi | Program loyalitas & VIP |
| C3 | ~7,2% | High-expectation, cancel rate tinggi | Perjelas info produk |
| C4 | ~6,6% | UK-loyal customers | Personalisasi regional |
| C5 | ~21,3% | Cautious, weekend buyers | Flash sale akhir pekan |
| C6 | ~7,3% | Balanced / moderate | Diversifikasi kampanye |

### Decision Tree — Perbandingan Baseline vs Optimasi

| Model | Tipe | Metrik | Nilai |
|-------|------|--------|-------|
| K-Means Baseline (K=6) | Clustering | SSE / Inertia | 14.878,27 |
| K-Means Baseline (K=6) | Clustering | Silhouette Score | 0,4340 |
| Decision Tree Baseline (max_depth=2) | Klasifikasi | Accuracy (test set) | 88,36% |
| **Decision Tree Optimasi (max_depth=13)** | Klasifikasi | **Accuracy (test set)** | **98,73%** |

| Algoritma | Precision | Recall | F1-Score | Accuracy |
|-----------|-----------|--------|----------|----------|
| DT Baseline (max_depth=2) | — | — | — | 88,36% |
| **DT Optimasi (max_depth=13)** | **0,99** | **0,99** | **0,99** | **98,73%** |

---

## 📓 Struktur Notebook

File: `3_Preliminary_Result_Kelompok5.ipynb`

| Bagian | Konten |
|--------|--------|
| **Bagian 1** | Setup Environment & Import Library |
| **Bagian 2** | Load Dataset & Eksplorasi Awal (EDA) |
| **Bagian 3** | Data Cleaning & Feature Engineering (RFM → Var1–Var11) |
| **Bagian 4A** | Normalisasi: Z-Score Scaling |
| **Bagian 4B** | Analisis Korelasi Antar Fitur (Heatmap) |
| **Bagian 5** | PCA: Reduksi Dimensi (11 → 6 komponen) |
| **Bagian 6** | K-Means Clustering — Elbow Method, Clustering K=6, Profil Kluster |
| **Bagian 7** | Decision Tree — CV Tuning, Evaluasi Test Set, Confusion Matrix |

---

## ⚙️ Teknologi & Library

| Library | Kegunaan |
|---------|----------|
| `pandas` | Manipulasi dan agregasi data transaksi |
| `numpy` | Operasi numerik & array |
| `matplotlib` / `seaborn` | Visualisasi (elbow curve, heatmap, radar chart, dll) |
| `scikit-learn` | PCA, KMeans, DecisionTree, CV, metrik evaluasi |

---

## 🚀 Cara Menjalankan

### Prasyarat
- Akun Google dengan Google Drive
- Google Colab (CPU sudah cukup, tidak perlu GPU)

### Langkah-langkah

1. **Buka notebook di Google Colab**
   - Klik tautan: [3_Preliminary_Result_Kelompok5.ipynb](https://colab.research.google.com/drive/1KL2ve56wxKnHb7S0Xq8h4sB4cXMNToaG)
   - Atau upload manual file `.ipynb` ke [colab.research.google.com](https://colab.research.google.com)

2. **Jalankan sel secara berurutan dari atas ke bawah**
   - Semua library sudah tersedia di Colab (tidak perlu install manual)
   - Dataset akan diunduh otomatis dari Zenodo di Bagian 2

3. **Estimasi Waktu Eksekusi**

   | Tahap | Estimasi |
   |-------|----------|
   | Load & EDA dataset | ~1–2 menit |
   | Data Cleaning & Feature Engineering | ~2–3 menit |
   | PCA & Normalisasi | < 1 menit |
   | K-Means (Elbow + Final) | ~2–3 menit |
   | Decision Tree (CV Tuning) | ~3–5 menit |
   | **Total** | **~10–15 menit** |

---

## 📁 Struktur Repositori

```
machine-learning-kelompok-5/
├── 📓 3_Preliminary_Result_Kelompok5.ipynb   # Notebook utama (preliminary)
├── 📄 1_Proposal_Kelompok5.pdf               # Proposal proyek
├── 📄 2_Paper_Referensi_Kelompok5.pdf        # Paper referensi utama (Wang, 2025)
└── 📝 README.md                              # Dokumentasi ini
```

---

## 📚 Referensi

1. **Paper Utama:**
   Wang, G. (2025). *Customer segmentation in the digital marketing using a Q-learning based differential evolution algorithm integrated with K-means clustering.* PLoS ONE 20(2): e0318519.
   🔗 https://doi.org/10.1371/journal.pone.0318519

2. **Dataset:**
   UCI Online Retail Dataset via Zenodo.
   🔗 https://zenodo.org/records/14614253

3. **Metode PCA:**
   Abdi, H., & Williams, L. J. (2010). Principal component analysis. *Wiley Interdisciplinary Reviews: Computational Statistics*, 2(4), 433–459.

4. **Metode Elbow & K-Means:**
   Syakur, M. A., et al. (2018). Integration K-Means Clustering Method and Elbow Method for Identification of The Best Customer Profile Cluster. *IOP Conference Series*, 336(1).

---

## 📝 Lisensi

Proyek ini dibuat untuk keperluan akademik — Tugas Proyek Machine Learning, Semester Genap 2024/2025.

---
