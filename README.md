#  Sentiment Analysis Tiktok Reviews - NLP Project

Proyek ini bertujuan untuk melakukan analisis sentimen terhadap review/ulasan aplikasi TikTok yang diambil dari Google Play Store menggunakan teknik Natural Language Processing (NLP).

---

##  Struktur Repository
Repository ini disusun dengan struktur sebagai berikut untuk memudahkan navigasi:

*   **`App_deployment/`**: Folder ini berisi seluruh file yang digunakan untuk menjalankan aplikasi di Hugging Face, termasuk `app.py`, `requirements.txt`, serta salinan model dan vectorizer.
*   **`Dataset/`**: Berisi file data ulasan TikTok yang digunakan dalam proyek ini.
*   **`ModelFile/`**: Berisi file hasil export model machine learning dalam proyek ini.
*   **`UTS_NLP_Kelompok_Ulala.ipynb`**: Notebook utama yang berisi seluruh proses mulai dari data collection (scraping), text preprocessing, feature extraction, hingga modeling dan perbandingan model serta deploy sederhan di google collab menggunakan gradio

---

## Teknologi & Model
Dalam pengerjaan proyek ini, kami menggunakan kombinasi teknologi berikut:

*   **Bahasa Pemrograman**: Python.
*   **Preprocessing**: Data Cleaning, Tokenization, Stopword removal, dan Stemming menggunakan library Sastrawi.
*   **Feature Extraction**: BOW, TF-IDF N-Gram, dan Fast Text
*   **Machine Learning Model**: Naive Bayes, Logistic Regression, dan SVM.
*   **Deployment Interface**: Huggingface Gradio.

---

## Pengembang (Kelompok Ulala)
Proyek ini dikembangkan oleh:
1.  **Anisa Fitriyani - 0110224145**
2.  **Azkia Amanda - 0110224099**
3.  **Muhammad Rizky Nur Awalin - 0110224070**
4.  **Muhamad Solihin - 0110224098**
5.  **Muhammad Zain Rizqullah - 0110221017**

---

## Live Demo
Aplikasi hasil klasifikasi sentimen ini telah dideploy dan dapat diakses secara publik melalui Hugging Face Spaces:
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kelulala/UTS_Ulala)
> **[Klik di sini untuk mencoba Aplikasi Klasifikasi Sentimen](https://huggingface.co/spaces/kelulala/UTS_Ulala)**
<img width="1894" height="625" alt="image" src="https://github.com/user-attachments/assets/dd2bc92f-7aba-4caa-bb94-f7bda22a81e3" />

---
*Proyek ini dibuat untuk memenuhi tugas UTS mata kuliah NLP (Natural Language Processing) - © 2026*
