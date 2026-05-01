import gradio as gr
import pandas as pd
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download resource NLTK yang diperlukan
nltk.download('punkt')
nltk.download('punkt_tab')

# --- 1. LOAD MODEL DAN VECTORIZER ---
# Pastikan file .pkl ini sudah diunggah ke Hugging Face Space kamu
svm_model_tfidf = joblib.load('best_svm_tfidf_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer_ngram.pkl')

# --- 2. DATA PENDUKUNG ---

slang_dict = {
    # Umum & Singkatan
    "gk": "tidak", "ga": "tidak", "nggak": "tidak", "tdk": "tidak", "nda": "tidak", "gak": "tidak", "kmrin": "kemarin", "bnyk":"banyak",
    "gbs": "tidak bisa", "gj": "tidak jelas", "tp": "tapi", "tpi": "tapi", "yg": "yang", "aja": "saja", "ktnya": "katanya", "enggak":"tidak",
    "bgt": "banget", "bngt": "banget", "bgtt": "banget", "dr": "dari", "udh": "sudah", "sdh": "sudah", "gue": "saya", "gpp": "tidak apa apa",
    "udah": "sudah", "hrs": "harus", "pdhl": "padahal", "krn": "karena", "dgn": "dengan", "sy": "saya", "ad": "ada", "gaada": "tidak ada",
    "gw": "saya", "gua": "saya", "lo": "kamu", "klo": "kalau", "kl": "kalau", "sm": "sama", "ama": "sama", "knp": "kenapa", "tmbh":"tambah",
    "knpa": "kenapa", " g":"tidak", "bngett":"banget", "baguss":"bagus", "acc":"disetujui", "aku":"saya", "yng":"yang", "hbis":"habis",
    "lg": "lagi", "lgi": "lagi", "bs": "bisa", "blm": "belum", "msh": "masih", "trs": "terus", "masak": "masa", "ak":"saya", "tlh":"telah",
    "ajah":"saja", "aja":"saja", "trus":"terus", "bet":"banget", "vn":"pesan suara", "kyk":"seperti", "g jelas":"tidak jelas", "utk":"untuk",
    "trims":"terima kasih", "ni":"ini", "seruu":"seru", "yangg":"yang", "adaa":"ada", "abis":"habis", "agk":"agak", "bs":"bisa",
    "bgs":"bagus", "bljr":"belajar", "baguszz":"bagus", "ori":"orisinal", "punyakh":"punya saya", "gada":"tidak ada", "tiba²":"tiba tiba",
    "bajak":"banyak", " do":"di", "apacoba":"apa coba", "apknya":"aplikasi nya", "gajelas":"tidak jelas", "debest":"terbaik",


    # Kata Kasar & Emosional (Penting untuk label Negative)
    "anjg": "anjing", "anjir": "anjing", "ajg": "anjing", "asu": "anjing", "jelek": "buruk", "jir":"anjing",
    "jancokk": "sialan", "jancok": "sialan", "dancok":"sialan","tolol": "bodoh", "gblk": "bodoh", "goblog": "bodoh", "bgst":"buruk",

    # Spesifik Aplikasi & Teknis
    "apl": "aplikasi", "apk": "aplikasi", "aplk": "aplikasi", "donlod": "unduh", "updet": "perbarui", "account":"akun", "download":"unduh",
    "ngleg": "lambat", "ngeleg": "lambat", "lemot": "lambat", "lalot": "lambat", "loding": "pemuatan", "leg":"lambat", "upload":"unggah",
    "bug": "kesalahan", "eror": "error", "vyp": "fyp", "vt": "video", "load": "pemuatan", "ngelek":"lambat", "post":"unggah",
    "update":"perbarui", "lag":"lambat", "ngepost":"mengunggah", "ngedownloadnya":"mengunduhnya", "abdet":"perbarui", "lemott":"lambat",
    "diupdate":"diperbarui", "epek":"efek", "pencet":"klik",
}


# Daftar Stopword Lengkap dari GitHub yang kamu berikan
full_stopword_list = {
    'ada', 'adalah', 'adanya', 'adapun', 'admin', 'agak', 'agaknya', 'agar', 'aja', 'akan', 'akankah', 
    'akhirnya', 'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara', 'antaranya', 
    'apa', 'apaan', 'apabila', 'apakah', 'apalagi', 'apatah', 'aplikasinya', 'atau', 'ataukah', 'ataupun', 
    'bagai', 'bagaikan', 'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagi', 'bahkan', 'bahwa', 'bahwasanya', 
    'bang', 'banyak', 'beberapa', 'begini', 'beginian', 'beginikah', 'beginilah', 'begitu', 'begitukah', 
    'begitulah', 'begitupun', 'belumlah', 'berapa', 'berapakah', 'berapalah', 'berapapun', 'bermacam', 
    'bersama', 'betulkah', 'biasanya', 'bila', 'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah', 'bolehlah', 
    'buat', 'bukankah', 'bukanlah', 'bukannya', 'cuma', 'cuman', 'dahulu', 'dalam', 'dan', 'dapat', 'dari', 
    'daripada', 'deh', 'dekat', 'demi', 'demikian', 'demikianlah', 'dengan', 'depan', 'di', 'dia', 'dialah', 
    'diantara', 'diantaranya', 'dikarenakan', 'dini', 'diri', 'dirinya', 'disini', 'disinilah', 'dong', 
    'dulu', 'enggak', 'enggaknya', 'entah', 'entahlah', 'hai', 'hal', 'halo', 'hampir', 'hanya', 'hanyalah', 
    'harus', 'haruslah', 'harusnya', 'hendak', 'hendaklah', 'hendaknya', 'hingga', 'ia', 'ialah', 'ibarat', 
    'ingin', 'inginkah', 'inginkan', 'ini', 'inikah', 'inilah', 'itu', 'itukah', 'itulah', 'jangankan', 
    'janganlah', 'jika', 'jikalau', 'juga', 'justru', 'kak', 'kala', 'kalau', 'kalaulah', 'kalaupun', 
    'kali', 'kalian', 'kami', 'kamilah', 'kamu', 'kamulah', 'kan', 'kapan', 'kapankah', 'kapanpun', 
    'karena', 'karenanya', 'kasih', 'ke', 'kecil', 'kemudian', 'kenapa', 'kepada', 'kepadanya', 'ketika', 
    'khususnya', 'kini', 'kinilah', 'kiranya', 'kita', 'kitalah', 'kok', 'lagi', 'lagian', 'lah', 'lain', 
    'lainnya', 'lalu', 'lama', 'lamanya', 'lebih', 'macam', 'maka', 'makanya', 'makin', 'malah', 'malahan', 
    'mampu', 'mampukah', 'mana', 'manakala', 'manalagi', 'masih', 'masihkah', 'masing', 'mau', 'maupun', 
    'melainkan', 'melalui', 'memang', 'mengapa', 'mereka', 'merekalah', 'merupakan', 'meski', 'meskipun', 
    'mohon', 'mungkin', 'mungkinkah', 'nah', 'namun', 'nanti', 'nantinya', 'nih', 'nya', 'nyaris', 'oleh', 
    'olehnya', 'pada', 'padahal', 'padanya', 'paling', 'pantas', 'para', 'pasti', 'pastilah', 'per', 
    'percuma', 'pernah', 'pula', 'pun', 'rupanya', 'saat', 'saatnya', 'sajalah', 'saling', 'sama', 'sambil', 
    'sampai', 'sana', 'sangat', 'sangatlah', 'saya', 'sayalah', 'se', 'sebab', 'sebabnya', 'sebagai', 
    'sebagaimana', 'sebagainya', 'sebaliknya', 'sebanyak', 'sebegini', 'sebegitu', 'sebelum', 'sebelumnya', 
    'sebenarnya', 'seberapa', 'sebetulnya', 'sebisanya', 'sebuah', 'sedang', 'sedangkan', 'sedemikian', 
    'sedikit', 'sedikitnya', 'segala', 'segalanya', 'segera', 'seharusnya', 'sehingga', 'sejak', 'sejenak', 
    'sekali', 'sekalian', 'sekaligus', 'sekalipun', 'sekarang', 'seketika', 'sekiranya', 'sekitar', 
    'sekitarnya', 'sela', 'selagi', 'selain', 'selaku', 'selalu', 'selama', 'selamanya', 'seluruh', 
    'seluruhnya', 'semacam', 'semakin', 'semasih', 'semaunya', 'sementara', 'sempat', 'semua', 'semuanya', 
    'semula', 'sendiri', 'sendirinya', 'seolah', 'seorang', 'sepanjang', 'sepantasnya', 'sepantasnyalah', 
    'seperti', 'sepertinya', 'sering', 'seringnya', 'serta', 'serupa', 'sesaat', 'sesama', 'sesegera', 
    'sesekali', 'seseorang', 'sesuatu', 'sesuatunya', 'sesudah', 'sesudahnya', 'setelah', 'seterusnya', 
    'setiap', 'setidaknya', 'sewaktu', 'si', 'siapa', 'siapakah', 'siapapun', 'sini', 'sinilah', 'suatu', 
    'sudah', 'sudahkah', 'sudahlah', 'supaya', 'tadi', 'tadinya', 'tah', 'tak', 'tapi', 'telah', 'tentang', 
    'tentu', 'tentulah', 'tentunya', 'terdiri', 'terhadap', 'terhadapnya', 'terimakasih', 'terlalu', 
    'terlebih', 'tersebut', 'tersebutlah', 'tertentu', 'tetapi', 'tiap', 'tidakkah', 'toh', 'tolong', 
    'udah', 'waduh', 'wah', 'wahai', 'walau', 'walaupun', 'wong', 'yah', 'yaitu', 'yakni', 'yang', 'yg'
}

# KOMENTAR REVISI: Daftar kata negasi yang harus DIPERTAHANKAN
negation_words = {'tidak', 'kurang', 'belum', 'bukan', 'jangan', 'tidaklah', 'tanpa','lumayan', 'biasa', 'saja'}

# Buat set stopword final (Daftar Github dikurangi Kata Negasi)
final_stopword_set = {word for word in full_stopword_list if word not in negation_words}

# Inisialisasi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- 3. FUNGSI PREPROCESSING ---

def clean_text(text):
    # Logika fungsi clean_text milikmu (Lengkap)
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'@\w+|#\w+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'([aeiou])\1+', r'\1', text)
    text = re.sub(r'([b-df-hj-np-tv-z])\1{2,}', r'\1', text)
    
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(review_text):
    if not review_text.strip():
        return "Masukkan teks ulasan terlebih dahulu..."
    
    # 1. Cleaning & Slang Handling
    cleaned = clean_text(review_text)
    
    # 2. Tokenization
    tokens = word_tokenize(cleaned)
    
    # 3. Stopword Removal (Sesuai daftar Github, tapi NEGASI AMAN)
    tokens_filtered = [w for w in tokens if w not in final_stopword_set]
    
    # 4. Stemming
    # Proses ini menyatukan kata kembali lalu di-stem
    stemmed_text = stemmer.stem(' '.join(tokens_filtered))
    
    # 5. Transformasi ke TF-IDF
    text_vector = tfidf_vectorizer.transform([stemmed_text])
    
    # 6. Prediksi menggunakan SVM (Model Terbaik)
    prediction = svm_model_tfidf.predict(text_vector)[0]
    
    label_map = {
        'positive': "😊 POSITIF", 
        'negative': "😡 NEGATIF", 
        'neutral': "😐 NETRAL"
    }
    return label_map.get(prediction, prediction)

# --- 4. GRADIO INTERFACE ---
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, label="Input Review TikTok", placeholder="Ketik review di sini..."),
    outputs=gr.Label(label="Hasil Analisis"),
    title="🚀 Analisis Review Aplikasi (Play Store) - TikTok Sentiment Analysis - Kelompok Ulala",
    description="Analisis sentimen ulasan pengguna TikTok menggunakan SVM + TF-IDF N-gram (Akurasi: 74.60%)",
    examples=[
        ["Aplikasi bagus sekali, sangat menghibur!"],
        ["Jelek banget, aplikasinya tidak berguna dan lambat."],
        ["Biasa saja, belum ada fitur baru yang wah."],          
        ["Lumayan lah cukup membantu sehari-hari."]
    ],
    theme="Monochrome"
)

if __name__ == "__main__":
    demo.launch()
