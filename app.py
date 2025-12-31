import streamlit as st
import pickle
import cv2
import numpy as np
from streamlit_option_menu import option_menu
from predict_utils import build_feature
import os
import base64

# Konfigurasi halaman aplikasi
st.set_page_config(
    page_title="Deteksi Gambar AI vs Real",
    page_icon="🖼️",
    layout="wide"
)

# Mengatur lebar sidebar agar tidak terlalu besar
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    width: 300px !important;
}
</style>
""", unsafe_allow_html=True)

# Memuat model klasifikasi
@st.cache_resource(show_spinner=False)
def load_model():
    with open("random_forest_ai_vs_real_model.sav", "rb") as file:
        return pickle.load(file)

placeholder = st.empty()

with placeholder.container():
    st.markdown(
        """
        <div style="text-align:center; padding:40px;">
            <h4>🔄 Memuat Model Klasifikasi</h4>
            <p>Mohon tunggu, sistem sedang menyiapkan model!</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    model = load_model()

placeholder.empty()

# Sidebar navigasi menu
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Beranda", "Prediksi Gambar", "Analisis Model"],
        icons=["house", "image", "bar-chart"],
        menu_icon="cast",
        default_index=0
    )

# HALAMAN: BERANDA
if selected == "Beranda":
    import os
    import streamlit as st
    import base64

    # CSS
    st.markdown("""
    <style>
    .hero {
        padding: 35px;
        border-radius: 20px;
        background: linear-gradient(135deg, #0ea5e9, #22c55e);
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }

    .hero h1 {
        font-size: 40px;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.95;
    }

    .card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        height: 100%;
    }

    .metric-card {
        background: #f9fafb;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #e5e7eb;
    }

    .member-card {
        background: white;
        padding: 20px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }

    .member-card:hover {
        transform: translateY(-5px);
    }

    .footer {
        text-align: center;
        color: gray;
        font-size: 14px;
        margin-top: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🖼️ Deteksi Gambar AI vs Gambar Nyata</h1>
        <p>Sistem klasifikasi citra berbasis Machine Learning
        menggunakan algoritma <b>Random Forest</b></p>
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        st.markdown("""
        <div class="card">
            <h3>📌 Deskripsi Aplikasi</h3>
            <p> Aplikasi ini dirancang untuk membedakan gambar hasil kecerdasan buatan (AI) dan gambar nyata secara otomatis melalui analisis karakteristik visual pada citra digital. </p>
            <p> Dengan memanfaatkan pendekatan Machine Learning, sistem mengekstraksi fitur-fitur penting dari setiap citra dan melakukan proses klasifikasi secara objektif dan konsisten. </p>
            <p> Aplikasi ini dapat digunakan sebagai alat bantu analisis citra, khususnya dalam menghadapi potensi penyalahgunaan teknologi AI di ranah visual. </p>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("""
        <div class="card">
            <h3>🔄 Alur Machine Learning</h3>
            <ul>
                <li>📂 Dataset gambar AI dan gambar nyata</li>
                <li>🔧 Pra-pemrosesan resize dan normalisasi</li>
                <li>🎨 Ekstraksi fitur RGB Histogram, LBP, dan HOG</li>
                <li>📊 Pembagian data latih dan data uji</li>
                <li>🌳 Pelatihan dan pengujian model Random Forest</li>
                <li>📈 Evaluasi menggunakan Confusion Matrix</li>
                <li>🖼️ Sistem deteksi gambar AI dan gambar nyata</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    <div class="metric-card">
        <h2>📂</h2>
        <h3>Dataset</h3>
        <p>Gambar AI & Nyata</p>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
    <div class="metric-card">
        <h2>🧠</h2>
        <h3>Algoritma</h3>
        <p>Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown("""
    <div class="metric-card">
        <h2>🎯</h2>
        <h3>Ekstraksi Fitur</h3>
        <p>RGB Histogram, LBP, dan HOG</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("👥 Anggota Kelompok")

    def svg_avatar():
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg" width="130" height="130" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="12" fill="#e5e7eb"/>
            <circle cx="12" cy="9" r="4" fill="#9ca3af"/>
            <path d="M4 20c1.5-4 14.5-4 16 0" fill="#9ca3af"/>
        </svg>
        """
        b64 = base64.b64encode(svg.encode()).decode()
        return f"<img src='data:image/svg+xml;base64,{b64}' width='130'/>"

    def show_member(photo, name, nim):
        if os.path.exists(photo):
            img = f"<img src='{photo}' width='130' style='border-radius:50%;'>"
        else:
            img = svg_avatar()

        st.markdown(f"""
        <div class="member-card">
            {img}
            <h4>{name}</h4>
            <p>{nim}</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_member("assets/anggota1.jpg", "M Insani I U", "NIM 10122352")
    with col2:
        show_member("assets/anggota2.jpg", "Krisnover A", "NIM 10122353")
    with col3:
        show_member("assets/anggota3.jpg", "Atam Kartam", "NIM 10122367")
    with col4:
        show_member("assets/anggota4.jpg", "Andhika F M", "NIM 10122379")

    st.markdown("""
    <div class="footer">
        © 2025 | Proyek Sains Data <br>
        Sistem Deteksi Gambar AI vs Gambar Nyata
    </div>
    """, unsafe_allow_html=True)

# HALAMAN : PREDIKSI GAMBAR
elif selected == "Prediksi Gambar":

    import streamlit as st
    import numpy as np
    import cv2

    # CSS kustom tampilan
    st.markdown("""
    <style>
        .hero {
            padding: 35px;
            border-radius: 20px;
            background: linear-gradient(135deg, #0ea5e9, #22c55e);
            color: white;
            text-align: center;
            margin-bottom: 35px;
        }

        .hero h1 {
            font-size: 38px;
            margin-bottom: 10px;
        }

        .hero p {
            font-size: 17px;
            opacity: 0.95;
        }

        .card {
            background: #ffffff;
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.08);
            margin-top: 25px;
        }

        .metric-card {
            background: #f9fafb;
            padding: 22px;
            border-radius: 14px;
            text-align: center;
            border: 1px solid #e5e7eb;
        }

        .metric-card h2 {
            margin-bottom: 5px;
        }

        .footer {
            text-align: center;
            color: #6b7280;
            font-size: 14px;
            margin-top: 45px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Bagian hero
    st.markdown("""
    <div class="hero">
        <h1>🔍 Prediksi Gambar AI dan Gambar Nyata</h1>
        <p>
            Unggah citra digital untuk mendeteksi apakah gambar merupakan hasil kecerdasan buatan (AI) atau gambar nyata.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Unggah gambar
    uploaded = st.file_uploader(
        "Format yang didukung: JPG, JPEG, dan PNG. "
        "Pastikan gambar memiliki resolusi yang memadai.",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is None:
        st.info("Silakan unggah gambar terlebih dahulu untuk memulai proses klasifikasi.")

    # Proses jika gambar tersedia
    if uploaded is not None:
        try:
            # Membaca dan mengonversi gambar menjadi citra digital
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Pratinjau gambar
            col = st.columns([1, 2, 1])
            with col[1]:
                st.image(image, use_container_width=True)

            # Proses klasifikasi citra
            with st.spinner("🔎 Sistem sedang melakukan klasifikasi citra..."):
                X = build_feature(image)
                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                confidence = np.max(proba) * 100

            label = "Gambar Hasil AI" if pred == 1 else "Gambar Nyata"

            # Hasil klasifikasi
            result_html = f"""
            <div class="card">
                <h3>📊 Hasil Klasifikasi</h3>

                <div style="
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-top: 20px;
                ">
                    <div class="metric-card">
                        <h2>🏷️</h2>
                        <h4>Label Prediksi</h4>
                        <p><b>{label}</b></p>
                    </div>

                    <div class="metric-card">
                        <h2>🎯</h2>
                        <h4>Tingkat Akurasi</h4>
                        <p><b>{confidence:.2f}%</b></p>
                    </div>
                </div>

                <p style="margin-top:20px;">
                    Berdasarkan hasil analisis citra, sistem
                    mengklasifikasikan citra sebagai
                    <b>{label}</b> dengan tingkat akurasi
                    sebesar <b>{confidence:.2f}%</b>.
                </p>
            </div>
            """
            st.html(result_html)

            if confidence < 60:
                st.info(
                    "ℹ️ Tingkat kepercayaan model masih relatif rendah. "
                    "Disarankan menggunakan gambar dengan kualitas visual yang lebih baik."
                )

        except Exception:
            st.error("❌ Terjadi kesalahan saat memproses citra.")
            st.warning(
                "Pastikan file yang diunggah merupakan gambar valid "
                "dengan format JPG atau PNG."
            )

    # Footer
    st.markdown("""
    <div class="footer">
        © 2025 | Proyek Sains Data <br>
        Sistem Deteksi Gambar AI vs Gambar Nyata
    </div>
    """, unsafe_allow_html=True)

elif selected == "Analisis Model":

    # Import library
    import os
    from pathlib import Path
    from glob import glob

    import cv2
    import numpy as np
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns

    from skimage.feature import hog, local_binary_pattern
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix


    # Fungsi dataset lokal
    def get_dataset_paths():
        base_path = Path.cwd() / "dataset"
        ai_path = base_path / "AI Image"
        real_path = base_path / "Real Image"

        if not ai_path.exists() or not real_path.exists():
            raise FileNotFoundError(
                "Folder dataset tidak ditemukan!\n"
                f"AI Path   : {ai_path}\n"
                f"Real Path : {real_path}"
            )
        return ai_path, real_path


    def load_image_paths(folder_path):
        return sorted(glob(str(folder_path / "*")))


    ai_path, real_path = get_dataset_paths()
    ai_images = load_image_paths(ai_path)
    real_images = load_image_paths(real_path)


    # CSS global
    st.markdown("""
    <style>
        .hero {
            padding: 35px;
            border-radius: 20px;
            background: linear-gradient(135deg, #0ea5e9, #22c55e);
            color: white;
            text-align: center;
            margin-bottom: 35px;
        }
        .hero h1 {
            font-size: 38px;
            margin-bottom: 10px;
        }
        .hero p {
            font-size: 17px;
            opacity: 0.95;
        }
        .card {
            background: #ffffff;
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.08);
            margin-top: 25px;
        }
        .metric-card {
            background: #f9fafb;
            padding: 22px;
            border-radius: 14px;
            text-align: center;
            border: 1px solid #e5e7eb;
        }
        .footer {
            text-align: center;
            color: #6b7280;
            font-size: 14px;
            margin-top: 45px;
        }
    </style>
    """, unsafe_allow_html=True)


    # Hero section
    st.markdown("""
    <div class="hero">
        <h1>📊 Analisis Pembangunan Model</h1>
        <p>
            Tahapan pembangunan sistem deteksi gambar AI dan gambar nyata
            menggunakan algoritma <b>Random Forest</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)


    # Import library (tampilan)
    st.markdown("""
    <div class="card">
        <h3>📦 Import Library</h3>
        <p>Library yang digunakan dalam pembangunan sistem deteksi gambar AI dan gambar nyata</p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
# Core
import os
from pathlib import Path
from glob import glob
import numpy as np

# Image Processing
import cv2

# Feature Extraction
from skimage.feature import hog, local_binary_pattern

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
    """, language="python")


    # Dataset
    st.markdown("""
    <div class="card">
        <h3>📂 Dataset</h3>
        <p>Dataset terdiri dari dua kelas, yaitu AI Image dan Real Image.</p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
def get_dataset_paths():
    base_path = Path.cwd() / "dataset"
    ai_path = base_path / "AI Image"
    real_path = base_path / "Real Image"

    if not ai_path.exists() or not real_path.exists():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan!\\n"
            f"AI Path   : {ai_path}\\n"
            f"Real Path : {real_path}"
        )
    return ai_path, real_path

def load_image_paths(folder_path):
    return sorted(glob(str(folder_path / "*")))

ai_path, real_path = get_dataset_paths()
ai_images = load_image_paths(ai_path)
real_images = load_image_paths(real_path)

print(f"Jumlah AI Images   : {len(ai_images)}")
print(f"Jumlah Real Images : {len(real_images)}")
    """, language="python")


    # Statistik dataset
    st.markdown("""
    <div class="card">
        <h4>📊 Statistik Dataset</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>🤖 Gambar AI</strong>
            <h2>{len(ai_images)}</h2>
            <p>Dataset</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <strong>🖼️ Gambar Real</strong>
            <h2>{len(real_images)}</h2>
            <p>Dataset</p>
        </div>
        """, unsafe_allow_html=True)


    # Contoh gambar dataset
    st.markdown("""
    <div class="card">
        <h4>🖼️ Contoh Gambar Dataset</h4>
    </div>
    """, unsafe_allow_html=True)

    if ai_images and real_images:
        col1, col2 = st.columns(2)

        with col1:
            img_ai = cv2.cvtColor(cv2.imread(ai_images[0]), cv2.COLOR_BGR2RGB)
            st.image(img_ai, caption="🤖 Contoh Gambar AI", use_container_width=True)

        with col2:
            img_real = cv2.cvtColor(cv2.imread(real_images[0]), cv2.COLOR_BGR2RGB)
            st.image(img_real, caption="🖼️ Contoh Gambar Real", use_container_width=True)
    else:
        st.warning("Dataset kosong atau tidak ditemukan.")


    # Pra-pemrosesan citra
    st.markdown("""
    <div class="card">
        <h3>🔧 Pra-pemrosesan Citra</h3>
        <ul>
            <li>Resize citra menjadi 128 × 128 piksel</li>
            <li>Konversi warna ke RGB dan Grayscale</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
def preprocess_images(image_paths, label, size=(128, 128)):
    gray_images = []
    rgb_images = []
    labels = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, size)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        rgb_images.append(img_rgb)
        gray_images.append(img_gray)
        labels.append(label)

    return gray_images, rgb_images, labels

ai_gray, ai_rgb, ai_labels = preprocess_images(ai_images, label=1)
real_gray, real_rgb, real_labels = preprocess_images(real_images, label=0)
    """, language="python")

    # Feature Extraction
    st.markdown("""
    <div class="card">
        <h3>🎨 Ekstraksi Fitur</h3>
        <p>
            Pada tahap ini, citra diubah menjadi fitur numerik berdasarkan warna, bentuk, dan tekstur untuk keperluan klasifikasi.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # RGB Histogram
    st.markdown("""
    <div class="sub-card">
        <h4>🔴 RGB Histogram</h4>
        <p>
            Mengekstraksi distribusi intensitas warna pada kanal
            Red, Green, dan Blue.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
    def extract_rgb_histogram(images, bins=32):
        features = []
        for img in images:
            hist = []
            for i in range(3):
                h = cv2.calcHist([img], [i], None, [bins], [0,256])
                hist.extend(cv2.normalize(h, h).flatten())
            features.append(hist)
        return np.array(features)
    """, language="python")

    # HOG
    st.markdown("""
    <div class="sub-card">
        <h4>🟢 Histogram of Oriented Gradients (HOG)</h4>
        <p>
            Mengekstraksi informasi tepi dan bentuk objek berdasarkan
            arah gradien citra.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
    def extract_hog(images):
        return np.array([
            hog(img,
                orientations=9,
                pixels_per_cell=(8,8),
                cells_per_block=(2,2),
                block_norm='L2-Hys')
            for img in images
        ])
    """, language="python")

    # LBP
    st.markdown("""
    <div class="sub-card">
        <h4>🔵 Local Binary Pattern (LBP)</h4>
        <p>
            Mengekstraksi pola tekstur lokal untuk membedakan
            karakteristik citra AI dan citra nyata.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
    def extract_lbp(images):
        features = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            lbp = local_binary_pattern(gray, 24, 3, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0,26))
            features.append(hist / (hist.sum() + 1e-6))
        return np.array(features)
    """, language="python")

    # Feature Construction / Dataset Final
    st.markdown("""
    <div class="card">
        <h3>🧩 Pipeline Feature Fusion</h3>
        <p>Pada tahap ini, semua fitur digabungkan menjadi matriks fitur <code>X</code> dan label <code>y</code> untuk pelatihan model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
    def build_features(use_rgb=True, use_hog=True, use_lbp=True):
        X = []

        if use_rgb:
            X.append(extract_rgb_histogram(ai_rgb + real_rgb))
        if use_lbp:
            X.append(extract_lbp(ai_rgb + real_rgb))
        if use_hog:
            X.append(extract_hog(ai_gray + real_gray))

        return np.hstack(X)

    X = build_features(use_rgb=True, use_hog=False, use_lbp=False)
    y = np.array(ai_labels + real_labels)

    print("Final feature shape:", X.shape)
    """, language="python")

    # Training
    st.markdown("""
    <div class="card">
        <h3>🌳 Pelatihan Model Random Forest</h3>
        <p>
            Model Random Forest dilatih menggunakan data latih
            dengan parameter yang telah ditentukan.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(
        y_test, y_pred,
        target_names=["Real", "AI"]
    ))
    """, language="python")

    st.code(
    """
    precision    recall  f1-score   support

    Real           0.80      0.81      0.81        80
    AI             0.81      0.80      0.81        80

    accuracy                           0.81       160
    macro avg      0.81      0.81      0.81       160
    weighted avg   0.81      0.81      0.81       160
    """,
    language="text"
    )

    # Evaluation
    st.markdown("""
    <div class="card">
        <h3>📈 Evaluasi Model</h3>
        <p>
            Evaluasi performa model dilakukan menggunakan
            classification report dan confusion matrix.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt='d',
        xticklabels=["Real","AI"],
        yticklabels=["Real","AI"],
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    """, language="python")

    cm = np.array([
        [65, 15],
        [16, 64]
    ])

    fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=120)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "AI"],
        yticklabels=["Real", "AI"],
        ax=ax,
        cbar=True,
        square=True
    )

    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    ax.set_title("Confusion Matrix", fontsize=9)
    ax.tick_params(axis='both', labelsize=8)

    st.pyplot(fig, use_container_width=False)

    # Save Model
    st.markdown("""
    <div class="card">
        <h3>💾 Simpan Model</h3>
        <p>
            Model Random Forest yang telah dilatih disimpan ke dalam file SAV.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
    import pickle

    filename = "random_forest_ai_vs_real_model.sav"

    with open(filename, "wb") as file:
        pickle.dump(model, file)

    print(f"Model berhasil disimpan sebagai {filename}")
    """, language="python")

    # Footer
    st.markdown("""
    <div class="footer">
        © 2025 | Proyek Sains Data <br>
        Sistem Deteksi Gambar AI vs Gambar Nyata
    </div>
    """, unsafe_allow_html=True)
