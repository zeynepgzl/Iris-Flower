import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image

# Sayfa başlığı ve arka plan ayarları
st.set_page_config(page_title="Iris Çiçeği Sınıflandırma", page_icon="🌸", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
            font-family: 'Roboto', sans-serif;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            color: #4CAF50;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stSlider>div>div>div {
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .input-container {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 1rem;
        }
        .slider-container {
            flex: 1;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
            border: none;
        }
        .result-container {
            margin-top: 2rem;
            text-align: center;
        }
        .prediction-proba {
            color: #777;
        }
        .info-box {
            font-size: 12px;
            color: #666;
            padding: 8px;
            background-color: #f0f0f0;
            border-radius: 5px;
            margin-top: 1rem;
            text-align: center;
        }
        .info-box span {
            font-weight: bold;
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Başlık
st.title("Iris Çiçeği Sınıflandırma")

# CSV dosyasını yükleme
@st.cache_data
def load_data():
    return pd.read_csv("IRIS.csv")

data = load_data()

# Özellikler ve etiketler
X = data.drop(columns=["Id", "Species"])  # "Id" sütununu ve hedef etiket olan "Species"i çıkarıyoruz
y = data["Species"]

# Eğitim ve Test Setine Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma ve eğitim
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Modelin sınıf adlarını almak için model.classes_ kullanıyoruz
target_names = model.classes_

# species_info sözlüğü
species_info = {
    'setosa': 'Setosa çiçeği, genellikle küçük boyutlarda ve mavi çiçeklere sahip bir türdür.',
    'versicolor': 'Versicolor çiçeği, orta boyutlarda ve renk değişimlerine sahip bir çiçek türüdür.',
    'virginica': 'Virginica çiçeği, daha büyük boyutlarda ve koyu renkli çiçeklere sahip bir türdür.'
}

# Görsellerin dosya yolları
species_images = {
    'setosa': 'images/setosa.png',
    'versicolor': 'images/versicolor.png',
    'virginica': 'images/virginica.png'
}

# Kullanıcıdan giriş almak
def user_input_features():
    uzunluk_sepal = st.number_input("Sepal Uzunluğu (cm)  ", min_value=0.0, max_value=10.0, value=5.0)
    genislik_sepal = st.number_input("Sepal Genişliği (cm) ", min_value=0.0, max_value=7.0, value=3.0)
    uzunluk_petal = st.number_input("Petal Uzunluğu (cm)  ", min_value=0.0, max_value=7.0, value=4.0)
    genislik_petal = st.number_input("Petal Genişliği (cm) ", min_value=0.0, max_value=3.0, value=1.2)
    
    data = {'SepalLengthCm': uzunluk_sepal, 
            'SepalWidthCm': genislik_sepal,
            'PetalLengthCm': uzunluk_petal, 
            'PetalWidthCm': genislik_petal}
    
    features = pd.DataFrame(data, index=[0])
    return features

# Kullanıcıdan alınan veriler
input_df = user_input_features()

# Kullanıcı girişine göre tahmin yapmak
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Sonuçları göstermek
st.subheader("Tahmin Sonuçları")

# Tahmin edilen türün ismini doğrudan prediction[0] ile alıyoruz
st.markdown(f"<h2 style='color: #4CAF50;'>Tahmin Edilen Çiçek Türü: <b>{prediction[0]}</b></h2>", unsafe_allow_html=True)
st.markdown(f"<h4 style='color: #4CAF50;'>Tahmin Edilen Türün Olasılığı: <b>{prediction_proba[0][target_names.tolist().index(prediction[0])] * 100:.2f}%</b></h4>", unsafe_allow_html=True)

# Diğer türlerin olasılıklarını küçük yazı ile alt alta gösterme
for i, name in enumerate(target_names):
    st.markdown(f"<p class='prediction-proba'>{name} Olasılığı: {prediction_proba[0][i] * 100:.2f}%</p>", unsafe_allow_html=True)

# Tahmin edilen tür ismini 'Iris-' önekinden arındırarak doğru formatta al
predicted_species = prediction[0].split('-')[1].lower()

# Tahmin edilen türle ilgili açıklamayı ekleyin
st.markdown(f"<p>{species_info[predicted_species]}</p>", unsafe_allow_html=True)

# Tahmin edilen çiçek türüne ait resmi ekleyin
image = Image.open(species_images[predicted_species])  # Resmi yükle
st.image(image, caption=f"{predicted_species.capitalize()} Çiçeği", use_container_width=True)  # use_container_width parametresi

# Sepal ve Petal terimlerinin anlamlarını açıklayan kutu
st.markdown("""
    <div class="info-box">
        <span>Sepal:</span> Çiçeğin dış kısmını oluşturan yaprakların altındaki yeşil yapraklar. Genellikle çiçeğin koruyucu katmanıdır.<br>
        <span>Petal:</span> Çiçeğin renkli kısmı, genellikle polinasyon amacıyla çekici hale gelir.<br><br>
        <span>Sepal Uzunluğu ve Genişliği:</span> Sepalin boyutları.<br>
        <span>Petal Uzunluğu ve Genişliği:</span> Petalin boyutları.
    </div>
""", unsafe_allow_html=True)
