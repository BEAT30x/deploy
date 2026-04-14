# ============================================
# STREAMLIT APP - PREDIKSI KALORI
# ============================================
# Cara menjalankan:
# streamlit run streamlit_app.py
# ============================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Prediksi Kalori",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    """Memuat model dari file pickle (dijalankan 1 kali saja)"""
    with open('model_complete.pkl', 'rb') as file:
        model_objects = pickle.load(file)
    
    return model_objects

# Load model
try:
    model_objects = load_model()
    model = model_objects['model']
    scaler = model_objects['scaler']
    le = model_objects['label_encoder']
    features = model_objects['features']
    model_info = model_objects.get('model_info', {})
    
    st.success("✅ Model berhasil dimuat!")
    
except FileNotFoundError:
    st.error("❌ File 'model_complete.pkl' tidak ditemukan!")
    st.info("Pastikan file model_complete.pkl berada di folder yang sama dengan file ini")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ============================================
# FUNGSI PREDIKSI
# ============================================
def predict_calories(age, weight, heart_rate, body_temp, gender, duration):
    """
    Memprediksi kalori yang terbakar
    """
    # Encoding gender
    gender_encoded = 0 if gender.lower() == 'male' else 1
    
    # Buat array input (urutan harus sesuai training)
    input_data = np.array([[age, weight, heart_rate, body_temp, gender_encoded, duration]])
    
    # Standarisasi
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)[0]
    
    # Kalori tidak mungkin negatif
    return max(0, prediction)

# ============================================
# SIDEBAR - INPUT DATA
# ============================================
st.sidebar.title("🔥 INPUT DATA ANDA")
st.sidebar.markdown("---")

# Input fields
st.sidebar.subheader("📊 Data Diri")
age = st.sidebar.number_input("Usia (tahun)", min_value=15, max_value=100, value=25, step=1)
weight = st.sidebar.number_input("Berat Badan (kg)", min_value=30, max_value=150, value=65, step=1)

st.sidebar.subheader("🏃 Aktivitas Fisik")
duration = st.sidebar.number_input("Durasi Olahraga (menit)", min_value=1, max_value=120, value=30, step=5)
heart_rate = st.sidebar.number_input("Detak Jantung (bpm)", min_value=50, max_value=200, value=120, step=5)

st.sidebar.subheader("🌡️ Data Fisik")
body_temp = st.sidebar.number_input("Suhu Tubuh (°C)", min_value=35.0, max_value=42.0, value=37.5, step=0.1, format="%.1f")

gender = st.sidebar.radio("Jenis Kelamin", options=['male', 'female'], horizontal=True)

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔥 PREDIKSI KALORI", type="primary", use_container_width=True)

# ============================================
# MAIN CONTENT - HEADER
# ============================================
st.title("🔥 PREDIKSI KONSUMSI KALORI HARIAN")
st.markdown("""
Aplikasi ini memprediksi jumlah kalori yang terbakar selama aktivitas fisik 
berdasarkan parameter seperti usia, berat badan, durasi olahraga, detak jantung, dan suhu tubuh.
""")

st.markdown("---")

# ============================================
# 2 KOLOM UNTUK INPUT RINGKASAN DAN HASIL
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 RINGKASAN INPUT")
    
    input_summary = pd.DataFrame({
        'Parameter': ['Usia', 'Berat Badan', 'Durasi', 'Detak Jantung', 'Suhu Tubuh', 'Jenis Kelamin'],
        'Nilai': [f"{age} tahun", f"{weight} kg", f"{duration} menit", f"{heart_rate} bpm", f"{body_temp} °C", "Pria" if gender == 'male' else "Wanita"]
    })
    
    st.dataframe(input_summary, hide_index=True, use_container_width=True)

with col2:
    st.subheader("🎯 HASIL PREDIKSI")
    
    if predict_button:
        with st.spinner("Menghitung..."):
            prediction = predict_calories(age, weight, heart_rate, body_temp, gender, duration)
        
        # Tampilkan hasil dengan format besar
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; 
                    border-radius: 20px; 
                    text-align: center;
                    color: white;">
            <h2 style="margin: 0; color: white;">🔥 Kalori Terbakar</h2>
            <h1 style="font-size: 64px; margin: 10px 0; color: white;">{prediction:.1f}</h1>
            <h3 style="margin: 0; color: white;">kalori</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Interpretasi hasil
        if prediction < 100:
            st.info("🏃‍♂️ **Intensitas Ringan** - Aktivitas ini membakar sedikit kalori. Cocok untuk pemula atau pemanasan.")
        elif prediction < 250:
            st.success("⚡ **Intensitas Sedang** - Pembakaran kalori cukup baik. Lanjutkan rutinitas ini!")
        else:
            st.warning("💪 **Intensitas Tinggi** - Pembakaran kalori sangat baik! Jaga hidrasi dan istirahat cukup.")
    else:
        st.info("👈 Masukkan data di sidebar, lalu klik tombol 'PREDIKSI KALORI'")

# ============================================
# GRAFIK PERBANDINGAN (BERDASARKAN INPUT)
# ============================================
if predict_button:
    st.markdown("---")
    st.subheader("📊 ANALISIS PREDIKSI")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Prediksi untuk berbagai durasi
        durations = [15, 30, 45, 60, 75, 90]
        preds_duration = [predict_calories(age, weight, heart_rate, body_temp, gender, d) for d in durations]
        
        fig_duration = px.line(
            x=durations, y=preds_duration,
            markers=True, template='plotly_white'
        )
        fig_duration.update_layout(
            title="Pengaruh Durasi terhadap Kalori",
            xaxis_title="Durasi (menit)",
            yaxis_title="Kalori",
            height=350
        )
        st.plotly_chart(fig_duration, use_container_width=True)
    
    with col4:
        # Prediksi untuk berbagai detak jantung
        heart_rates = [80, 100, 120, 140, 160, 180]
        preds_hr = [predict_calories(age, weight, hr, body_temp, gender, duration) for hr in heart_rates]
        
        fig_hr = px.line(
            x=heart_rates, y=preds_hr,
            markers=True, template='plotly_white',
            color_discrete_sequence=['orange']
        )
        fig_hr.update_layout(
            title="Pengaruh Detak Jantung terhadap Kalori",
            xaxis_title="Detak Jantung (bpm)",
            yaxis_title="Kalori",
            height=350
        )
        st.plotly_chart(fig_hr, use_container_width=True)

# ============================================
# BATCH PREDICTION (UPLOAD FILE)
# ============================================
st.markdown("---")
st.subheader("📁 BATCH PREDICTION (Upload File CSV)")

with st.expander("ℹ️ Cara menggunakan Batch Prediction"):
    st.markdown("""
    1. Buat file CSV dengan kolom berikut (urutan bebas):
       - `Age` (usia dalam tahun)
       - `Weight` (berat badan dalam kg)
       - `Heart_Rate` (detak jantung dalam bpm)
       - `Body_Temp` (suhu tubuh dalam °C)
       - `Gender` ('male' atau 'female')
       - `Duration` (durasi olahraga dalam menit)
    
    2. Upload file CSV
    
    3. Sistem akan memproses dan menampilkan hasil prediksi
    """)

uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    
    # Validasi kolom
    required_cols = ['Age', 'Weight', 'Heart_Rate', 'Body_Temp', 'Gender', 'Duration']
    missing_cols = [col for col in required_cols if col not in df_input.columns]
    
    if missing_cols:
        st.error(f"❌ Kolom yang hilang: {missing_cols}")
    else:
        # Prediksi batch
        predictions = []
        for _, row in df_input.iterrows():
            pred = predict_calories(
                age=row['Age'],
                weight=row['Weight'],
                heart_rate=row['Heart_Rate'],
                body_temp=row['Body_Temp'],
                gender=row['Gender'],
                duration=row['Duration']
            )
            predictions.append(pred)
        
        df_input['Predicted_Calories'] = predictions
        
        st.success(f"✅ Berhasil memprediksi {len(df_input)} data!")
        
        # Tampilkan hasil
        st.dataframe(df_input, use_container_width=True)
        
        # Download hasil
        csv_result = df_input.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Prediksi (CSV)",
            data=csv_result,
            file_name="hasil_prediksi_kalori.csv",
            mime="text/csv"
        )
        
        # Grafik distribusi hasil
        fig_hist = px.histogram(
            df_input, x='Predicted_Calories',
            title='Distribusi Hasil Prediksi',
            labels={'Predicted_Calories': 'Kalori'},
            template='plotly_white'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>Aplikasi Prediksi Kalori - Menggunakan Model KNN Regressor (R² = 99.54%)</p>
    <p>© 2026 - Teknik Informatika, Universitas Halu Oleo</p>
</div>
""", unsafe_allow_html=True)