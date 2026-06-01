import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------- Konfigurasi halaman ----------
st.set_page_config(
    page_title='Dashboard Kualitas Udara & Prediksi',
    page_icon=':bar_chart:',
    layout='wide', # Menggunakan layout wide agar map dan chart lebih luas
)

# ---------- Fungsi Load Data & Model (Cached) ----------
@st.cache_resource
def load_artefak():
    # Pastikan file pkl ada di folder yang sama
    try:
        model  = joblib.load('regresi_berganda.pkl')
        scaler = joblib.load('scaler.pkl')
        fitur  = joblib.load('fitur.pkl')
        return model, scaler, fitur
    except Exception:
        # Fallback sementara jika model belum ada agar web tidak error
        return None, None, []

@st.cache_data
def load_data():
    df = pd.read_csv('data_clean.csv')
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    
    # Mengembalikan nama stasiun dari One-Hot Encoding
    def get_station(row):
        if row.get('stasiun_DKI2 (Kelapa Gading)', 0) == 1: return 'Kelapa Gading'
        if row.get('stasiun_DKI3 (Jagakarsa)', 0) == 1: return 'Jagakarsa'
        if row.get('stasiun_DKI4 (Lubang Buaya)', 0) == 1: return 'Lubang Buaya'
        if row.get('stasiun_DKI5 (Kebon Jeruk)', 0) == 1: return 'Kebon Jeruk'
        return 'Bunderan HI'
    
    df['Stasiun'] = df.apply(get_station, axis=1)
    
    # Menambahkan koordinat untuk Peta
    koordinat = {
        'Bunderan HI': [-6.1947, 106.8236],
        'Kelapa Gading': [-6.1636, 106.9082],
        'Jagakarsa': [-6.3328, 106.8143],
        'Lubang Buaya': [-6.2925, 106.9077],
        'Kebon Jeruk': [-6.1950, 106.7636]
    }
    
    df['lat'] = df['Stasiun'].map(lambda x: koordinat[x][0])
    df['lon'] = df['Stasiun'].map(lambda x: koordinat[x][1])
    
    return df

# Load artefak dan data
model, scaler, FITUR = load_artefak()
df_clean = load_data()

# ---------- Header ----------
st.title('🌤️ Dashboard Prediksi & Sebaran Kualitas Udara Jakarta')
st.markdown('Jelajahi data historis kualitas udara dan lakukan prediksi regresi linear untuk mengestimasi tingkat polusi.')
st.divider()

# ---------- Menggunakan Tabs ----------
tab1, tab2 = st.tabs(["🗺️ Peta & Eksplorasi Data", "📊 Prediksi Regresi Linear"])

# ==========================================
# TAB 1: PETA & EKSPLORASI DATA
# ==========================================
with tab1:
    st.subheader("Peta Sebaran Stasiun Pemantau (ISPU)")
    
    # Menampilkan Peta
    # Mengambil satu baris per stasiun agar tidak tumpang tindih di peta
    df_map = df_clean[['Stasiun', 'lat', 'lon']].drop_duplicates()
    st.map(df_map, latitude='lat', longitude='lon', size=2000, color='#ff4b4b')
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tren Polutan per Tahun")
        # Ekstrak tahun
        df_clean['Tahun'] = df_clean['tanggal'].dt.year
        tren_tahunan = df_clean.groupby('Tahun')[['pm25', 'pm10', 'so2', 'co', 'o3', 'no2']].mean()
        st.line_chart(tren_tahunan)

    with col2:
        st.subheader("Perbandingan PM2.5 Antar Stasiun")
        # Bar chart rata-rata PM 2.5 per stasiun
        rata_stasiun = df_clean.groupby('Stasiun')['pm25'].mean().sort_values(ascending=False)
        st.bar_chart(rata_stasiun)
        
    with st.expander("Lihat Sampel Data Bersih"):
        st.dataframe(df_clean.head(10), use_container_width=True)


# ==========================================
# TAB 2: PREDIKSI MODEL
# ==========================================
with tab2:
    st.subheader('Prediksi Indeks Kualitas Udara (Regresi Linear)')
    
    col_input, col_hasil = st.columns([1, 2])
    
    with col_input:
        st.markdown('**Input Nilai Fitur**')
        input_user = {}
        if FITUR:
            # Membuat input dinamis dari array FITUR model Anda
            for f in FITUR:
                input_user[f] = st.number_input(
                    label=f,
                    value=0.0,
                    step=0.1,
                    format='%.4f',
                )
            tombol_prediksi = st.button('Mulai Prediksi', type='primary', use_container_width=True)
        else:
            st.error("Model/Artefak belum dimuat. Pastikan file .pkl tersedia.")
            tombol_prediksi = False

    with col_hasil:
        if tombol_prediksi:
            try:
                # Susun DataFrame sesuai urutan FITUR (hindari warning feature names)
                nilai = pd.DataFrame([[input_user[f] for f in FITUR]], columns=FITUR)
                nilai_sc = scaler.transform(nilai)
                pred = model.predict(nilai_sc)[0]

                # Tampilkan hasil
                st.success(f'### Hasil prediksi:  **{pred:,.4f}**')

                # Tampilkan input yang dipakai
                st.write('**Input yang Digunakan:**')
                st.dataframe(pd.DataFrame([input_user]), use_container_width=True, hide_index=True)

                # Tampilkan koefisien model (untuk transparansi)
                with st.expander('Lihat Koefisien Model (Terstandarisasi)'):
                    df_koef = pd.DataFrame({
                        'Fitur': FITUR,
                        'Koefisien': model.coef_.round(4),
                    })
                    st.dataframe(df_koef, use_container_width=True, hide_index=True)
                    st.caption(f'Intercept (β₀) = {model.intercept_:.4f}')

            except Exception as e:
                st.error(f'Terjadi error saat melakukan prediksi: {e}')
        else:
            st.info('👈 Masukkan nilai pada kolom di sebelah kiri, lalu klik **Mulai Prediksi**.')

# ---------- Footer ----------
st.divider()
st.caption('© 2026 Dibuat untuk PPKD Jakarta Selatan — Kejuruan Data Analyst')