import streamlit as st
import pandas as pd
import os
from main import VideoFeatureExtractor

# --- Ini Konfigurasi Halamannya ---
st.set_page_config(
    page_title="10122082 - Yusuf Simangunsong - IF-3",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Ini CSS Custom aja pak Buat Styling Web Streamlit nya---
st.markdown("""
<style>
    /* Style untuk container dengan border */
    .st-emotion-cache-1r4qj8v {
        border: 1px solid #e6e6e6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Style untuk tombol utama */
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.title("Dashboard Analisis Video Tugas Praktik Rekayasa Fitur")
    st.info("""
        **Welcome**\n
        Nama : Yusuf Simangunsong\n
        NIM  : 10122082\n
        Kelas: IF-3\n
        - **Data:** 5 video dengan karakteristik berbeda.
        - **Fitur:** Warna, Gerakan, Kompleksitas, dan Tekstur.
        - **Tujuan:** Menunjukkan bagaimana fitur dapat membedakan konten video secara kuantitatif.
    """)
    st.success("Tekan tombol 'Run Program' untuk Running Program")


# --- Halaman Utama nya ---
st.header("Program Analisis Fitur")
st.write("""
Klik button Run Program buat jalanin skrip `main.py`. Proses ini bakal ngebaca semua video yang ada di folder `data/`,
terus ngekstrak 4 fitur berbeda, terus nyimpen hasilnya ke `output/`, terus nampilin visualisasinya di sini.
""")

if st.button("Run Program", type="primary"):
    # 1. Ini buat inisialisasi dan jalanin prosesnya
    with st.spinner("Sebentar... Lagi NgeRunning Program... Agak Lama... Punten..."):
        try:
            extractor = VideoFeatureExtractor(data_folder='data/', output_folder='output/')
            success = extractor.run()
            if not success:
                st.error("Wah kayaknya ada error pas running program.")
            else:
                st.success("Analisis Selesai")
        except Exception as e:
            st.error(f"Terjadi error: {e}")
            success = False

    # 2. Kalo berhasil, Nampilin hasilnya
    if success:
        st.markdown("---")
        st.header("Hasil Analisis Kuantitatif & Visual")
        
        # Baca hasilnya dari file CSV
        try:
            df = pd.read_csv('output/features.csv')

            # Nampilin Metrik Ringkasannya dari csv
            st.subheader("Ringkasan Utama")
            cols = st.columns(len(df))
            for i, col in enumerate(cols):
                with col:
                    with st.container(border=True):
                        video_name = df.iloc[i]['filename'].replace('.mp4', '')
                        st.markdown(f"**{video_name}**")
                        st.metric("Gerakan (Flow)", f"{df.iloc[i]['avg_flow_magnitude']:.2f}")
                        st.metric("Kompleksitas (Edge)", f"{df.iloc[i]['avg_edge_density']:.3f}")

            # Nampilin visualisasinya dalam bentuk tab
            st.subheader("Visualisasi Detail Fitur")
            tab1, tab2, tab3, tab4 = st.tabs(["Gerakan", "Warna", "Kompleksitas", "Tekstur"])

            with tab1:
                st.image('output/visualisasi_gerakan.png')
            with tab2:
                st.image('output/visualisasi_warna.png')
            with tab3:
                st.image('output/visualisasi_kompleksitas.png')
            with tab4:
                st.image('output/visualisasi_tekstur.png')
            
            # Nampilin tabel data mentahnya
            with st.expander("Lihat Data Fitur Lengkap (Tabel)"):
                st.dataframe(df)

        except FileNotFoundError:
            st.warning("File hasil `features.csv` tidak ditemukan.")
else:
    st.markdown("---")
    st.info("Lagi Nunggu buat di-Run Programnya. Klik button Run Program buat mulainya pak.")