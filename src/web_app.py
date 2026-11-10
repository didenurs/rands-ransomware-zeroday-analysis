import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import joblib
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from tokenizer import extract_tokens_from_json

st.set_page_config(
    page_title="RanDS Ransomware Detection & Zero-Day Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber Dark CSS
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stMetric {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .status-badge-ransom {
        background-color: #ef4444;
        color: white;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 15px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
    }
    .status-badge-benign {
        background-color: #10b981;
        color: white;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 15px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }
    .card-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(PROJECT_ROOT, "data_new")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
DEMO_DIR = os.path.join(PROJECT_ROOT, "demo_samples")
COMP_FILE = os.path.join(OUTPUT_DIR, "all_experiments_comparison.json")
SPLIT1_CSV = os.path.join(DATA_DIR, "split_scenario1.csv")
SPLIT2_CSV = os.path.join(DATA_DIR, "split_scenario2.csv")
SC1_FILE = os.path.join(OUTPUT_DIR, "results_Scenario_1_Known_Family.json")
SC2_FILE = os.path.join(OUTPUT_DIR, "results_Scenario_2_Zero_Day.json")

def load_data():
    comp_data = {}
    if os.path.exists(COMP_FILE):
        try:
            with open(COMP_FILE, 'r') as f:
                comp_data = json.load(f)
        except Exception:
            pass
            
    if not comp_data:
        if os.path.exists(SC1_FILE):
            try:
                with open(SC1_FILE, 'r') as f:
                    comp_data['Scenario_1_Known_Family'] = json.load(f)
            except Exception:
                pass
        if os.path.exists(SC2_FILE):
            try:
                with open(SC2_FILE, 'r') as f:
                    comp_data['Scenario_2_Zero_Day'] = json.load(f)
            except Exception:
                pass
                
    df_sp1 = pd.read_csv(SPLIT1_CSV) if os.path.exists(SPLIT1_CSV) else None
    df_sp2 = pd.read_csv(SPLIT2_CSV) if os.path.exists(SPLIT2_CSV) else None
    return (comp_data if comp_data else None), df_sp1, df_sp2

@st.cache_resource
def load_inference_pipeline(scenario_key="Scenario_2_Zero_Day"):
    """
    Loads the trained TF-IDF vectorizer and trained models from disk for real-time inference.
    """
    tfidf_path = os.path.join(MODELS_DIR, f"tfidf_{scenario_key}.joblib")
    svm_path = os.path.join(MODELS_DIR, f"svm_{scenario_key}.joblib")
    lr_path = os.path.join(MODELS_DIR, f"lr_{scenario_key}.joblib")
    rf_path = os.path.join(MODELS_DIR, f"rf_{scenario_key}.joblib")
    xgb_path = os.path.join(MODELS_DIR, f"xgb_{scenario_key}.json")

    tfidf = joblib.load(tfidf_path)
    models = {
        "Linear SVM (Kalibre)": joblib.load(svm_path),
        "Logistic Regression": joblib.load(lr_path),
        "Random Forest": joblib.load(rf_path)
    }
    if os.path.exists(xgb_path):
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_path)
        models["XGBoost"] = xgb_model

    return tfidf, models

comp_data, df_sp1, df_sp2 = load_data()

st.title("RanDS: Davranışsal Analiz & Zero-Day Ransomware Tespit Sistemi")
st.caption("Bitirme Tezi Çalışması Kapsamında Canlı Değerlendirme Arayüzü | Gelişmiş ML & DL Boru Hattı")

# Sidebar Overview
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=80)
st.sidebar.title("Kontrol Paneli")

if st.sidebar.button("Sonuçları Yenile", use_container_width=True):
    st.rerun()

has_sc1 = comp_data is not None and 'Scenario_1_Known_Family' in comp_data
has_sc2 = comp_data is not None and 'Scenario_2_Zero_Day' in comp_data

st.sidebar.markdown(f"""
**Eğitim Durumu:**
- **Senaryo 1 (Known Family):** {'Tamamlandı' if has_sc1 else 'Devam Ediyor'}
- **Senaryo 2 (Zero-Day):** {'Tamamlandı' if has_sc2 else 'Eğitiliyor...'}

**Veri Seti:** RanDS PE Behavioral Activities  
**Toplam Örnek:** 82,840  
- **Ransomware:** 51,985 (462 Aile)  
- **Benign:** 30,855  
**Temsiliyet:** 15.000 Boyutlu TF-IDF  
""")

tabs = st.tabs([
    "📊 Veri Seti & Ayrım (EDA)",
    "📈 Model Performans Karşılaştırması",
    "🎯 Detaylı Model İncelemesi & CM",
    "⚡ Canlı Test ve Tehdit İnceleme Simülatörü"
])

# ================= TAB 1: VERİ SETİ & AYRIM =================
with tabs[0]:
    st.header("1. Veri Seti Yapısı ve Ayrım Senaryoları")
    st.markdown("""
    RanDS veri setinde ham JSON dosyalarından çıkarılan 16 davranışsal aktivite kategorisi bulunmaktadır 
    (API çağrıları, registry işlemleri, silinen/oluşturulan dosyalar, mutexler, ağ trafiği vb.).
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Toplam Örnek", "82,840", "100%")
    with col2:
        st.metric("Fidye Yazılımı (Ransomware)", "51,985", "62.76%")
    with col3:
        st.metric("Zararsız (Benign)", "30,855", "37.24%")
    with col4:
        st.metric("Fidye Yazılımı Ailesi", "462 Aile", "Long-Tail Dağılım")
        
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Senaryo 1: Bilinen Aileler (Known Family Split)")
        st.markdown("""
        * **Kurgu:** 462 ailenin her birinin kendi içindeki örnekleri rastgele %70 Eğitim, %15 Doğrulama ve %15 Test olarak bölünmüştür.
        * **Amaç:** Modelin bilinen ailelerin yeni varyantlarını tanıma başarısını ölçmek.
        * **Dağılım:** Train: 58,012 | Val: 12,411 | Test: 12,417
        """)
        if df_sp1 is not None:
            fig1 = px.pie(df_sp1, names='Split', title='Senaryo 1 Dağılım Oranları', color='Split',
                          color_discrete_map={'train': '#3b82f6', 'val': '#f59e0b', 'test': '#10b981'})
            st.plotly_chart(fig1, use_container_width=True)
            
    with c2:
        st.subheader("Senaryo 2: Görülmemiş Aileler (Zero-Day Disjoint Split)")
        st.markdown("""
        * **Kurgu:** Aileler bölünmeden, toplam örneklerin %70'i Eğitim (324 aile), %15'i Val (69 aile) ve %15'i Test (69 aile) kümesine atanmıştır.
        * **Zero-Day Garantisi:** Testteki 69 aile eğitimde HİÇ GÖRÜLMEMİŞTİR (%100 Ayrık / Disjoint).
        * **Dağılım:** Train: 57,987 (36,389 ran + 21,598 ben) | Val: 12,426 | Test: 12,427
        """)
        if df_sp2 is not None:
            fig2 = px.pie(df_sp2, names='Split', title='Senaryo 2 Dağılım Oranları (Milimetrik %70/%15/%15)', color='Split',
                          color_discrete_map={'train': '#8b5cf6', 'val': '#ec4899', 'test': '#06b6d4'})
            st.plotly_chart(fig2, use_container_width=True)
            
    if df_sp1 is not None:
        st.subheader("En Büyük 20 Fidye Yazılımı Ailesi Dağılımı")
        top_fams = df_sp1[df_sp1['Label'] == 1]['Family'].value_counts().head(20).reset_index()
        top_fams.columns = ['Aile Adı', 'Örnek Sayısı']
        fig_bar = px.bar(top_fams, x='Aile Adı', y='Örnek Sayısı', color='Örnek Sayısı',
                         color_continuous_scale='Viridis', text='Örnek Sayısı')
        st.plotly_chart(fig_bar, use_container_width=True)

# ================= TAB 2: MODEL PERFORMANSI =================
with tabs[1]:
    st.header("2. Model Performans Karşılaştırma Analizi")
    
    if comp_data is None:
        st.info("Model eğitim süreci devam ediyor. İlk senaryo sonuçları tamamlandığında buraya yansıyacaktır.")
    else:
        avail_scenarios = list(comp_data.keys())
        if len(avail_scenarios) > 1:
            mode = st.radio("Kıyaslama Modu Seçin:", 
                            ["Yan Yana Kıyaslama (Senaryo 1 vs Senaryo 2)", 
                             "Sadece Senaryo 1 (Known Family)", 
                             "Sadece Senaryo 2 (Zero-Day)"], horizontal=True)
        else:
            mode = "Sadece Senaryo 1 (Known Family)"
            st.success("**Senaryo 1 (Bilinen Aileler)** sonuçları başarıyla yüklendi!")
            
        filtered_data = {}
        if "Yan Yana" in mode:
            filtered_data = comp_data
        elif "Senaryo 1" in mode:
            if 'Scenario_1_Known_Family' in comp_data:
                filtered_data['Scenario_1_Known_Family'] = comp_data['Scenario_1_Known_Family']
        else:
            if 'Scenario_2_Zero_Day' in comp_data:
                filtered_data['Scenario_2_Zero_Day'] = comp_data['Scenario_2_Zero_Day']
                
        rows = []
        for sc_name, m_dict in filtered_data.items():
            sc_label = "Senaryo 1 (Bilinen)" if "1" in sc_name else "Senaryo 2 (Zero-Day)"
            for m_name, metrics in m_dict.items():
                r = {'Senaryo': sc_label, 'Model': m_name}
                r.update({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in metrics.items()})
                rows.append(r)
        df_results = pd.DataFrame(rows)
        
        st.dataframe(df_results, use_container_width=True)
        
        chart_df = []
        for sc_name, m_dict in filtered_data.items():
            sc_label = "Senaryo 1 (Bilinen)" if "1" in sc_name else "Senaryo 2 (Zero-Day)"
            for m_name, metrics in m_dict.items():
                chart_df.append({
                    'Model': m_name,
                    'Senaryo': sc_label,
                    'F1_Score': metrics.get('F1_Score', 0),
                    'Accuracy': metrics.get('Accuracy', 0),
                    'Precision': metrics.get('Precision', 0),
                    'Recall': metrics.get('Recall', 0),
                    'FPR': metrics.get('FPR', 0),
                    'ROC_AUC': metrics.get('ROC_AUC', 0)
                })
        df_chart = pd.DataFrame(chart_df)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_f1 = px.bar(df_chart, x='Model', y='F1_Score', color='Senaryo', barmode='group',
                            title='F1-Score Karşılaştırması (Yüksek İyidir)', range_y=[0.7, 1.0])
            st.plotly_chart(fig_f1, use_container_width=True)
        with col_m2:
            fig_fpr = px.bar(df_chart, x='Model', y='FPR', color='Senaryo', barmode='group',
                             title='Yanlış Alarm Oranı (FPR - Düşük İyidir)', range_y=[0, 0.15])
            st.plotly_chart(fig_fpr, use_container_width=True)
            
        st.subheader("Zero-Day Başarım Kaybı (Degradation Analysis)")
        st.markdown("""
        Bilinen ailelerde eğitilen modeller, hiç görülmemiş yepyeni bir Zero-Day fidye yazılımı ailesiyle karşılaştığında 
        genelleme başarısı (Recall ve F1) yüksek kararlılığını korumakta ve yanlış alarm oranı (FPR) %2.3 bandında kontrol altında tutulmaktadır.
        """)

# ================= TAB 3: MODEL DETAYLARI =================
with tabs[2]:
    st.header("3. Detaylı Model İncelemesi & Karmaşıklık Matrisi (Confusion Matrix)")
    if comp_data is not None:
        sel_sc = st.selectbox("Senaryo Seçin:", list(comp_data.keys()))
        sel_model = st.selectbox("Model Seçin:", list(comp_data[sel_sc].keys()))
        
        m_info = comp_data[sel_sc][sel_model]
        
        cm_c1, cm_c2 = st.columns([1, 2])
        with cm_c1:
            st.subheader(f"{sel_model} Metrik Kartı")
            st.metric("Accuracy", f"{m_info['Accuracy']*100:.2f}%")
            st.metric("Precision", f"{m_info['Precision']*100:.2f}%")
            st.metric("Recall (Sensitivity)", f"{m_info['Recall']*100:.2f}%")
            st.metric("F1-Score", f"{m_info['F1_Score']*100:.2f}%")
            st.metric("FPR (False Positive Rate)", f"{m_info['FPR']*100:.2f}%")
            st.metric("ROC-AUC", f"{m_info['ROC_AUC']:.4f}")
            
        with cm_c2:
            st.subheader("Karmaşıklık Matrisi (Confusion Matrix)")
            tp = m_info['TP']
            fp = m_info['FP']
            tn = m_info['TN']
            fn = m_info['FN']
            
            total_benign = tn + fp
            total_ransom = fn + tp
            tn_pct = (tn / total_benign) * 100 if total_benign > 0 else 0
            fp_pct = (fp / total_benign) * 100 if total_benign > 0 else 0
            fn_pct = (fn / total_ransom) * 100 if total_ransom > 0 else 0
            tp_pct = (tp / total_ransom) * 100 if total_ransom > 0 else 0
            
            z = [[tn, fp], [fn, tp]]
            text = [
                [f"<b>TN: {tn:,}</b><br><span style='font-size:13px;'>Doğru Zararsız (%{tn_pct:.1f})</span>",
                 f"<b>FP: {fp:,}</b><br><span style='font-size:13px;'>Yanlış Alarm (%{fp_pct:.1f})</span>"],
                [f"<b>FN: {fn:,}</b><br><span style='font-size:13px;'>Kaçırılan Tehdit (%{fn_pct:.1f})</span>",
                 f"<b>TP: {tp:,}</b><br><span style='font-size:13px;'>Doğru Tespit (%{tp_pct:.1f})</span>"]
            ]
            
            colorscale = [
                [0.0, '#1e293b'],
                [0.1, '#1e3a5f'],
                [0.5, '#1d4ed8'],
                [1.0, '#38bdf8']
            ]
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=z,
                x=['Tahmin: Benign (0)', 'Tahmin: Ransomware (1)'],
                y=['Gerçek: Benign (0)', 'Gerçek: Ransomware (1)'],
                text=text,
                texttemplate="%{text}",
                textfont={"size": 16, "color": "#ffffff", "family": "Inter, sans-serif"},
                colorscale=colorscale,
                showscale=False,
                hoverinfo='none'
            ))
            fig_cm.update_layout(
                height=420,
                margin=dict(t=30, b=30, l=30, r=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(
                    autorange='reversed',
                    tickfont=dict(color='#e2e8f0', size=13, family='Inter, sans-serif'),
                    gridcolor='rgba(255,255,255,0.05)'
                ),
                xaxis=dict(
                    tickfont=dict(color='#e2e8f0', size=13, family='Inter, sans-serif'),
                    gridcolor='rgba(255,255,255,0.05)',
                    side='bottom'
                )
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.markdown(f"""
            <div style='background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px; font-size: 13px; color: #cbd5e1;'>
                <b>Matris Özeti:</b> Toplam <b>{tn+fp+fn+tp:,}</b> test örneğinden <b>{tn+tp:,}</b> tanesi doğru sınıflandırılmıştır 
                (Başarı: <b>%{(tn+tp)/(tn+fp+fn+tp)*100:.2f}</b>). 
                Sadece <b>{fp}</b> yanlış alarm (FP) ve <b>{fn}</b> kaçırılan tehdit (FN) bulunmaktadır.
            </div>
            """, unsafe_allow_html=True)

# ================= TAB 4: CANLI TEST VE TEHDİT İNCELEME SİMÜLATÖRÜ =================
with tabs[3]:
    st.header("4. Canlı Test ve Tehdit İnceleme Motoru (Real-Time Threat Inspector)")
    st.markdown("""
    Test kümesinden ayrılmış **3 gerçek davranış raporundan** (**WannaCry**, **Zero-Day** veya **Benign** JSON dosyalarından) 
    istediğinizi aşağıdaki alana **sürükleyip bırakın** ve **Canlı Analizi Başlat** butonuna tıklayın.  
    Sistem dosyayı **o anda canlı olarak** ayrıştırır, **15.000 boyutlu TF-IDF** vektörüne dönüştürür ve eğitilmiş modellerimiz üzerinde **gerçek zamanlı çıkarım** yürütür.
    """)

    with st.expander("Sürükle-Bırak İçin Hazır Test Dosyaları", expanded=False):
        st.caption(f"Dosya Konumu: `{DEMO_DIR}` — Klasörden doğrudan sürükleyebilir veya buradan bilgisayarınıza indirebilirsiniz:")
        dcol1, dcol2, dcol3 = st.columns(3)
        sample_files = [
            ("1_WannaCry_Known_Family.json", "1. WannaCry (Bilinen Aile).json", dcol1),
            ("2_ZeroDay_Unseen_Family.json", "2. Zero-Day (Görülmemiş Aile).json", dcol2),
            ("3_Benign_System_File.json", "3. Benign (Zararsız Dosya).json", dcol3),
        ]
        for fname, label, col_obj in sample_files:
            fpath = os.path.join(DEMO_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as fp_dl:
                    col_obj.download_button(
                        label=label,
                        data=fp_dl.read(),
                        file_name=fname,
                        mime="application/json",
                        use_container_width=True
                    )

    scenario_choice = st.radio(
        "Çıkarım Senaryosu Seçin",
        options=["Senaryo 1 — Bilinen Aileler (Known-Family)", "Senaryo 2 — Görülmemiş Aileler (Zero-Day)"],
        index=1,
        horizontal=True,
        help="Senaryo 1 modelleri bilinen aile varyantlarını, Senaryo 2 modelleri sıfır-gün ailelerini tespit etmek için eğitilmiştir."
    )
    sc_key = "Scenario_1_Known_Family" if "Senaryo 1" in scenario_choice else "Scenario_2_Zero_Day"

    uploaded_file = st.file_uploader(
        "Davranışsal Analiz Raporunu (.json) Buraya Sürükleyin ve Bırakın",
        type=["json"],
        help="demo_samples klasöründeki 1_WannaCry_Known_Family.json, 2_ZeroDay_Unseen_Family.json veya 3_Benign_System_File.json dosyasını sürükleyin."
    )

    if uploaded_file is not None:
        st.info(f"**Yüklenen Dosya:** `{uploaded_file.name}` ({uploaded_file.size:,} bayt) — Canlı analize hazır.")

        if st.button("Canlı Analizi Başlat", type="primary", use_container_width=True):
            t_start = time.perf_counter()

            # 1. Live Tokenization & Regex Normalization
            raw_bytes = uploaded_file.getvalue()
            tokens_str = extract_tokens_from_json(raw_bytes)
            token_list = tokens_str.split() if tokens_str else []

            # Group extracted tokens by semantic prefix for display
            grouped_tokens = {
                "API Çağrıları (API:)": [],
                "Kayıt Defteri (REG_CRE / REG_ACC)": [],
                "Dosya Eylemleri (FILE_CRE / DEL / CHG / ACC)": [],
                "Komut & İşlem (CMD / PROC_CRE / KIL)": [],
                "Mutex & Modüller (MUTEX / MOD)": [],
                "Ağ & DNS (NET_IP / DNS)": []
            }
            for tok in token_list:
                if tok.startswith("API:"):
                    grouped_tokens["API Çağrıları (API:)"].append(tok)
                elif tok.startswith("REG_"):
                    grouped_tokens["Kayıt Defteri (REG_CRE / REG_ACC)"].append(tok)
                elif tok.startswith("FILE_"):
                    grouped_tokens["Dosya Eylemleri (FILE_CRE / DEL / CHG / ACC)"].append(tok)
                elif tok.startswith("CMD:") or tok.startswith("PROC_"):
                    grouped_tokens["Komut & İşlem (CMD / PROC_CRE / KIL)"].append(tok)
                elif tok.startswith("MUTEX_") or tok.startswith("MOD:"):
                    grouped_tokens["Mutex & Modüller (MUTEX / MOD)"].append(tok)
                elif tok.startswith("NET_") or tok.startswith("DNS:"):
                    grouped_tokens["Ağ & DNS (NET_IP / DNS)"].append(tok)

            # 2. Live TF-IDF Transformation & Model Inference
            tfidf_vec, live_models = load_inference_pipeline(sc_key)

            X_live = tfidf_vec.transform([tokens_str])
            active_vocab_features = X_live.nnz

            model_rows = []
            probs_list = []
            model_order = ["Linear SVM (Kalibre)", "XGBoost", "Logistic Regression", "Random Forest"]
            for m_name in model_order:
                if m_name in live_models:
                    m_obj = live_models[m_name]
                    prob_ransom = float(m_obj.predict_proba(X_live)[0, 1]) * 100.0
                    probs_list.append(prob_ransom)
                    decision = "RANSOMWARE" if prob_ransom >= 50.0 else "BENIGN"
                    model_rows.append({
                        "Model": m_name,
                        "Fidye Yazılımı Olasılığı (%)": round(prob_ransom, 2),
                        "Karar": decision
                    })

            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            avg_prob = float(np.mean(probs_list)) if probs_list else 0.0
            primary_prob = model_rows[0]["Fidye Yazılımı Olasılığı (%)"] if model_rows else avg_prob

            st.markdown("---")
            # Top Live Telemetry Metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Çıkarılan Davranış Token'ı", f"{len(token_list):,}", "16 Kategori")
            mcol2.metric("Eşleşen TF-IDF Özniteliği", f"{active_vocab_features:,} / 15,000", "Sıfır Sızıntılı Sözlük")
            mcol3.metric("Ortalama Tehdit Olasılığı", f"%{avg_prob:.2f}", f"Linear SVM: %{primary_prob:.2f}")
            mcol4.metric("Canlı Çıkarım Süresi", f"{t_elapsed_ms:.1f} ms", "Gerçek Zamanlı")

            st.subheader("Canlı Karar Motoru Sonucu:")
            if avg_prob >= 50.0:
                badge_label = "ZERO-DAY RANSOMWARE - TEHLİKELİ" if sc_key == "Scenario_2_Zero_Day" else "RANSOMWARE - TEHLİKELİ"
                st.markdown(f'<span class="status-badge-ransom">{badge_label} (Canlı Olasılık: %{primary_prob:.2f})</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="status-badge-benign">BENIGN - GÜVENLİ YAZILIM (Canlı Olasılık: %{primary_prob:.2f})</span>', unsafe_allow_html=True)

            st.write("")
            res_col1, res_col2 = st.columns([1.15, 1])

            with res_col1:
                st.markdown("#### Modellerin Canlı Tahmin Tablosu")
                p_df = pd.DataFrame(model_rows)
                st.dataframe(p_df, use_container_width=True, hide_index=True)

                fig_live = px.bar(
                    p_df,
                    x="Fidye Yazılımı Olasılığı (%)",
                    y="Model",
                    orientation="h",
                    color="Karar",
                    text="Fidye Yazılımı Olasılığı (%)",
                    range_x=[0, 105],
                    color_discrete_map={"RANSOMWARE": "#ef4444", "BENIGN": "#10b981"},
                    title="Modellerin Canlı Fidye Yazılımı Olasılık Dağılımı (%)"
                )
                fig_live.update_layout(height=280, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig_live, use_container_width=True)

            with res_col2:
                st.markdown("#### Dosyadan Canlı Çıkarılan Davranışsal İzler")
                for cat_title, items in grouped_tokens.items():
                    unique_items = list(dict.fromkeys(items))
                    if unique_items:
                        preview = ", ".join(f"`{x}`" for x in unique_items[:6])
                        extra = f" *(+{len(unique_items)-6} benzersiz iz daha)*" if len(unique_items) > 6 else ""
                        st.markdown(f"* **{cat_title} ({len(items)} olay):** {preview}{extra}")
                    else:
                        st.markdown(f"* **{cat_title}:** *Kayda değer aktivite yok*")
