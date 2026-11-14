import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE

# ==========================================
# KONFIGURASI HALAMAN & TEMA VISUAL
# ==========================================
st.set_page_config(
    page_title="Enterprise Churn Analytics | Aris Hardiansah",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set global style untuk Matplotlib & Seaborn agar tampak modern & bersih
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#E2E8F0'
plt.rcParams['axes.linewidth'] = 0.8

# Custom CSS untuk Enterprise Dashboard Look
st.markdown("""
<style>
    /* Global Background & Font */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Hero Banner Header */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #2563EB 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 6px 0;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #CBD5E1;
        margin: 0;
        font-weight: 400;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 10px 18px;
        border-radius: 12px;
        text-align: right;
    }
    .hero-badge-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #93C5FD;
        display: block;
    }
    .hero-badge-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.04);
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: #2563EB;
    }
    .kpi-card.accent-green::before { background: #10B981; }
    .kpi-card.accent-amber::before { background: #F59E0B; }
    .kpi-card.accent-rose::before { background: #F43F5E; }
    .kpi-card.accent-indigo::before { background: #6366F1; }

    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #475569;
    }

    /* Section Card Wrapper */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
        margin-top: 10px;
        margin-bottom: 16px;
    }

    /* Sidebar Styling */
    .sidebar-profile {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 16px;
        border-radius: 12px;
        color: white;
        margin-bottom: 16px;
        border: 1px solid #334155;
    }
    .sidebar-profile-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 2px 0 0 0;
    }
    .sidebar-profile-role {
        font-size: 0.78rem;
        color: #38BDF8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0;
    }

    /* Custom Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.92rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #1E3A8A !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
    }

    /* Prediction Result Box */
    .pred-card-churn {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
        border: 1px solid #FECDD3;
        border-left: 6px solid #E11D48;
        border-radius: 14px;
        padding: 24px;
        color: #881337;
    }
    .pred-card-retain {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 1px solid #A7F3D0;
        border-left: 6px solid #059669;
        border-radius: 14px;
        padding: 24px;
        color: #064E3B;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR: BRANDING, UPLOAD & KONFIGURASI
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-profile">
        <p class="sidebar-profile-role">🏦 Churn Intelligence System</p>
        <p class="sidebar-profile-name">👨‍💻 Aris Hardiansah</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 1. Sumber Dataset")
    uploaded_file = st.file_uploader("Unggah File Dataset (.CSV)", type=["csv"], help="Pastikan memiliki kolom: Gender, Age, CreditScore, EstimatedSalary, HasCrCard, Exited")
    use_default = st.toggle("Gunakan Dataset Bawaan", value=True)

    st.markdown("---")
    st.markdown("### ⚙️ 2. Parameter Pipeline ML")
    selected_model_name = st.selectbox(
        "Algoritma Klasifikasi",
        ["Random Forest", "Gradient Boosting", "Decision Tree", "K-Nearest Neighbor (KNN)"]
    )
    test_size_pct = st.slider("Rasio Data Testing (%)", min_value=10, max_value=40, value=25, step=5)
    
    st.markdown("##### 🔧 Optimasi Fitur & Kelas")
    use_fe = st.toggle("Feature Engineering (CreditScore/Age)", value=True)
    use_smote = st.toggle("SMOTE Oversampling (Imbalance)", value=True)

    st.markdown("---")
    st.caption("© 2026 **Aris Hardiansah** • Enterprise ML Dashboard")

# ==========================================
# FUNGSI LOAD & PREPROCESSING DATA
# ==========================================
@st.cache_data
def load_default_data():
    try:
        return pd.read_csv("Dataset1_Customer_Churn.csv")
    except FileNotFoundError:
        url = "https://raw.githubusercontent.com/intanwidiyanti/Bank-Customer-Churn/main/Dataset1_Customer_Churn.csv"
        return pd.read_csv(url)

df = None
dataset_source_name = ""
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    dataset_source_name = uploaded_file.name
elif use_default:
    df = load_default_data()
    dataset_source_name = "Dataset1_Customer_Churn.csv (Default)"

# ==========================================
# HERO BANNER HEADER
# ==========================================
st.markdown(f"""
<div class="hero-banner">
    <div>
        <h1 class="hero-title">🏦 Bank Customer Churn Classification</h1>
        <p class="hero-subtitle">Platform Analitik & Prediksi Retensi Nasabah Berbasis Machine Learning</p>
    </div>
    <div class="hero-badge">
        <span class="hero-badge-label">Developed By</span>
        <span class="hero-badge-name">Aris Hardiansah</span>
    </div>
</div>
""", unsafe_allow_html=True)

if df is None:
    st.info("👈 Silakan unggah file CSV dataset Anda melalui panel **Sumber Dataset** di Sidebar sebelah kiri, atau aktifkan tombol **Gunakan Dataset Bawaan**.")
    st.stop()

# Validasi kolom wajib
required_cols = {'Gender', 'Age', 'CreditScore', 'EstimatedSalary', 'HasCrCard', 'Exited'}
if not required_cols.issubset(set(df.columns)):
    st.error(f"❌ Struktur kolom CSV tidak valid! Pastikan file CSV mengandung kolom berikut: `{', '.join(required_cols)}`")
    st.stop()

# ==========================================
# DATA PIPELINE & MODEL TRAINING
# ==========================================
df_clean = df.copy()
if 'CustomerId' in df_clean.columns:
    df_clean = df_clean.drop(['CustomerId'], axis=1)

df_encoded = pd.get_dummies(df_clean, drop_first=True)
if 'Gender_Male' in df_encoded.columns:
    df_encoded['Gender_Male'] = df_encoded['Gender_Male'].astype(int)

df_model = df_encoded.copy()
if use_fe:
    df_model['Credit Score Given Age'] = df_model['CreditScore'] / df_model['Age']

X = df_model.drop(columns=["Exited"]).copy()
y = df_model["Exited"].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(test_size_pct / 100.0), random_state=42
)

if use_smote:
    smote = SMOTE(random_state=42)
    X_train_fit, y_train_fit = smote.fit_resample(X_train, y_train)
else:
    X_train_fit, y_train_fit = X_train, y_train

@st.cache_resource
def train_all_classifiers(X_tr, y_tr, _X_full, _y_full, _X_te, _y_te):
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbor (KNN)": KNeighborsClassifier()
    }
    trained = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        y_pred = m.predict(_X_te)
        acc = accuracy_score(_y_te, y_pred)
        auc_val = roc_auc_score(_y_full, m.predict(_X_full))
        fpr, tpr, _ = roc_curve(_y_full, m.predict_proba(_X_full)[:, 1])
        trained[name] = {
            "model": m,
            "y_pred": y_pred,
            "acc": acc,
            "report": classification_report(_y_te, y_pred, output_dict=True),
            "cm": confusion_matrix(_y_te, y_pred),
            "auc": auc_val,
            "fpr": fpr,
            "tpr": tpr
        }
    return trained

trained_models = train_all_classifiers(X_train_fit, y_train_fit, X, y, X_test, y_test)
active_res = trained_models[selected_model_name]

# ==========================================
# GLOBAL KPI SUMMARY BAR
# ==========================================
total_rows = len(df)
churn_total = int(df['Exited'].sum())
retained_total = total_rows - churn_total
churn_pct = (churn_total / total_rows) * 100

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Populasi Nasabah</div>
        <div class="kpi-value">{total_rows:,}</div>
        <div class="kpi-sub">Sumber: {dataset_source_name[:26]}</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi-card accent-green">
        <div class="kpi-title">Nasabah Bertahan (Retained)</div>
        <div class="kpi-value">{retained_total:,}</div>
        <div class="kpi-sub">{100 - churn_pct:.1f}% dari total populasi</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="kpi-card accent-rose">
        <div class="kpi-title">Nasabah Churn (Exited)</div>
        <div class="kpi-value">{churn_total:,}</div>
        <div class="kpi-sub">Churn Rate: {churn_pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="kpi-card accent-indigo">
        <div class="kpi-title">Model Aktif ({selected_model_name})</div>
        <div class="kpi-value">AUC {active_res['auc']:.3f}</div>
        <div class="kpi-sub">Akurasi Test: {active_res['acc']*100:.1f}% | {'SMOTE ON' if use_smote else 'SMOTE OFF'}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ==========================================
# NAVIGASI TABS UTAMA
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🗂️ Eksplorasi & Preprocessing Data",
    "📊 Visualisasi Analitik (EDA)",
    "📈 Performa & Evaluasi Model",
    "🎯 Simulator Prediksi Nasabah"
])

# ------------------------------------------
# TAB 1: EKSPLORASI & PREPROCESSING DATA
# ------------------------------------------
with tab1:
    col_t1_a, col_t1_b = st.columns([1.35, 1])
    with col_t1_a:
        st.markdown('<div class="section-title">📋 Tabel Data Mentah (Raw Dataset)</div>', unsafe_allow_html=True)
        st.dataframe(df.head(12), use_container_width=True, height=340)
        
        csv_download = df_model.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Unduh Dataset Hasil Preprocessing (.CSV)",
            data=csv_download,
            file_name="Processed_Customer_Churn_ArisHardiansah.csv",
            mime="text/csv"
        )

    with col_t1_b:
        st.markdown('<div class="section-title">⚙️ Transformasi & Encoding Fitur</div>', unsafe_allow_html=True)
        st.dataframe(
            df_model.head(12).style.format({"EstimatedSalary": "{:,.2f}", "Credit Score Given Age": "{:.2f}"} if use_fe else {"EstimatedSalary": "{:,.2f}"}),
            use_container_width=True,
            height=340
        )
        st.caption("Kolom `CustomerId` dihapus, `Gender` diubah menjadi biner (`Gender_Male`), dan fitur rasio `Credit Score Given Age` ditambahkan.")

    st.markdown('<div class="section-title">📐 Ringkasan Statistik & Keseimbangan Kelas Training</div>', unsafe_allow_html=True)
    col_stat1, col_stat2 = st.columns([1.5, 1])
    with col_stat1:
        st.dataframe(df_clean.describe().T.style.format("{:,.2f}"), use_container_width=True)
    with col_stat2:
        dist_df = pd.DataFrame({
            "Sebelum SMOTE": y_train.value_counts(),
            "Sesudah Pipeline": y_train_fit.value_counts()
        }).rename(index={0: "Kelas 0 (Retained)", 1: "Kelas 1 (Exited)"})
        st.dataframe(dist_df, use_container_width=True)
        st.caption(f"Total fitur yang masuk ke model: **{X.shape[1]} Fitur** (`{', '.join(X.columns)}`)")

# ------------------------------------------
# TAB 2: VISUALISASI ANALITIK (EDA)
# ------------------------------------------
with tab2:
    col_v1, col_v2 = st.columns([1, 1.3])
    
    with col_v1:
        st.markdown('<div class="section-title">🍩 Proporsi Retensi vs Churn</div>', unsafe_allow_html=True)
        fig_donut, ax_donut = plt.subplots(figsize=(6, 4.5))
        label_counts = df['Exited'].value_counts()
        wedges, texts, autotexts = ax_donut.pie(
            label_counts,
            labels=['Retained (0)', 'Exited / Churn (1)'],
            colors=['#10B981', '#F43F5E'],
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.78,
            explode=(0.04, 0.04),
            textprops={'fontsize': 11, 'weight': 'bold', 'color': '#1E293B'}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
        centre_circle = plt.Circle((0, 0), 0.55, fc='white')
        fig_donut.gca().add_artist(centre_circle)
        ax_donut.axis('equal')
        plt.tight_layout()
        st.pyplot(fig_donut)

    with col_v2:
        st.markdown('<div class="section-title">🔥 Matriks Korelasi Antar Variabel</div>', unsafe_allow_html=True)
        fig_corr, ax_corr = plt.subplots(figsize=(7.5, 4.5))
        sns.heatmap(
            df_model.corr(),
            annot=True,
            cmap='RdBu_r',
            center=0,
            fmt=".2f",
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax_corr
        )
        plt.xticks(rotation=25, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_corr)

    st.markdown('<div class="section-title">📊 Distribusi Atribut Kategorikal & Numerik terhadap Status Churn</div>', unsafe_allow_html=True)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_cat, axes_cat = plt.subplots(1, 2, figsize=(9, 3.8))
        sns.countplot(x='HasCrCard', hue='Exited', data=df, palette=['#10B981', '#F43F5E'], ax=axes_cat[0])
        axes_cat[0].set_title("Kepemilikan Kartu Kredit", fontsize=10, weight='bold')
        axes_cat[0].set_xlabel("HasCrCard (0=Tidak, 1=Ya)")
        
        sns.countplot(x='Gender', hue='Exited', data=df, palette=['#10B981', '#F43F5E'], ax=axes_cat[1])
        axes_cat[1].set_title("Jenis Kelamin Nasabah", fontsize=10, weight='bold')
        axes_cat[1].set_xlabel("Gender")
        plt.tight_layout()
        st.pyplot(fig_cat)

    with col_c2:
        fig_box, axes_box = plt.subplots(1, 3, figsize=(10, 3.8))
        sns.boxplot(y='CreditScore', x='Exited', hue='Exited', data=df, palette=['#10B981', '#F43F5E'], ax=axes_box[0], legend=False)
        axes_box[0].set_title("Credit Score", fontsize=10, weight='bold')
        
        sns.boxplot(y='Age', x='Exited', hue='Exited', data=df, palette=['#10B981', '#F43F5E'], ax=axes_box[1], legend=False)
        axes_box[1].set_title("Usia (Age)", fontsize=10, weight='bold')
        
        sns.boxplot(y='EstimatedSalary', x='Exited', hue='Exited', data=df, palette=['#10B981', '#F43F5E'], ax=axes_box[2], legend=False)
        axes_box[2].set_title("Estimated Salary", fontsize=10, weight='bold')
        plt.tight_layout()
        st.pyplot(fig_box)

# ------------------------------------------
# TAB 3: PERFORMA & EVALUASI MODEL
# ------------------------------------------
with tab3:
    st.markdown('<div class="section-title">🏆 Leaderboard Komparasi Semua Algoritma</div>', unsafe_allow_html=True)
    
    comp_rows = []
    for m_name, res in trained_models.items():
        r_dict = res["report"]
        comp_rows.append({
            "Algoritma Model": m_name,
            "Accuracy (%)": res["acc"] * 100,
            "ROC-AUC Score": res["auc"],
            "Precision Churn (1)": r_dict["1"]["precision"] * 100,
            "Recall Churn (1)": r_dict["1"]["recall"] * 100,
            "F1-Score Churn (1)": r_dict["1"]["f1-score"] * 100
        })
    comp_df = pd.DataFrame(comp_rows).sort_values(by="ROC-AUC Score", ascending=False).reset_index(drop=True)
    st.dataframe(
        comp_df.style.format({
            "Accuracy (%)": "{:.2f}%",
            "ROC-AUC Score": "{:.4f}",
            "Precision Churn (1)": "{:.2f}%",
            "Recall Churn (1)": "{:.2f}%",
            "F1-Score Churn (1)": "{:.2f}%"
        }).highlight_max(subset=["ROC-AUC Score", "Accuracy (%)", "Recall Churn (1)"], color="#DCFCE7"),
        use_container_width=True
    )

    col_ev1, col_ev2 = st.columns([1, 1.15])
    with col_ev1:
        st.markdown(f'<div class="section-title">🎯 Detail Evaluasi: {selected_model_name}</div>', unsafe_allow_html=True)
        rep_df = pd.DataFrame(active_res["report"]).transpose()
        st.dataframe(rep_df.style.format("{:.3f}"), use_container_width=True)

        st.markdown(f'<div class="section-title">🧩 Confusion Matrix ({selected_model_name})</div>', unsafe_allow_html=True)
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 3.6))
        sns.heatmap(
            active_res["cm"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["Pred: Retained (0)", "Pred: Exited (1)"],
            yticklabels=["Actual: Retained (0)", "Actual: Exited (1)"],
            ax=ax_cm
        )
        plt.tight_layout()
        st.pyplot(fig_cm)

    with col_ev2:
        st.markdown('<div class="section-title">📈 Kurva ROC-AUC Multi-Model</div>', unsafe_allow_html=True)
        fig_roc, ax_roc = plt.subplots(figsize=(7, 4.2))
        palette_roc = ['#2563EB', '#10B981', '#F59E0B', '#8B5CF6']
        for idx, (name, res) in enumerate(trained_models.items()):
            lw = 2.8 if name == selected_model_name else 1.5
            alpha = 1.0 if name == selected_model_name else 0.65
            ax_roc.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['auc']:.4f})", linewidth=lw, alpha=alpha, color=palette_roc[idx])
        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='#94A3B8', label='Random Guess (0.5000)')
        ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", weight='bold')
        ax_roc.set_ylabel("True Positive Rate (Sensitivity)", weight='bold')
        ax_roc.legend(loc="lower right", frameon=True, facecolor="white")
        plt.tight_layout()
        st.pyplot(fig_roc)

        if hasattr(active_res["model"], "feature_importances_"):
            st.markdown(f'<div class="section-title">🌟 Feature Importance ({selected_model_name})</div>', unsafe_allow_html=True)
            fi_df = pd.DataFrame({
                "Fitur": X.columns,
                "Importance": active_res["model"].feature_importances_
            }).sort_values(by="Importance", ascending=True)
            
            fig_fi, ax_fi = plt.subplots(figsize=(7, 3.0))
            bars = ax_fi.barh(fi_df["Fitur"], fi_df["Importance"], color="#1E3A8A", edgecolor="none", height=0.6)
            ax_fi.bar_label(bars, fmt="%.3f", padding=5, fontsize=9, weight='bold')
            ax_fi.set_xlim(0, fi_df["Importance"].max() * 1.18)
            ax_fi.set_xlabel("Tingkat Kepentingan Fitur (Relative Importance)")
            plt.tight_layout()
            st.pyplot(fig_fi)

# ------------------------------------------
# TAB 4: SIMULATOR PREDIKSI NASABAH
# ------------------------------------------
with tab4:
    col_sim_left, col_sim_right = st.columns([1.15, 1])

    with col_sim_left:
        st.markdown('<div class="section-title">📝 Input Parameter Profil Nasabah</div>', unsafe_allow_html=True)
        with st.form("enterprise_prediction_form"):
            f_c1, f_c2 = st.columns(2)
            with f_c1:
                in_gender = st.selectbox("Jenis Kelamin (Gender)", ["Female", "Male"])
                in_age = st.slider("Usia Nasabah (Tahun)", min_value=18, max_value=95, value=42)
                in_crcard = st.selectbox(
                    "Status Kepemilikan Kartu Kredit",
                    [1, 0],
                    format_func=lambda x: "Ya - Memiliki Kartu Kredit (1)" if x == 1 else "Tidak Memiliki Kartu Kredit (0)"
                )
            with f_c2:
                in_creditscore = st.number_input("Skor Kredit (Credit Score)", min_value=300, max_value=850, value=619, step=5)
                in_salary = st.number_input("Estimasi Pendapatan Tahunan ($)", min_value=0.0, max_value=500000.0, value=101348.88, step=1000.0)
                st.markdown("<br>", unsafe_allow_html=True)
                st.caption(f"Model Aktif: **{selected_model_name}** ({'Dengan SMOTE' if use_smote else 'Tanpa SMOTE'})")

            btn_predict = st.form_submit_button("🚀 Jalankan Analisis & Klasifikasi Nasabah", use_container_width=True)

    with col_sim_right:
        st.markdown('<div class="section-title">📡 Output Keputusan Klasifikasi</div>', unsafe_allow_html=True)
        
        gender_val = 1 if in_gender == "Male" else 0
        input_dict = {
            'Age': [in_age],
            'CreditScore': [in_creditscore],
            'EstimatedSalary': [in_salary],
            'HasCrCard': [in_crcard],
            'Gender_Male': [gender_val]
        }
        if use_fe:
            input_dict['Credit Score Given Age'] = [in_creditscore / in_age]

        input_df = pd.DataFrame(input_dict)[X.columns]
        model_used = active_res["model"]
        pred_label = int(model_used.predict(input_df)[0])
        pred_prob = model_used.predict_proba(input_df)[0]
        prob_retain = pred_prob[0] * 100
        prob_churn = pred_prob[1] * 100

        if pred_label == 1:
            st.markdown(f"""
            <div class="pred-card-churn">
                <h3 style="margin:0 0 8px 0; color:#9F1239;">⚠️ STATUS: BERPOTENSI CHURN / EXITED (KELAS 1)</h3>
                <p style="margin:0; font-size:0.95rem;">Model memprediksi nasabah ini memiliki risiko tinggi untuk menutup rekening dengan tingkat keyakinan <b>{prob_churn:.2f}%</b>.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-card-retain">
                <h3 style="margin:0 0 8px 0; color:#065F46;">✅ STATUS: NASABAH BERTAHAN / RETAINED (KELAS 0)</h3>
                <p style="margin:0; font-size:0.95rem;">Model memprediksi nasabah ini akan tetap loyal menggunakan layanan bank dengan tingkat keyakinan <b>{prob_retain:.2f}%</b>.</p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown("##### 📊 Distribusi Probabilitas Prediksi")
        st.progress(int(prob_churn), text=f"Risiko Churn (Exited): {prob_churn:.2f}%  |  Peluang Bertahan (Retained): {prob_retain:.2f}%")

        st.markdown("##### 🔍 Vektor Fitur Input Model")
        st.dataframe(
            input_df.style.format({"EstimatedSalary": "${:,.2f}", "Credit Score Given Age": "{:.3f}"} if use_fe else {"EstimatedSalary": "${:,.2f}"}),
            use_container_width=True
        )
