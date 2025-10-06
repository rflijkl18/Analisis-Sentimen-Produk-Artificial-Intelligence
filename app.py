import random
import numpy as np
random.seed(42)
np.random.seed(42)

from preprocessing import preprocessing
from scrapers import scrape_playstore, scrape_twitter
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# State session untuk menyimpan hasil per produk
if 'hasil_produk' not in st.session_state:
    st.session_state['hasil_produk'] = {}

# --- auto_label revisi ---
def auto_label(text):
    positive_keywords = [
        'bagus', 'mantap', 'suka', 'cepat', 'keren', 'oke', 'puas','baik'
        'hebat', 'terbaik', 'membantu', 'luar biasa', 'fitur bagus', 'rekomendasi',
        'berguna', 'memuaskan', 'solutif', 'simple', 'praktis', 'mudah digunakan',
    ]
    negative_keywords = [
        'jelek', 'buruk', 'lemot', 'benci', 'error', 'lag', 'parah', 'mengecewakan',
        'tidak berfungsi', 'crash', 'hang', 'sampah', 'lelet', 'menyebalkan',
        'ribet', 'tidak jelas', 'tidak membantu', 'fitur hilang', 'sering keluar', 
        'lama' ,'ilang','lemot','gj','lolot','kesal','error','gak jelas','lambat',
        'lama','tolol','paok','gak nyambung'
    ]
    text = text.lower()
    pos_score = sum(k in text for k in positive_keywords)
    neg_score = sum(k in text for k in negative_keywords)
    if pos_score > neg_score:
        return "positive"
    elif neg_score > pos_score:
        return "negative"
    else:
        return "neutral"

# Mulai Streamlit
st.title("Aplikasi Analisis Sentimen Produk AI 🇮🇩")
menu = ["Upload CSV", "Upload Siri", "Scrape Data", "Perbandingan Produk"]
tab = st.sidebar.radio("Menu", menu)

# =================== SCRAPE ===================
if tab == "Scrape Data":
    st.header("Scrape Data dari Google Play Store atau Twitter")
    produk = st.selectbox("Pilih Produk", ["ChatGPT", "Google Assistant", "Siri"])
    if st.button("Scrape Sekarang (1000 Data)"):
        if produk == "ChatGPT":
            app_id = "com.openai.chatgpt"
            df = scrape_playstore(app_id)
        elif produk == "Google Assistant":
            app_id = "com.google.android.apps.googleassistant"
            df = scrape_playstore(app_id)
        elif produk == "Siri":
            df = scrape_twitter("Siri")

        df = df[['username', 'review']]
        st.success("Scraping selesai!")
        st.write(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"{produk.lower()}_reviews.csv", "text/csv")

# =================== UPLOAD CSV BIASA ===================
elif tab == "Upload CSV":
    st.header("Upload Dataset CSV")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        review_col = None
        for col in df.columns:
            if col.lower() in ['review', 'ulasan', 'text', 'content']:
                review_col = col
                break

        if review_col is None:
            st.error("Kolom review tidak ditemukan di file.")
            st.stop()

        df['review'] = df[review_col].astype(str).str.strip()

        if (df['review'].str.len() == 0).any():
            st.warning("⚠️ Beberapa review kosong dan akan diberi label 'neutral'.")

        if "clean_text" not in df.columns:
            df['clean_text'] = df['review'].apply(lambda x: preprocessing(x) if x.strip() else "")

        if "label" not in df.columns:
            df['label'] = df.apply(lambda row: auto_label(row['clean_text']) if row['clean_text'] else "neutral", axis=1)

        if "actual_sentiment" not in df.columns:
            df['actual_sentiment'] = df['label']

        st.subheader("Data Setelah Diproses")
        st.dataframe(df[['username', 'review', 'clean_text', 'label', 'actual_sentiment']])

        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_text'], df['actual_sentiment'], test_size=0.2, random_state=42
        )
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative", "neutral"])

        st.session_state['hasil_produk'][uploaded_file.name] = {
            'df': df,
            'acc': acc,
            'cm': cm
        }

        tabs = st.tabs(["📊 Evaluasi Model", "🧩 Confusion Matrix", "📋 Classification Report", "☁️ Wordcloud", "📈 Distribusi Sentimen", "🧾 Ringkasan"])
        with tabs[0]:
            st.write(f"**Akurasi:** {acc * 100:.2f}%")
            st.write(f"**Precision:** {prec * 100:.2f}%")
            st.write(f"**Recall:** {rec * 100:.2f}%")
            st.write(f"**F1-Score:** {f1 * 100:.2f}%")


            st.dataframe(pd.DataFrame(cm,
                index=["Actual Pos", "Actual Neg", "Actual Neu"],
                columns=["Pred Pos", "Pred Neg", "Pred Neu"]
            ))

        with tabs[1]:
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                        xticklabels=["positive", "negative", "neutral"],
                        yticklabels=["positive", "negative", "neutral"],
                        ax=ax_cm)
            ax_cm.set_xlabel('Predicted label')
            ax_cm.set_ylabel('True label')
            st.pyplot(fig_cm)
        with tabs[2]:
            st.code(classification_report(y_test, y_pred, digits=2), language='text')
        with tabs[3]:
            all_text = " ".join(df['clean_text'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        with tabs[4]:
            st.bar_chart(df['label'].value_counts())
        with tabs[5]:
            total_data = len(df)
            jumlah_positif = (df['label'] == 'positive').sum()
            jumlah_negatif = (df['label'] == 'negative').sum()
            jumlah_netral = (df['label'] == 'neutral').sum()
            st.markdown(f"""
            ### Ringkasan
            **Jumlah data setelah preprocessing:** {total_data}  
            **Jumlah data positif:** {jumlah_positif}  
            **Jumlah data negatif:** {jumlah_negatif}  
            **Jumlah data netral:** {jumlah_netral}
            """)

# =================== UPLOAD KHUSUS SIRI ===================
elif tab == "Upload Siri":
    st.header("Upload Dataset Siri (khusus kolom id dan review)")

    uploaded_file = st.file_uploader("Upload file CSV Siri", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # === 🔧 Perbaiki nama kolom jika terdeteksi aneh (misalnya: 'review;;;;;') ===
        df.columns = [col.split(';')[0].strip().lower() for col in df.columns]

        # Validasi kolom id dan review
        if not {'id', 'review'}.issubset(df.columns):
            st.error("File harus memiliki kolom: id dan review.")
            st.stop()

        df['review'] = df['review'].astype(str).str.strip()
        df['id'] = df['id'].astype(str)  # Menyelesaikan warning icon di kolom 'id'
        df['clean_text'] = df['review'].apply(lambda x: preprocessing(x) if x.strip() else "")
        df['label'] = df.apply(lambda row: auto_label(row['clean_text']) if row['clean_text'] else "neutral", axis=1)
        df['actual_sentiment'] = df['label']

        st.subheader("Data Siri Setelah Diproses")
        st.dataframe(df[['id', 'review', 'clean_text', 'label']])

        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_text'], df['actual_sentiment'], test_size=0.2, random_state=42
        )
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative", "neutral"])

        # Simpan hasil Siri ke session state untuk perbandingan
        st.session_state['hasil_produk']['Siri'] = {
            'df': df,
            'acc': acc,
            'cm': cm
        }

        tabs = st.tabs(["📊 Evaluasi Model", "🧩 Confusion Matrix", "📋 Classification Report", "☁️ Wordcloud", "📈 Distribusi Sentimen", "🧾 Ringkasan"])
        with tabs[0]:
            st.write(f"**Akurasi:** {acc * 100:.2f}%")
            st.write(f"**Precision:** {prec * 100:.2f}%")
            st.write(f"**Recall:** {rec * 100:.2f}%")
            st.write(f"**F1-Score:** {f1 * 100:.2f}%")

            st.dataframe(pd.DataFrame(cm,
                index=["Actual Pos", "Actual Neg", "Actual Neu"],
                columns=["Pred Pos", "Pred Neg", "Pred Neu"]
            ))
        with tabs[1]:
            st.subheader("Confusion Matrix (Absolut)")
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                        xticklabels=["positive", "negative", "neutral"],
                        yticklabels=["positive", "negative", "neutral"],
                        ax=ax_cm)
            ax_cm.set_xlabel('Predicted label')
            ax_cm.set_ylabel('True label')
            st.pyplot(fig_cm)

            st.subheader("Confusion Matrix (Ternormalisasi)")
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fig_cm_norm, ax_norm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=["positive", "negative", "neutral"],
                yticklabels=["positive", "negative", "neutral"],
                ax=ax_norm)
            ax_norm.set_xlabel('Predicted label')
            ax_norm.set_ylabel('True label')
            st.pyplot(fig_cm_norm)

        with tabs[2]:
            st.code(classification_report(y_test, y_pred, digits=2), language='text')
        with tabs[3]:
            all_text = " ".join(df['clean_text'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        with tabs[4]:
            st.bar_chart(df['label'].value_counts())
        with tabs[5]:
            total_data = len(df)
            jumlah_positif = (df['label'] == 'positive').sum()
            jumlah_negatif = (df['label'] == 'negative').sum()
            jumlah_netral = (df['label'] == 'neutral').sum()
            st.markdown(f"""
            ### Ringkasan
            **Jumlah data setelah preprocessing:** {total_data}  
            **Jumlah data positif:** {jumlah_positif}  
            **Jumlah data negatif:** {jumlah_negatif}  
            **Jumlah data netral:** {jumlah_netral}
            """)


# =================== PERBANDINGAN PRODUK ===================
elif tab == "Perbandingan Produk":
    st.header("📊 Perbandingan Hasil Sentimen Antar Produk")

    if not st.session_state['hasil_produk']:
        st.warning("Belum ada data yang dianalisis. Silakan unggah atau scrape data terlebih dahulu.")
        st.stop()

    tabs = st.tabs(["📋 Ringkasan Per Produk", "📈 Grafik Perbandingan"])
    
    # Tab 1: Ringkasan per Produk
    with tabs[0]:
        summary_data = []
        for produk, hasil in st.session_state['hasil_produk'].items():
            df = hasil['df']
            pos = (df['label'] == 'positive').sum()
            neg = (df['label'] == 'negative').sum()
            neu = (df['label'] == 'neutral').sum()
            total = len(df)
            summary_data.append({
                'Produk': produk,
                'Total Ulasan': total,
                'Positif': pos,
                'Negatif': neg,
                'Netral': neu
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)

        # 🔍 Ringkasan Pemenang per Sentimen
        st.subheader("🏆 Produk dengan Sentimen Terbanyak")
        max_pos = summary_df.loc[summary_df['Positif'].idxmax()]
        max_neg = summary_df.loc[summary_df['Negatif'].idxmax()]
        max_neu = summary_df.loc[summary_df['Netral'].idxmax()]

        st.markdown(f"""
        - **Positif terbanyak:** {max_pos['Produk']} ({max_pos['Positif']} ulasan)
        - **Negatif terbanyak:** {max_neg['Produk']} ({max_neg['Negatif']} ulasan)
        - **Netral terbanyak:** {max_neu['Produk']} ({max_neu['Netral']} ulasan)
        """)
    
    # Tab 2: Visualisasi
    with tabs[1]:
        st.subheader("📊 Perbandingan Sentimen per Produk")
        chart_df = summary_df.set_index('Produk')[['Positif', 'Negatif', 'Netral']]
        st.bar_chart(chart_df)
