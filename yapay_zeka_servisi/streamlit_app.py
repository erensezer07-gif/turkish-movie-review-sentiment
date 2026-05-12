"""Streamlit UI for ensemble.

Çalıştırma:
    streamlit run yapay_zeka_servisi/streamlit_app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    px = None

from yapay_zeka_servisi import ensemble_core
from yapay_zeka_servisi.ensemble_core import (
    ensemble_single, ensemble_batch,
    tfidf_err, bert_err, bert_meta, set_stats_hook,
)

# ---------------------------------------------------------------------
# STATS (Streamlit'e özgü; ensemble_core'a hook ile bağlanır)
# ---------------------------------------------------------------------
_STATS_TEMPLATE = {
    "total": 0, "guardrail": 0, "neutralrule": 0, "uncertainneutral": 0,
    "tfidf": 0, "bert": 0, "ensemble": 0, "error": 0,
}


def init_stats():
    if "analysis_stats" not in st.session_state or not isinstance(st.session_state.analysis_stats, dict):
        st.session_state.analysis_stats = dict(_STATS_TEMPLATE)
    else:
        for k, v in _STATS_TEMPLATE.items():
            st.session_state.analysis_stats.setdefault(k, v)


def reset_stats():
    st.session_state.analysis_stats = dict(_STATS_TEMPLATE)


def _stats_callback(key: str, n: int = 1):
    init_stats()
    if key in st.session_state.analysis_stats:
        st.session_state.analysis_stats[key] += int(n)


set_stats_hook(_stats_callback)

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="AI Sinema Eleştirmeni (Pro)", page_icon="🎬", layout="centered")
init_stats()
st.title("🎬 AI Sinema Eleştirmeni (Ensemble Pro)")
st.caption("3-Sınıf (Neg/Neu/Pos) | BERT ağırlıklı Ensemble | Gelişmiş Belirsizlik Yönetimi")
st.divider()

if tfidf_err:
    st.warning(f"⚠️ {tfidf_err} (BERT-only modda devam edilebilir)")
if bert_err:
    st.error(f"❌ {bert_err}")
    st.stop()

with st.sidebar:
    st.header("⚙️ Ayarlar")
    st.success(f"Cihaz: {bert_meta['device'].upper()}")

    use_guardrail = st.toggle("Guardrails aktif", value=True)
    guard_cutoff = st.slider("Guardrail fuzzy cutoff", 0.70, 0.95, 0.85, 0.01)

    st.divider()
    st.subheader("Ensemble Ağırlıkları")
    tfidf_weight = st.slider("TF-IDF Ağırlığı", 0.0, 1.0, 0.30, 0.05)
    bert_weight = 1.0 - tfidf_weight
    st.caption(f"BERT Ağırlığı: {bert_weight:.2f}")

    st.divider()
    st.subheader("Belirsizlik Yönetimi")
    uncertain_to_neutral_on = st.toggle("Belirsizse NÖTR'e çek", value=True)
    conf_threshold = st.slider("Güven Eşiği (Max < x)", 0.30, 0.80, 0.50, 0.05)
    margin_threshold = st.slider("Fark Eşiği (Margin < x)", 0.05, 0.30, 0.15, 0.01)
    min_neutral_prob = st.slider("Min Nötr Olasılığı", 0.10, 0.50, 0.25, 0.05)

    st.divider()
    neutral_on = st.toggle("Nötr Kuralı (Regex/Phrase)", value=False)
    use_neutral_band = st.toggle("BERT Bandı (Opsiyonel)", value=False)
    bert_neutral_low = 0.45
    bert_neutral_high = 0.65

    st.divider()
    bert_batch_size = st.selectbox("Batch size", [8, 16, 32, 64], index=1)
    bert_max_len = st.selectbox("Max length", [96, 128, 160, 192, 256], index=3)
    debug_mode = st.checkbox("Debug Modu (Tekli)", value=False)

    if st.button("🧹 İstatistikleri Sıfırla"):
        reset_stats()
        st.success("Sıfırlandı.")

    st.divider()
    st.subheader("📊 İstatistikler")
    stats = st.session_state.analysis_stats
    if stats["total"] > 0:
        st.metric("Toplam", stats["total"])
        st.write(f"🛡️ Guard: {stats['guardrail']}")
        st.write(f"🟦 TF-IDF: {stats['tfidf']}")
        st.write(f"🟪 BERT: {stats['bert']}")
        st.write(f"🤷 Uncertain→Nötr: {stats['uncertainneutral']}")
        st.write(f"🧩 Ensemble: {stats['ensemble']}")
        st.write(f"❌ Error: {stats['error']}")

tab1, tab2 = st.tabs(["💬 Tekli Analiz", "📂 Toplu Analiz"])

with tab1:
    if "single_text" not in st.session_state:
        st.session_state["single_text"] = ""

    def clear_single_text():
        st.session_state["single_text"] = ""

    c1, c2 = st.columns([1, 4])
    yorum = st.text_area("Yorum:", height=110, key="single_text")

    if c1.button("ANALİZ ET 🚀", key="btn_single"):
        if len(str(yorum).strip()) < 2:
            st.warning("Yorum girin.")
        else:
            with st.spinner("Analiz ediliyor..."):
                label, conf, src, dbg = ensemble_single(
                    yorum, use_guardrail, guard_cutoff, neutral_on, use_neutral_band,
                    bert_neutral_low, bert_neutral_high, tfidf_weight, bert_weight,
                    bert_batch_size, bert_max_len, debug_mode, uncertain_to_neutral_on,
                    conf_threshold, margin_threshold, min_neutral_prob,
                )
            st.subheader("Sonuç")
            color = "green" if label == "OLUMLU" else "red" if label == "OLUMSUZ" else "orange"
            st.markdown(f"### :{color}[{label}]")
            st.progress(int(conf * 100))
            st.info(f"Güven: %{conf*100:.1f} | Kaynak: {src}")
            with st.expander("Detaylar & Debug"):
                st.write(dbg)

    c2.button("TEMİZLE", on_click=clear_single_text)

with tab2:
    st.info("CSV/Excel yükleyin. 'yorum' sütunu aranır.")
    up = st.file_uploader("Dosya Yükle", type=["csv", "xlsx"], key="file_uploader")
    if up:
        try:
            df = pd.read_csv(up, on_bad_lines="skip") if up.name.endswith(".csv") else pd.read_excel(up)
            cols = [c for c in df.columns if any(x in str(c).lower() for x in ["yorum", "text", "review"])]
            target_col = st.selectbox(
                "Sütun Seç:",
                df.columns.tolist(),
                index=df.columns.tolist().index(cols[0]) if cols else 0,
            )

            if st.button("Başlat"):
                texts = df[target_col].fillna("").astype(str).tolist()
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(p):
                    progress_bar.progress(int(p * 100))
                    status_text.text(f"%{int(p*100)}")

                with st.spinner("Çalışıyor..."):
                    labels, confs, srcs = ensemble_batch(
                        texts, use_guardrail, guard_cutoff, neutral_on, use_neutral_band,
                        bert_neutral_low, bert_neutral_high, tfidf_weight, bert_weight,
                        bert_batch_size, bert_max_len, update_progress,
                        uncertain_to_neutral_on, conf_threshold, margin_threshold, min_neutral_prob,
                    )

                progress_bar.empty()
                status_text.empty()

                df["AI_Karari"] = labels
                df["Guven_%"] = np.round(np.array(confs) * 100, 1)
                df["Kaynak"] = srcs
                st.success("Bitti!")

                counts = df["AI_Karari"].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("Olumlu", counts.get("OLUMLU", 0))
                c2.metric("Olumsuz", counts.get("OLUMSUZ", 0))
                c3.metric("Nötr", counts.get("NÖTR", 0))

                st.dataframe(df.head(20))
                csv = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("İndir (CSV)", csv, "sonuc.csv", "text/csv")

                if px:
                    fig = px.pie(values=counts.values, names=counts.index, title="Dağılım")
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Hata: {e}")
