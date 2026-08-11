"""Streamlit demo arayüzü — TMDB Hit Classifier."""

import os

import requests
import streamlit as st

st.set_page_config(
    page_title="TMDB Hit Classifier",
    page_icon="🎬",
    layout="wide",
)

# ── Sidebar — bağlantı ayarları ───────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_url = st.text_input(
        "API URL", value=os.getenv("API_URL", "http://localhost:8000")
    )

    if st.button("Bağlantıyı Test Et", use_container_width=True):
        try:
            h = requests.get(f"{api_url}/health", timeout=3)
            i = requests.get(f"{api_url}/info",   timeout=3)
            if h.ok and i.ok:
                st.session_state["api_ok"]   = True
                st.session_state["api_info"] = i.json()
            else:
                st.session_state["api_ok"] = False
        except Exception:
            st.session_state["api_ok"] = False

    if st.session_state.get("api_ok"):
        info = st.session_state["api_info"]
        st.success("API bağlı")
        st.caption(
            f"Model: **{info['registry']}** v{info['version']} ({info['stage']})\n\n"
            f"Threshold: {info['threshold']} | {info['n_features']} özellik"
        )
    elif "api_ok" in st.session_state:
        st.error("API'ye ulaşılamıyor")


# ── Yardımcı ─────────────────────────────────────────────────────────────────

def _check_api() -> bool:
    if not st.session_state.get("api_ok"):
        st.warning("Önce sol panelden API bağlantısını test edin.")
        return False
    return True


# ── Başlık ────────────────────────────────────────────────────────────────────

st.title("🎬 Film Hit Tahmini")
st.caption(
    "Film bilgilerini girin — yönetmen deneyimi, oyuncu geçmişi ve şirket itibarı "
    "otomatik hesaplanır."
)

# ── Form ──────────────────────────────────────────────────────────────────────

GENRES = [
    "Action", "Adventure", "Animation", "Documentary",
    "Drama", "Family", "Fantasy", "Horror", "Science Fiction",
]

with st.form("predict_raw_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Film Bilgileri")
        title   = st.text_input("Film adı", placeholder="örn. Inception")
        budget  = st.number_input(
            "Prodüksiyon bütçesi (milyon USD)",
            min_value=0.0, value=50.0, step=5.0, format="%.1f",
        )
        runtime = st.slider("Süre (dakika)", min_value=60, max_value=240, value=110)
        release_date = st.date_input("Vizyon tarihi")
        genres  = st.multiselect("Türler", options=GENRES, default=["Action"])

        st.subheader("Franchise & Platform")
        belongs_to_collection = st.checkbox("Franchise / seri parçası")
        has_homepage = st.checkbox("Resmi web sitesi var")

    with col2:
        st.subheader("Ekip")
        director = st.text_input("Yönetmen", placeholder="örn. Christopher Nolan")

        st.markdown("**Başrol oyuncuları** (en fazla 3 isim)")
        actor1 = st.text_input("1. Oyuncu", placeholder="örn. Leonardo DiCaprio", label_visibility="collapsed")
        actor2 = st.text_input("2. Oyuncu", placeholder="örn. Joseph Gordon-Levitt", label_visibility="collapsed")
        actor3 = st.text_input("3. Oyuncu", placeholder="örn. Elliot Page", label_visibility="collapsed")

        st.subheader("Yapım")
        companies_raw = st.text_input(
            "Yapım şirketleri (virgülle ayırın)",
            placeholder="örn. Warner Bros., Legendary Pictures",
        )
        is_us = st.checkbox("ABD yapımı", value=True)
        is_sequel = st.checkbox("Devam filmi (sequel)")
        is_independent = st.checkbox("Bağımsız yapım (independent film)")

    submitted = st.form_submit_button(
        "🎯 Tahmin Et", use_container_width=True, type="primary"
    )

# ── Tahmin ────────────────────────────────────────────────────────────────────

if submitted:
    if not _check_api():
        st.stop()

    lead_actors = [a.strip() for a in [actor1, actor2, actor3] if a.strip()]
    companies   = [c.strip() for c in companies_raw.split(",") if c.strip()]
    keywords    = []
    if is_sequel:
        keywords.append("sequel")
    if is_independent:
        keywords.append("independent film")

    payload = {
        "title":                 title,
        "budget":                budget * 1_000_000,
        "runtime":               runtime,
        "release_date":          str(release_date),
        "genres":                genres,
        "production_companies":  companies,
        "production_countries":  ["US"] if is_us else [],
        "belongs_to_collection": belongs_to_collection,
        "homepage":              "https://placeholder.com" if has_homepage else None,
        "director":              director,
        "lead_actors":           lead_actors,
        "keywords":              keywords,
    }

    try:
        resp = requests.post(f"{api_url}/predict/raw", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        st.error(f"İstek başarısız: {e}")
        st.stop()

    proba = result["probability"]
    hit   = result["hit"]

    st.divider()

    r1, r2, r3 = st.columns(3)
    r1.metric("Hit Olasılığı", f"%{proba * 100:.1f}")
    r2.metric("Karar", "✅ HIT" if hit else "❌ MISS",
              delta=f"eşik: {result['threshold']}")
    r3.metric("Model Versiyonu", f"v{result['model_version']}")

    st.progress(proba, text=f"%{proba * 100:.1f} hit olasılığı")
