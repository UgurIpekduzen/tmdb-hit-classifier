"""Streamlit demo UI — TMDB Hit Classifier."""

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
    api_url = st.text_input("API URL", value="http://localhost:8000")

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


def _binary(label: str, key: str) -> int:
    return int(st.checkbox(label, key=key))


# ── Tahmin ────────────────────────────────────────────────────────────────────

st.title("🎬 Film Hit Tahmini")
st.caption("26 özelliği girerek modelin box-office hit olasılığını hesaplayın.")

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Bütçe & Prodüksiyon")
        budget_adjusted       = st.number_input("Bütçe (USD, 2010 bazlı)", min_value=0.0, value=30_000_000.0, step=1_000_000.0, format="%.0f")
        runtime               = st.slider("Süre (dakika)", 0, 600, 110)
        num_companies         = st.number_input("Yapım şirketi sayısı", min_value=0, value=2)
        is_major_studio       = _binary("Major stüdyo (Warner, Universal, Disney, Sony, Paramount, Fox)", "is_major_studio")
        is_us_production      = _binary("ABD yapımı", "is_us_production")

        st.subheader("Pazarlama")
        has_homepage          = _binary("Resmi web sitesi var", "has_homepage")

    with col2:
        st.subheader("Tür & Franchise")
        num_genres            = st.number_input("Tür sayısı", min_value=0, max_value=20, value=2)
        is_franchise          = _binary("Franchise / seri film", "is_franchise")
        monthly_franchise_density = st.slider("Aylık franchise yoğunluğu (0–1)", 0.0, 1.0, 0.3, step=0.01)

        st.subheader("Takvim")
        is_summer             = _binary("Yaz vizyonu (Haz–Ağu)", "is_summer")
        is_holiday            = _binary("Tatil vizyonu (Kas–Ara)", "is_holiday")

        st.subheader("Yönetmen & Oyuncu")
        director_film_count   = st.number_input("Yönetmenin önceki film sayısı", min_value=0, value=3)
        director_collab_count = st.number_input("Yönetmenin işbirliği sayısı", min_value=0, value=10)
        actor_hist_roi        = st.number_input("Başrol oyuncu geçmiş ROI (log1p)", min_value=0.0, value=1.0, step=0.1, format="%.2f")
        company_hist_roi      = st.number_input("Yapım şirketi geçmiş ROI (log1p)", min_value=0.0, value=1.0, step=0.1, format="%.2f")

    with col3:
        st.subheader("Türler")
        genre_Action          = _binary("Action",         "gA")
        genre_Adventure       = _binary("Adventure",      "gAd")
        genre_Animation       = _binary("Animation",      "gAn")
        genre_Documentary     = _binary("Documentary",    "gD")
        genre_Drama           = _binary("Drama",          "gDr")
        genre_Family          = _binary("Family",         "gF")
        genre_Fantasy         = _binary("Fantasy",        "gFa")
        genre_Horror          = _binary("Horror",         "gH")
        genre_sf              = _binary("Science Fiction","gSF")

        st.subheader("Anahtar Kelimeler")
        kw_independent_film   = _binary("Bağımsız film",  "kInd")
        kw_sequel             = _binary("Devam filmi",    "kSeq")

    submitted = st.form_submit_button("Tahmin Et", use_container_width=True, type="primary")

if submitted:
    if not _check_api():
        st.stop()
    payload = {
        "budget_adjusted":           budget_adjusted,
        "runtime":                   runtime,
        "num_companies":             num_companies,
        "is_major_studio":           is_major_studio,
        "is_us_production":          is_us_production,
        "has_homepage":              has_homepage,
        "num_genres":                num_genres,
        "is_franchise":              is_franchise,
        "is_summer":                 is_summer,
        "is_holiday":                is_holiday,
        "director_film_count":       director_film_count,
        "director_collab_count":     director_collab_count,
        "actor_hist_roi":            actor_hist_roi,
        "company_hist_roi":          company_hist_roi,
        "monthly_franchise_density": monthly_franchise_density,
        "genre_Action":              genre_Action,
        "genre_Adventure":           genre_Adventure,
        "genre_Animation":           genre_Animation,
        "genre_Documentary":         genre_Documentary,
        "genre_Drama":               genre_Drama,
        "genre_Family":              genre_Family,
        "genre_Fantasy":             genre_Fantasy,
        "genre_Horror":              genre_Horror,
        "genre_Science Fiction":     genre_sf,
        "kw_independent_film":       kw_independent_film,
        "kw_sequel":                 kw_sequel,
    }
    try:
        resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        st.error(f"İstek başarısız: {e}")
        st.stop()

    proba = result["probability"]
    hit   = result["hit"]

    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Hit Olasılığı", f"%{proba*100:.1f}")
    r2.metric("Karar", "✅ HIT" if hit else "❌ MISS",
              delta=f"eşik: {result['threshold']}")
    r3.metric("Model Versiyonu", f"v{result['model_version']}")

    st.progress(proba, text=f"{proba*100:.1f}%")
