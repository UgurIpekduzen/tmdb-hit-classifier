# TMDB Hit Classifier

Box-office hit tahmini için uçtan uca makine öğrenmesi projesi. TMDB 5000 film veri seti kullanılarak bir filmin vizyona girmeden önce **box-office hit olup olmayacağını** tahmin eder.

> **Hit tanımı:** CPI-düzeltmeli gelir (2010 bazlı) / prodüksiyon bütçesi ≥ 2×

---

## Sonuçlar

| Metrik | Değer |
|--------|-------|
| **F2** | **0.867** |
| **PR-AUC** | **0.921** |
| Recall | 0.882 |
| Precision | 0.811 |
| ROC-AUC | 0.916 |

> Model: `voting_top2` (LightGBM + CatBoost soft voting)
> Test seti: ≥2016 vizyonu, n=72 film — temporal split ile değerlendirildi.

---

## Proje Yapısı

```
.
├── notebooks/
│   ├── preprocess.ipynb  # Ham veri temizleme ve feature engineering
│   ├── eda.ipynb         # Keşifsel veri analizi
│   ├── tuning.ipynb      # Baseline eğitimi + Optuna hiperparametre tuning
│   └── modeling.ipynb    # Değerlendirme, hibrit modeller, SHAP, feature deneyleri (ana notebook)
│
├── src/                  # Yeniden kullanılabilir modüller
│   ├── modeling.py       # Model eğitimi ve SHAP analizi
│   ├── tuning.py         # Optuna hiperparametre optimizasyonu
│   ├── evaluation.py     # Bootstrap CI, champion tracking, MLflow kayıt
│   ├── ensemble.py       # Voting / Stacking / Blending
│   └── validation.py     # Temporal split, leakage kontrolü
│
├── app/                  # Servis katmanı
│   ├── api.py            # FastAPI inference endpoint
│   ├── monitoring.py     # Feature drift (PSI + KS testi)
│   └── ui.py             # Streamlit demo arayüzü
│
├── data/
│   └── tmdb_model.csv    # Model için hazırlanmış veri seti
│
├── Dockerfile
├── docker-compose.yaml   # MLflow + API servisleri
└── requirements.txt
```

---

## Hızlı Başlangıç

### Gereksinimler

- Docker & Docker Compose
- Python 3.10+

### 1. Servisleri başlat

```bash
make up-all          # MLflow (port 5000) + API (port 8000)
```

### 2. Demo UI'ı çalıştır

```bash
make run-ui          # http://localhost:8501
```

### 3. Notebookları çalıştır (opsiyonel)

```bash
# Sırayla çalıştırın:
jupyter notebook notebooks/preprocess.ipynb
jupyter notebook notebooks/eda.ipynb
jupyter notebook notebooks/tuning.ipynb    # tuned_models'ı notebooks/artifacts/'a kaydeder
jupyter notebook notebooks/modeling.ipynb  # tuning.ipynb'in çıktısını yükler, devam eder
```

---

## API Endpoint'leri

Servis ayağa kalktıktan sonra Swagger UI: `http://localhost:8000/docs`

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/health` | Servis sağlık kontrolü |
| GET | `/info` | Model versiyonu ve konfigürasyon |
| POST | `/predict` | Tek film tahmini |
| POST | `/predict/batch` | Toplu tahmin (maks. 500) |
| POST | `/drift` | Feature drift raporu (PSI + KS) |

**Örnek istek:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "budget_adjusted": 150000000,
    "runtime": 130,
    "is_franchise": 1,
    "is_major_studio": 1,
    "is_summer": 1,
    "genre_Action": 1,
    "num_genres": 2,
    "director_film_count": 5,
    "actor_hist_roi": 2.1,
    "company_hist_roi": 1.8,
    "num_companies": 2,
    "is_us_production": 1,
    "has_homepage": 1,
    "is_holiday": 0,
    "director_collab_count": 20,
    "monthly_franchise_density": 0.4,
    "genre_Adventure": 1,
    "genre_Animation": 0, "genre_Documentary": 0, "genre_Drama": 0,
    "genre_Family": 0, "genre_Fantasy": 0, "genre_Horror": 0,
    "genre_Science Fiction": 0,
    "kw_independent_film": 0, "kw_sequel": 1
  }'
```

---

## Metodoloji

### Temporal Split (Veri Sızıntısı Önleme)

| Bölüm | Yıllar | Film Sayısı |
|-------|--------|-------------|
| Train | ≤ 2013 | 2.899 (%90) |
| Val | 2014–2015 | 255 (%8) |
| Test | ≥ 2016 | 72 (%2) |

`actor_hist_roi`, `director_hist_roi`, `company_hist_roi` feature'ları her film için yalnızca **vizyona giriş öncesindeki** filmlerden hesaplanır — temporal leakage yoktur.

### Feature Engineering (26 Özellik)

| Özellik | Açıklama |
|---------|----------|
| `budget_adjusted` | Filmin prodüksiyon bütçesi. Enflasyona göre düzeltilmiş — 1990'daki 10 milyon dolar ile 2010'daki 10 milyon doları eşit değerde sayar. |
| `runtime` | Filmin süresi (dakika cinsinden). |
| `num_companies` | Filmi birlikte yapan şirket sayısı. Büyük projeler genellikle birden fazla şirketin ortaklığıyla üretilir. |
| `is_major_studio` | Filmin Warner Bros., Universal, Disney, Sony, Paramount veya Fox gibi büyük bir stüdyo tarafından yapılıp yapılmadığı. |
| `is_us_production` | Filmin ABD yapımı olup olmadığı. |
| `has_homepage` | Filmin kendine ait resmi bir web sitesinin olup olmadığı. Pazarlama bütçesinin göstergesi olarak kullanılır. |
| `num_genres` | Filme atanan tür sayısı (örn. hem aksiyon hem macera = 2). |
| `is_franchise` | Filmin bir serinin parçası olup olmadığı (örn. Marvel, Fast & Furious, Harry Potter). |
| `monthly_franchise_density` | Filmin vizyona girdiği ay, rakip franchise filmlerinin yoğunluğu. Yüksekse o ay çok franchise filmi var demektir. |
| `is_summer` | Filmin yaz döneminde (Haziran–Ağustos) vizyona girip girmediği. Yaz, gişe için en rekabetçi ve kârlı dönemdir. |
| `is_holiday` | Filmin tatil döneminde (Kasım–Aralık) vizyona girip girmediği. Yılbaşı sezonu da gişede kritik öneme sahiptir. |
| `director_film_count` | Yönetmenin bu filmden önce çektiği uzun metraj sayısı. Deneyimli yönetmen mi, ilk filmi mi? |
| `director_collab_count` | Yönetmenin kariyeri boyunca birlikte çalıştığı farklı oyuncu ve ekip üyesi sayısı. Geniş bir ağa sahip mi? |
| `actor_hist_roi` | Başrol oyuncusunun önceki filmlerinin ortalama gişe getirisi. Bilet satışlarını artıran bir isim mi? |
| `company_hist_roi` | Yapım şirketinin geçmişteki filmlerinin ortalama gişe getirisi. Hit üretme sicili olan bir şirket mi? |
| `genre_Action` | Film aksiyon türünde mi? |
| `genre_Adventure` | Film macera türünde mi? |
| `genre_Animation` | Film animasyon mu? |
| `genre_Documentary` | Film belgesel mi? |
| `genre_Drama` | Film drama türünde mi? |
| `genre_Family` | Film aile izleyicisine yönelik mi? |
| `genre_Fantasy` | Film fantezi türünde mi? |
| `genre_Horror` | Film korku türünde mi? |
| `genre_Science Fiction` | Film bilim kurgu türünde mi? |
| `kw_independent_film` | Film bağımsız yapım olarak etiketlenmiş mi? Bağımsız filmler genellikle daha düşük bütçeyle üretilir. |
| `kw_sequel` | Film bir önceki filmin devamı mı? Devam filmleri yerleşik bir izleyici kitlesine sahip olur. |

> `actor_hist_roi` ve `company_hist_roi` hesaplanırken yalnızca o filmin **vizyona giriş tarihinden önceki** veriler kullanılır. Böylece model, gelecekteki bilgilere erişmemiş olur.

### Model Seçimi

5 model Optuna ile 100 trial hiperparametre optimizasyonuna tabi tutuldu:

| Model | Val F2 |
|-------|--------|
| **voting_top2** (LightGBM + CatBoost) | **0.847** |
| CatBoost | 0.722 |
| LightGBM | 0.717 |
| XGBoost | 0.718 |
| Logistic Regression | 0.717 |
| Random Forest | 0.711 |

F2 skoru, yanlış negatif (kaçırılan hit) maliyetinin yanlış pozitiften daha yüksek olduğunu yansıtır.

### MLOps

- **MLflow** — deney takibi, model registry, artifact yönetimi
- **Temporal CV** — 8 yıl train / 2 yıl val sliding window ile overfitting tanısı
- **Bootstrap CI** — n=1000 bootstrap ile %95 güven aralığı
- **Drift Monitoring** — PSI + KS two-sample test ile feature drift tespiti

---

## Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| ML | LightGBM, CatBoost, XGBoost, scikit-learn |
| Tuning | Optuna |
| Tracking | MLflow |
| Serving | FastAPI + Uvicorn |
| UI | Streamlit |
| Infra | Docker Compose |
| Veri | pandas, numpy, scipy |

---

## Feature Version Geçmişi

Detaylı versiyon geçmişi için [README — Feature Version History](README.md#feature-version-history) bölümüne bakın.

<details>
<summary>Versiyon özeti</summary>

| Versiyon | Özellik Sayısı | Val PR-AUC | Temel Değişiklik |
|----------|---------------|------------|-----------------|
| v14 (final) | 26 | 0.915 | Train cutoff 2009→2013, CPI hit tanımı |
| v13 | 26 | — | `is_english`, `kw_3d` drop |
| v12_simplified | 26 | — | 4 düşük sinyalli feature drop |
| v9_budget_adjusted | 30 | 0.848 | CPI-düzeltmeli bütçe |
| v6_shap_cleaned | 27 | 0.813 | SHAP multikolinearite temizliği |
| v4_univariate_selected | 30 | 0.815 | Bonferroni feature seçimi |

</details>

---

## Feature Version History

Her versiyonda yapılan değişiklikler, gerekçeler ve metrik etkileri.

---

### v6_shap_cleaned — ADVBK-61 *(Nisan 2026)*

**Özellik sayısı:** 28 → 27

**Değişiklikler:**
- Çıkarıldı: `relative_budget_score` — SHAP korelasyon analizi: `budget` ile r=0.781, zıt SHAP yönleri (budget ↑hit, relative_budget_score ↓miss) multikolinearite belirtisi

**Gerekçe:** Aynı bilgiyi zıt yönlerle taşıyan iki feature modelde gürültü yaratır; `budget` tek başına yeterli.

| Metrik | v5 (önceki) | v6 | Δ |
|--------|------------|-----|---|
| PR-AUC | 0.8134 | 0.8127 | −0.0007 |
| ROC-AUC | 0.8725 | 0.8693 | −0.0032 |
| F1 | 0.7037 | 0.7139 | +0.0102 |
| Precision | 0.7355 | 0.6848 | −0.0507 |
| Recall | 0.6746 | 0.7456 | +0.0710 |

**Final model:** voting_top2 (logreg + rf)

**Karar:** PR-AUC değişimi gürültü seviyesinde (−0.0007) — drop onaylandı.

---

### v5 *(implicit)* — ADVBK-60 *(Nisan 2026)*

**Özellik sayısı:** 30 → 28

**Değişiklikler:**
- Çıkarıldı: `kw_aftercreditsstinger` (#24, mean_abs=0.0068) — post-release leakage
- Çıkarıldı: `kw_duringcreditsstinger` (#28, mean_abs=0.0012) — post-release leakage

| Metrik | v4 (önceki) | v5 | Δ |
|--------|------------|-----|---|
| PR-AUC | 0.8153 | 0.8134 | −0.0019 |
| ROC-AUC | 0.8736 | 0.8725 | −0.0011 |
| F1 | 0.7163 | 0.7037 | −0.0126 |
| Precision | 0.6701 | 0.7355 | +0.0654 |
| Recall | 0.7692 | 0.6746 | −0.0946 |

**Final model:** blending

---

### v4_univariate_selected — ADVBK-58 + ADVBK-59 *(Nisan 2026)*

**Özellik sayısı:** ~49 → 30

**Yöntem:** Chi-squared (binary) + Mann-Whitney U (continuous), Bonferroni düzeltmesi (p > 0.0010 → drop)

| Metrik | v4 |
|--------|-----|
| PR-AUC | 0.8153 |
| ROC-AUC | 0.8736 |
| F1 | 0.7163 |

**Final model:** voting_top3

---

### v3_new_features — ADVBK-56 *(Nisan 2026)*

**Özellik sayısı:** 41 → 45

**Eklendi:**
- `budget_per_minute` (median $239K/dk)
- `is_franchise` (hit rate %56 vs %32)
- `director_hist_roi` — temporal loop, ROI @p99=72x cap
- `actor_hist_roi` — temporal loop, aynı metodoloji

---

### v1 / v2 — ADVBK-2..5 *(Mart 2026 ve öncesi)*

İlk LGBM overfitting tespiti ve tuning çalışmaları.

---

### Özet Tablosu

| Versiyon | Özellik Sayısı | PR-AUC | Final Model | Temel Değişiklik |
|----------|---------------|--------|-------------|-----------------|
| v6_shap_cleaned | 27 | 0.8127 | voting_top2 | relative_budget_score drop |
| v5 *(implicit)* | 28 | 0.8134 | blending | kw_postcredit ×2 drop (leakage) |
| v4_univariate_selected | 30 | 0.8153 | voting_top3 | Bonferroni + multikolinearite seçimi |
| v3_new_features | 45 | — | — | Temporal feature engineering |
| v1/v2 | ~41 | — | — | LGBM baseline + tuning |
