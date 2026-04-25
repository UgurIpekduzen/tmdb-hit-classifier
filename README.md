# tmdb-5000-dataset

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

**Karar:** PR-AUC değişimi gürültü seviyesinde (−0.0007) — drop onaylandı. F1/Recall artışı final model değişiminden (blending → voting_top2) kaynaklanıyor, drop'tan değil.

---

### v5 *(implicit)* — ADVBK-60 *(Nisan 2026)*

> Not: FEATURE_VERSION string'i güncellenmedi; `v4_univariate_selected` etiketi altında çalıştırıldı.

**Özellik sayısı:** 30 → 28

**Değişiklikler:**
- Çıkarıldı: `kw_aftercreditsstinger` (#24, mean_abs=0.0068) — post-release, kullanıcı film izledikten sonra etiketler
- Çıkarıldı: `kw_duringcreditsstinger` (#28, mean_abs=0.0012) — post-release, aynı gerekçe
- Leakage kontrolü whitelist tabanlı dinamik yapıya taşındı

**Gerekçe:** SHAP analizi bu iki keyword'ün zayıf sinyal + leakage riskini gösterdi.

| Metrik | v4 (önceki) | v5 | Δ |
|--------|------------|-----|---|
| PR-AUC | 0.8153 | 0.8134 | −0.0019 |
| ROC-AUC | 0.8736 | 0.8725 | −0.0011 |
| F1 | 0.7163 | 0.7037 | −0.0126 |
| Precision | 0.6701 | 0.7355 | +0.0654 |
| Recall | 0.7692 | 0.6746 | −0.0946 |

**Final model:** blending

**Karar:** PR-AUC düşüşü gürültü seviyesinde; leakage riski temizlendi — drop onaylandı.

---

### v4_univariate_selected — ADVBK-58 + ADVBK-59 *(Nisan 2026)*

**Özellik sayısı:** ~49 → 30

**Değişiklikler:**
- 18 feature Bonferroni eşiği dışı (p > 0.0010): `tagline_length`, `overview_length`, `num_countries`, `num_languages`, `genre_Romance`, `genre_Crime`, `genre_History`, `genre_Western`, `genre_Mystery`, `genre_Comedy`, `genre_War`, `genre_Foreign`, `genre_Thriller`, `genre_Music`, `kw_based_on_novel`, `kw_woman_director`, `kw_biography`, `kw_musical`
- Çıkarıldı: `budget_per_minute` — multikolinearite: `budget` ile r=0.969
- Eklendi: `relative_budget_score`, `num_languages`, `kw_aftercreditsstinger`, `kw_duringcreditsstinger` (SHAP'a bırakılan açık kararlar)
- BASE_FEATURES güncellendi: `budget_per_minute`, `num_countries`, `tagline_length`, `overview_length` çıkarıldı; `relative_budget_score` eklendi

**Yöntem:** Chi-squared (binary) + Mann-Whitney U (continuous), Bonferroni düzeltmesi

| Metrik | v3 (önceki) | v4 | Δ |
|--------|------------|-----|---|
| PR-AUC | — | 0.8153 | — |
| ROC-AUC | — | 0.8736 | — |
| F1 | — | 0.7163 | — |
| Precision | — | 0.6701 | — |
| Recall | — | 0.7692 | — |

**Final model:** voting_top3

**Karar:** İlk istatistiksel feature seçimi iterasyonu — sonraki SHAP analiziyle açık kararlar çözüldü.

---

### v3_new_features — ADVBK-56 *(Nisan 2026)*

**Özellik sayısı:** 41 → 45

**Değişiklikler:**
- Eklendi: `budget_per_minute` (median $239K/dk)
- Eklendi: `is_franchise` (hit rate %56 vs %32)
- Eklendi: `director_hist_roi` — temporal loop, ROI @p99=72x cap
- Eklendi: `actor_hist_roi` — temporal loop, aynı metodoloji

**Gerekçe:** Film bazında vizyona giriş öncesi bilinen, sızıntısız yeni feature'lar.

**Karar:** Feature engineering tamamlandı; univariate seçimine (v4) zemin hazırladı.

---

### v1 / v2 — ADVBK-2..5 *(Mart 2026 ve öncesi)*

İlk LGBM overfitting tespiti ve tuning çalışmaları. Detaylı metrik kaydı mevcut değil.

---

### Özet Tablosu

| Versiyon | Özellik Sayısı | PR-AUC | Final Model | Temel Değişiklik |
|----------|---------------|--------|-------------|-----------------|
| v6_shap_cleaned | 27 | 0.8127 | voting_top2 | relative_budget_score drop (SHAP r=0.781) |
| v5 *(implicit)* | 28 | 0.8134 | blending | kw_postcredit ×2 drop (leakage) |
| v4_univariate_selected | 30 | 0.8153 | voting_top3 | Bonferroni + multikolinearite seçimi |
| v3_new_features | 45 | — | — | Temporal feature engineering |
| v1/v2 | ~41 | — | — | LGBM baseline + tuning |
