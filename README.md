# 🎬 Sezer Film - AI Destekli Film Analiz Platformu
### Turkish Movie Review Sentiment Analysis with BERT + TF-IDF Ensemble

**AI destekli, Türkçe film yorumları üzerinde hibrit duygu analizi yapan ve TMDB API ile zenginleştirilmiş modern bir film analiz platformu.**



## 🌐 Canlı Demo

🔗 **Demo:** https://turkish-movie-review-sentiment.onrender.com


> ⚠️ **Teknik Not:** AI duygu analizi modeli, harici bir API yerine doğrudan Django uygulaması içinde (**Direct Mode**) çalışmaktadır. Sunucu uyku modundaysa (Cold Start), modelin belleğe yüklenmesi nedeniyle ilk analizde 10-15 saniyelik bir gecikme yaşanabilir. Sonraki analizler milisaniyeler sürer.


## ✨ Özellikler

- **🚀 Direct AI Integration:** Direct Mode mimarisinde AI modelleri Django uygulaması içinde bellek üzerinde yüklenir ve inference işlemleri doğrudan Python katmanında gerçekleştirilir.
- **🕷️ Veri Madenciliği:** TMDB API entegrasyonu ile binlerce film verisi ve posteri otomatik olarak çekilir.
- **🧠 Hibrit Duygu Analizi:** Yorumlar; Derin Öğrenme (BERT), Makine Öğrenmesi (TF-IDF) ve Kural Tabanlı sistemlerin ağırlıklı ortalaması ile analiz edilir.
- **🎨 Modern Arayüz:** Responsive tasarım, Netflix tarzı Hero Carousel ve dinamik ızgara (Grid) yapısı.
- **🔒 Güvenlik:** `.env` yönetimi ve CSRF korumaları.

## 🧠 Kullanılan AI Modeli

Duygu analizi sistemi **3 sınıflı (Olumlu / Nötr / Olumsuz)** sınıflandırma yapacak şekilde eğitilmiştir.

**Model Künyesi:**
* 📦 **Model Versiyonu:** `benim_bert_modelim_3cls_v2`
* 🤖 **Mimari:** Fine-tuned BERT (`dbmdz/bert-base-turkish-cased`)
* 📊 **Yardımcı Model:** TF-IDF + Logistic Regression (Not: `film_tfidf_3cls.pkl`, ~10MB boyutu ve deployment tutarlılığı için repo'ya dahil edilmiştir.)
* 🛡️ **Guardrails:** İroni, Spam ve Anlamsız Metin Filtresi

🚀 **Veri Seti:** Model, **Beyazperde** ve diğer kaynaklardan toplanan **190.000+ satırlık** temizlenmiş Türkçe film yorum veri seti ile eğitilmiştir.

## 🔄 AI Analiz Akışı

1. **Giriş:** Kullanıcı yorumu Django view katmanına ulaşır.
2. **Ön İşleme:** Metin temizlenir (noktalama, lower-case) ve Guardrail kontrolünden geçer.
3. **Derin Analiz:** Fine-tuned BERT modeli metnin bağlamını (context) analiz eder.
4. **İstatistiksel Analiz:** TF-IDF modeli kelime köklerini ve frekanslarını değerlendirir.
5. **Karar (Ensemble):** Her iki modelin çıktıları ağırlıklı bir algoritma ile birleştirilerek nihai karar verilir.

## 🏗️ Proje Mimarisi (Direct Mode)

Bu projede mikroservis karmaşası yerine, performans ve yönetim kolaylığı için **Monolithic AI** yaklaşımı benimsenmiştir.

```mermaid
graph TD
    User[Kullanıcı] -->|HTTP Request| Django[Django Web App]
    Django -->|Veri Çekme| TMDB[TMDB API]
    subgraph "AI Engine (In-Memory)"
        Django -->|Metin| Preprocess[Ön İşleme]
        Preprocess --> BERT[BERT Model]
        Preprocess --> TFIDF[TF-IDF Model]
        BERT --> Ensemble[Ağırlıklı Karar]
        TFIDF --> Ensemble
    end
    Ensemble -->|Sonuç: Olumlu/Olumsuz| Django
```


```text
┌──────────────────────────┐
│      Django Web App      │
│   (UI + Backend + AI)    │
│            │             │
│   ┌──────────────────┐   │
│   │ AI Ensemble Core │   │
│   │ - Fine-tuned BERT│   │
│   │ - TF-IDF + LR    │   │
│   │ - Guardrails     │   │
│   └──────────────────┘   │
│            │             │
│   TMDB API v3 (Film Data)│
└──────────────────────────┘
```

## 📸 Ekran Görüntüleri

Projenin arayüzünden bazı kareler:


| Ana Sayfa | Film Detay |
|-----------|------------|
| ![Ana Sayfa](screenshots/anasayfa.png) | ![Detay](screenshots/detay.png) |

| Kullanıcı Giriş | Yorumlar & Analiz |
|-----------------|-------------------|
| ![Giriş](screenshots/giris.png) | ![Yorumlar](screenshots/yorumlar.png) |

| AI Analiz Dashboard | Spam Koruması |
|---------------------|---------------|
| ![AI Dashboard](screenshots/ai_dashboard.png) | ![Spam](screenshots/spam.png) |

## 🛠️ Kurulum (Local Development)

Projeyi kendi bilgisayarınızda çalıştırmak için:

1. **Repoyu klonlayın:**
   ```bash
   git clone https://github.com/erensezer07-gif/turkish-movie-review-sentiment.git
   cd turkish-movie-review-sentiment
   ```

2. **Sanal ortamı kurun:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```

3. **Bağımlılıkları yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Modeli indirin:**
   ```bash
   python scripts/download_model.py
   ```

    > ⚠️ **Python Sürümü Notu:**
    > Proje production ortamında (Render) `runtime.txt` dosyasında belirtilen **Python 3.11.9** ile çalışmaktadır.
    > Local geliştirme ortamınızda **Python 3.11+** (ör. 3.11.4) kullanmanız yeterlidir, ancak birebir uyumluluk için 3.11.9 önerilir.


5. **Ortam değişkenlerini (.env) ayarlayın:**
   Projeyi çalıştırmak için `.env` dosyası oluşturun:
   ```ini
   SECRET_KEY=gizli_anahtariniz
   DEBUG=True
   TMDB_API_KEY=tmdb_api_key_buraya
   ```

6. **Veritabanını başlatın ve çalıştırın:**
   ```bash
   python manage.py runserver
   ```

## 📦 Dependency Yönetimi

Bu projede bağımlılıklar standartlaştırılmıştır ve **pip-tools** ile yönetilmektedir.

### 1. Kurulum Standardı (Local / CI / Prod)
Tüm ortamlar için tek geçerli kurulum komutu şudur:
```bash
pip install -r requirements.txt
```
> ⚠️ **UYARI:** `requirements.txt` dosyası asla elle düzenlenmemelidir. Bu dosya otomatize bir süreçle kilitlenmiştir (pinned & hashed).

### 2. Paket Ekleme/Çıkarma
Yeni bir paket eklemek veya çıkarmak için:
1. `requirements.in` dosyasını düzenleyin.
2. Lock dosyasını güncelleyin:
   ```bash
   pip-compile --generate-hashes requirements.in -o requirements.txt
   ```
3. Değişiklikleri yükleyin:
   ```bash
   pip-sync
   # veya
   pip install -r requirements.txt
   ```

### 3. Güncelleme Politikası (Routine Bump)
Bağımlılıklar **aylık veya 2 aylık periyotlarla** güncellenmelidir.

**Örnek Güncelleme Komutları:**
```bash
# Tüm paketleri güvenli aralıkta güncelle
pip-compile --upgrade requirements.in -o requirements.txt

# Sadece belirli bir paketi güncelle (Ör: Django)
pip-compile --upgrade-package django --generate-hashes requirements.in -o requirements.txt

# Birden fazla paketi güncelle
pip-compile --upgrade-package transformers --upgrade-package torch --generate-hashes requirements.in -o requirements.txt
```

### 4. Doğrulama (CI Check)
Her güncelleme sonrası mutlaka testler çalıştırılmalıdır:
```bash
python manage.py test
```

## 👨‍💻 Geliştirici

**Eren Sezer**
Computer Engineering Student | AI & NLP Enthusiast
