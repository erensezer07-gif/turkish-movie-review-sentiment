# 🎬 Sezer Film - AI Destekli Film Analiz Platformu

**AI destekli, Türkçe film yorumları üzerinde hibrit duygu analizi yapan ve TMDB API ile zenginleştirilmiş modern bir film analiz platformu.**

## 🌐 Canlı Demo

Projeniz Render üzerinde yayınlandığında canlı demosu burada görünecektir.

� **Deploy Etmek İçin:** Aşağıdaki "Render'da Yayınla" adımlarını takip ederek kendi linkinizi oluşturun.

*(Kurulum sonrası bu alanı kendi URL'niz ile güncelleyebilirsiniz: `https://proje-adiniz.onrender.com`)*

## ✨ Özellikler

-   **Direct AI Integration:** Direct Mode mimarisinde AI modelleri Django uygulaması içinde bellek üzerinde yüklenir ve inference işlemleri doğrudan Python katmanında gerçekleştirilir.
-   **Veri Madenciliği:** TMDB API entegrasyonu ile binlerce film verisi ve posteri otomatik olarak çekilir.
-   **Hibrit Duygu Analizi:** Yorumlar; Derin Öğrenme (BERT), Makine Öğrenmesi (TF-IDF) ve Kural Tabanlı sistemlerin ağırlıklı ortalaması ile analiz edilir.
-   **Modern Arayüz:** Responsive tasarım, Netflix tarzı Hero Carousel ve dinamik ızgara (Grid) yapısı.
-   **Güvenlik:** `.env` yönetimi ve CSRF korumaları.

## 🧠 Kullanılan AI Modeli

Duygu analizi sistemi **3 sınıflı (Olumlu / Nötr / Olumsuz)** sınıflandırma yapacak şekilde eğitilmiştir.

**Model Künyesi:**
* 📦 **Model Versiyonu:** `benim_bert_modelim_3cls_v2`
* 🤖 **Mimari:** Fine-tuned BERT (`dbmdz/bert-base-turkish-cased`)
* 📊 **Yardımcı Model:** TF-IDF + Logistic Regression
* 🛡️ **Guardrails:** İroni, Spam ve Anlamsız Metin Filtresi

🚀 **Veri Seti:** Model, **Beyazperde** ve diğer kaynaklardan toplanan **190.000+ satırlık** temizlenmiş Türkçe film yorum veri seti ile eğitilmiştir.

## 🔄 AI Analiz Akışı

1.  **Giriş:** Kullanıcı yorumu Django view katmanına ulaşır.
2.  **Ön İşleme:** Metin temizlenir (noktalama, lower-case) ve Guardrail kontrolünden geçer.
3.  **Derin Analiz:** Fine-tuned BERT modeli metnin bağlamını (context) analiz eder.
4.  **İstatistiksel Analiz:** TF-IDF modeli kelime köklerini ve frekanslarını değerlendirir.
5.  **Karar (Ensemble):** Her iki modelin çıktıları ağırlıklı bir algoritma ile birleştirilerek nihai karar verilir.

## 🏗️ Proje Mimarisi (Direct Mode)

Bu projede mikroservis karmaşası yerine, performans ve yönetim kolaylığı için **Monolithic AI** yaklaşımı benimsenmiştir.

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

## ☁️ Deployment

Proje Render üzerinde tek bir web servisi olarak deploy edilmiştir. Uygulama başlatıldığında AI modelleri (~450MB) belleğe **preload** edilir ve sonraki isteklerde **düşük gecikmeli (low-latency) inference** sağlanır. Bu mimari, soğuk başlangıç (cold-start) sonrası maksimum performans sunar.

## 🛠️ Teknolojiler

| Katman | Teknoloji |
| :--- | :--- |
| Backend & AI | Django 6.0 + PyTorch |
| MLOps | Model Versioning, Direct Inference Pipeline, Monolithic AI Integration |
| NLP | Transformers (Hugging Face) + Scikit-learn |
| Veritabanı | SQLite (Dev) / PostgreSQL (Prod) |
| API | TMDB API v3 |
| Frontend | HTML5 / CSS3 / Bootstrap 5 |

## 🚀 Kurulum ve Çalıştırma

### 1. Projeyi Klonlayın

```bash
git clone https://github.com/KULLANICI_ADINIZ/Ilk_AI_Projem.git
cd Ilk_AI_Projem
```

### 2. Sanal Ortam ve Bağımlılıklar

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Mac/Linux

pip install -r requirements.txt
```

### 3. Ortam Değişkenleri (.env)

Proje kök dizininde `.env` dosyası oluşturun:

```ini
SECRET_KEY=gizli-anahtariniz
DEBUG=True
TMDB_API_KEY=tmdb_api_key_buraya
```

### 4. Uygulamayı Başlatın

```bash
python manage.py migrate
python manage.py runserver
```

Tarayıcınızda `http://127.0.0.1:8000` adresine gidin. Model otomatik yüklenecektir.

## 📁 Proje Yapısı

```
Ilk_AI_Projem/
├── filmler/                  # Django Uygulaması
├── sinema_sitesi/            # Ana Proje Ayarları
├── yapay_zeka_servisi/       # AI Motoru (Direct Mode)
│   ├── benim_bert_modelim_3cls_v2/
│   ├── film_tfidf_3cls.pkl
│   └── app_ensemble.py       # Analiz Mantığı
├── screenshots/              # Görseller
├── manage.py
└── README.md
```

## 🧪 AI Analiz Testi (Django Shell)

Terminalden manuel test için:

```bash
python manage.py shell
```

```python
from sinema_sitesi.ai_client import analiz_yap

# Test
print(analiz_yap("Bu film sinema tarihinin en iyisiydi."))
# Çıktı: {'karar': 'Olumlu', 'skor': 0.98, ...}
```

## 📸 Proje Ekran Görüntüleri 

*   🏠 **Ana Sayfa ve Liste**(![alt text](screenshots/anasayfa.png))
*   🔐 **Kullanıcı Giriş**(![alt text](screenshots/giris.png))
*   ⚙️ **Ai Yorum Algılama**(![alt text](screenshots/yorumlar.png))
*   🎬 **Film Detay**(![alt text](screenshots/detay.png))
*   🛡️ **Spam/Guardrail Koruması**(![alt text](screenshots/spam.png))
*   🧠 **Ai Dashboard** (![alt text](screenshots/ai_dashboard.png))

## 📄 Lisans

Bu proje **MIT Lisansı** ile lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakabilirsiniz.
