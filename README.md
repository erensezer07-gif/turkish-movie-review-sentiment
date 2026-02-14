# 🎬 Sezer Film - AI Destekli Film Analiz Platformu

**Direct Mode** mimarisiyle geliştirilmiş, hibrit yapay zeka (BERT + TF-IDF) destekli yeni nesil film analiz ve öneri platformu.

## 🌐 Canlı Demo

Projenin çalışan versiyonunu aşağıdaki linkten deneyebilirsiniz:

🔗 **[Sezer Film - AI Platformu](https://sezerfilm.onrender.com)**

> ⚠️ **Teknik Not:** AI duygu analizi modeli, harici bir API yerine doğrudan Django uygulaması içinde (**Direct Mode**) çalışmaktadır. Sunucu uyku modundaysa (Cold Start), modelin belleğe yüklenmesi nedeniyle ilk analizde 15-20 saniyelik bir gecikme yaşanabilir. Sonraki analizler milisaniyeler sürer.

## ✨ Özellikler

-   **Direct AI Integration:** Direct Mode mimarisinde AI modelleri Django uygulaması içinde bellek üzerinde yüklenir ve inference işlemleri doğrudan Python katmanında gerçekleştirilir. Böylece harici API çağrıları ve ağ gecikmeleri ortadan kaldırılmıştır.
-   **Veri Madenciliği:** TMDB API entegrasyonu ile binlerce film verisi ve posteri otomatik olarak çekilir.
-   **Hibrit Duygu Analizi:** Yorumlar; Derin Öğrenme (BERT), Makine Öğrenmesi (TF-IDF) ve Kural Tabanlı sistemlerin ağırlıklı ortalaması ile analiz edilir.
-   **Modern Arayüz:** Responsive tasarım, Netflix tarzı Hero Carousel ve dinamik ızgara (Grid) yapısı.
-   **Güvenlik:** `.env` yönetimi ve CSRF korumaları.

## 🧠 Kullanılan AI Modeli

Duygu analizi sistemi **3 sınıflı (Olumlu / Nötr / Olumsuz)** sınıflandırma yapacak şekilde eğitilmiştir.

**Model Bileşenleri:**
* **Fine-tuned BERT:** `dbmdz/bert-base-turkish-cased` modeli, Türkçe dil yapısını anlamak için fine-tune edilmiştir.
* **TF-IDF + Logistic Regression:** Kelime frekansına dayalı klasik ML modeli, BERT'in gözden kaçırabileceği basit sinyalleri yakalar.
* **Rule-Based Guardrails:** İroni ("Şaka yapıyorum"), spam ve anlamsız yorumları filtreleyen özel Python kuralları.

🚀 **Veri Seti:** Model, **Beyazperde** ve diğer kaynaklardan toplanan **190.000+ satırlık** temizlenmiş Türkçe film yorum veri seti ile eğitilmiştir.

## 🔄 AI Analiz Akışı

Kullanıcı bir yorum gönderdiğinde sistem şu adımları izler:

1.  **Giriş:** Kullanıcı yorumu Django view katmanına ulaşır.
2.  **Ön İşleme:** Metin temizlenir (noktalama, lower-case) ve Guardrail kontrolünden geçer (Spam/İroni).
3.  **Derin Analiz:** Fine-tuned BERT modeli metnin bağlamını (context) analiz eder.
4.  **İstatistiksel Analiz:** TF-IDF modeli kelime köklerini ve frekanslarını değerlendirir.
5.  **Karar (Ensemble):** Her iki modelin ve kuralların çıktıları ağırlıklı bir algoritma ile birleştirilerek nihai **Olumlu/Nötr/Olumsuz** kararı verilir.

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

Proje Render üzerinde tek bir web servisi olarak deploy edilmiştir. Uygulama başladığında AI modelleri (yaklaşık 500MB) belleğe yüklenir ve HTTP istekleri gelmeden sistem hazır hale gelir. Bu sayede harici bir işlemciye ihtiyaç duyulmaz.

## 🛠️ Teknolojiler

| Katman | Teknoloji |
| :--- | :--- |
| Backend & AI | Django 6.0 + PyTorch |
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

*   🖥️ **Temel Kullanıcı Arayüzü**
*   🏠 **Ana Sayfa ve Liste**
*   🔐 **Kullanıcı Giriş**
*   ⚙️ **Özellikler**
*   🎬 **Film Detay**
*   🛡️ **Spam/Guardrail Koruması**
*   🧠 ⭐ **Öne Çıkan Özellik:** Gelişmiş AI Analiz Paneli

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.
