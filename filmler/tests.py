from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from unittest.mock import patch, MagicMock
from filmler.models import Film, Yorum
from filmler.services.moderation_service import kufur_kontrol, anlamsiz_mi
from filmler.services.sentiment_service import analyze_comment

class ModerationServiceTest(TestCase):
    def test_kufur_kontrol(self):
        # Pozitif (Küfürlü)
        self.assertTrue(kufur_kontrol("Senin amk"))
        self.assertTrue(kufur_kontrol("O bir yavşak"))
        
        # Negatif (Temiz)
        self.assertFalse(kufur_kontrol("Çok güzel bir film"))
        self.assertFalse(kufur_kontrol("Harika oyunculuk"))

    def test_anlamsiz_mi(self):
        # Edge Case: Klavye ezme / Gibberish
        self.assertTrue(anlamsiz_mi("asdasdasd"))
        self.assertTrue(anlamsiz_mi("qweqweqwe"))
        
        # Edge Case: Sessiz harf yığını (Rule A)
        self.assertTrue(anlamsiz_mi("dmşkamk")) # 4+ sessiz
        
        # Edge Case: Sesli harf yokluğu (Rule B)
        self.assertTrue(anlamsiz_mi("trkc")) # >3 harf, sesli yok
        
        # Edge Case: Noktalama istismarı (Rule C)
        self.assertTrue(anlamsiz_mi("s.a.l.a.k"))
        
        # Valid Case: Normal Cümle
        self.assertFalse(anlamsiz_mi("Bu film gerçekten harikaydı."))
        self.assertFalse(anlamsiz_mi("Oyunculuk çok başarılı."))


class SentimentServiceTest(TestCase):
    @patch('filmler.services.sentiment_service.analiz_yap')
    def test_analyze_comment_fallback(self, mock_analiz_yap):
        # Senaryo: AI Servisi Exception Fırlatıyor (Down)
        mock_analiz_yap.side_effect = Exception("Connection refused")
        
        result = analyze_comment("Test yorumu")
        
        # Beklenen: Fallback (NÖTR)
        self.assertEqual(result["decision"], "NÖTR")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["source"], "api_error")

    @patch('filmler.services.sentiment_service.analiz_yap')
    def test_analyze_comment_import_error(self, mock_analiz_yap):
        # Senaryo: AI Client None (Import Error simülasyonu)
        # analyze_comment içinde analiz_yap import edilemezse None oluyor.
        # Bunu test etmek için views.py'yi reload etmek gerekebilir ama
        # mock ile fonksiyonun kendisini None yapamayız (patch hata verir).
        # Ancak analyze_comment içindeki `if analiz_yap is None` bloğunu test etmek için:
        
        with patch('filmler.services.sentiment_service.analiz_yap', None):
             result = analyze_comment("Test")
             self.assertEqual(result["decision"], "NÖTR")
             self.assertEqual(result["source"], "api_error")


class ViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='password')
        self.film = Film.objects.create(isim="Matrix", puan=8.7, yil=1999)

    def test_live_search_schema(self):
        # Test: /live-search/?term=Mat
        response = self.client.get(reverse('live_search'), {'term': 'Mat'})
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertTrue(len(data) > 0)
        
        item = data[0]
        self.assertIn("id", item)
        self.assertIn("isim", item)
        self.assertIn("poster", item)
        self.assertIn("yil", item)
        self.assertIn("puan", item)

    @patch('filmler.views.analyze_comment') # View içindeki servisi mockluyoruz
    def test_film_detay_post_flow(self, mock_analyze):
        self.client.login(username='testuser', password='password')
        url = reverse('film_detay', args=[self.film.id])
        
        # 1. Senaryo: Küfürlü Yorum
        response = self.client.post(url, {'yorum_icerigi': 'Bu bir amk filmi'})
        # Küfür yakalanmalı, kaydedilmemeli. 
        # Mesaj framework'ü kullanıldığı için redirect döner (302)
        # Mesajı kontrol etmek integration test işi, count kontrolü yeterli.
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Yorum.objects.count(), 0)

        # 2. Senaryo: Normal Yorum
        # AI sonucunu mockla
        mock_analyze.return_value = {
            "decision": "OLUMLU",
            "confidence": 0.95,
            "source": "mock_ai",
            "duration": 0.1
        }
        
        response = self.client.post(url, {'yorum_icerigi': 'Harika bir filmdi kesinlikle izleyin.'})
        self.assertEqual(response.status_code, 302) # Redirect to self
        self.assertEqual(Yorum.objects.count(), 1)
        
        yorum = Yorum.objects.first()
        self.assertEqual(yorum.icerik, 'Harika bir filmdi kesinlikle izleyin.')
        self.assertEqual(yorum.ai_kaynak, 'mock_ai')
