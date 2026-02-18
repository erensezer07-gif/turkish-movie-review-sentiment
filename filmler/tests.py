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

    def test_anlamsiz_mi_edge_cases(self):
        """
        Anlamsız metin kontrolü için detaylı edge-case testleri.
        """
        # 1. Klavye Ezme (Gibberish)
        self.assertTrue(anlamsiz_mi("asdasdasd"), "Standart gibberish yakalanmalı")
        self.assertTrue(anlamsiz_mi("qweqweqwe"), "QWE deseni yakalanmalı")
        
        # 2. Sessiz Harf Yığını (Rule A: Consonant Cluster)
        self.assertTrue(anlamsiz_mi("dmşkamk"), "Sessiz harf yığını yakalanmalı")
        self.assertTrue(anlamsiz_mi("strkpl"), "Sesli harf içermeyen uzun kelime yakalanmalı")
        
        # 3. Sesli Harf Yokluğu / Kısa Anlamsız (Rule B)
        self.assertTrue(anlamsiz_mi("trkc"), "Sesli harf yoksa yakalanmalı")
        self.assertTrue(anlamsiz_mi("bc"), "Çok kısa ve sesli yoksa yakalanmalı")
        
        # 4. Noktalama İstismarı (Rule C)
        self.assertTrue(anlamsiz_mi("s.a.l.a.k"), "Noktalama ile ayrılmış harfler yakalanmalı")
        self.assertTrue(anlamsiz_mi("......."), "Sadece noktalama içeren yorum yakalanmalı")
        self.assertTrue(anlamsiz_mi("??????"), "Sadece noktalama içeren yorum yakalanmalı")

        # 5. Tekrar Eden Karakterler (Repetitive Chars)
        self.assertTrue(anlamsiz_mi("aaaaaaaaaa"), "Tekrar eden aynı karakter yakalanmalı")
        self.assertTrue(anlamsiz_mi("hahahahahah"), "Tekrar eden hece (eğer stopwords değilse) yakalanabilir (context bağlı)")

        # 6. Valid Case: Normal Cümleler & Kısa Geçerli Kelimeler
        self.assertFalse(anlamsiz_mi("Bu film gerçekten harikaydı."), "Normal cümle geçerli olmalı")
        self.assertFalse(anlamsiz_mi("Oyunculuk çok başarılı."), "Normal cümle geçerli olmalı")
        self.assertFalse(anlamsiz_mi("Ok."), "Kısa ama geçerli kelime (stopwords/common) geçmeli")
        self.assertFalse(anlamsiz_mi("Evet"), "Kısa ama geçerli kelime geçmeli")
        self.assertFalse(anlamsiz_mi("İyi"), "Kısa Türkçe kelime geçmeli")


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

    @patch('filmler.services.sentiment_service.analiz_yap', None)
    def test_analyze_comment_import_error(self):
        # Senaryo: AI Client None (Import Error simülasyonu)
        result = analyze_comment("Test")
        self.assertEqual(result["decision"], "NÖTR")
        self.assertEqual(result["source"], "api_error")

    def test_invalid_inputs(self):
        """
        Geçersiz input tipleri (None, boş string, int) NÖTR dönmeli.
        """
        invalid_inputs = [None, "", 123, [], {}]
        for bad_input in invalid_inputs:
            result = analyze_comment(bad_input)
            self.assertEqual(result["decision"], "NÖTR", f"Input failed: {bad_input}")
            self.assertEqual(result["source"], "invalid_input")

    @patch('filmler.services.sentiment_service.analiz_yap')
    def test_normalization(self, mock_analiz_yap):
        """
        API farklı formatlarda dönse bile (Olumlu, OLUMLU, olumlu)
        servis standart çıktı üretmeli.
        """
        scenarios = [
            ("Olumlu", "OLUMLU"),
            ("olumlu", "OLUMLU"),
            ("OLUMLU", "OLUMLU"),
            ("Olumsuz", "OLUMSUZ"),
            ("Kararsız", "NÖTR"),
            ("Nötr", "NÖTR"),
        ]

        for api_output, expected_db in scenarios:
            mock_analiz_yap.return_value = {
                "karar": api_output,
                "guven_skoru": 0.99,
                "kaynak": "test",
                "sure_sn": 0.1
            }
            
            result = analyze_comment("Test")
            self.assertEqual(result["decision"], expected_db, f"Normalization failed for {api_output}")


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

    def test_live_search_edge_cases(self):
        # 1. Boş Query -> Boş Liste
        response = self.client.get(reverse('live_search'), {'term': ''})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

        # 2. Tek Karakter -> Boş Liste (Performans optimizasyonu)
        response = self.client.get(reverse('live_search'), {'term': 'M'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

        # 3. Bulunamayan Film -> Boş Liste
        response = self.client.get(reverse('live_search'), {'term': 'Zzzzzzzzz'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    @patch('filmler.views.analyze_comment') # View içindeki servisi mockluyoruz
    def test_film_detay_post_standard_flow(self, mock_analyze):
        """
        Standart POST akışı (Redirect ile)
        """
        self.client.login(username='testuser', password='password')
        url = reverse('film_detay', args=[self.film.id])
        
        # AI Sonucunu Mockla
        mock_analyze.return_value = {
            "decision": "OLUMLU",
            "confidence": 0.95,
            "source": "mock_ai",
            "duration": 0.1
        }
        
        # Valid POST
        response = self.client.post(url, {'yorum_icerigi': 'Harika bir filmdi kesinlikle izleyin.'})
        
        # Redirect döner (302)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Yorum.objects.count(), 1)
        
        # Veriyi doğrula
        yorum = Yorum.objects.first()
        self.assertEqual(yorum.icerik, 'Harika bir filmdi kesinlikle izleyin.')
        self.assertEqual(yorum.ai_karari, 'OLUMLU')

    def test_film_detay_ajax_errors(self):
        """
        AJAX isteklerinde hataların 400 + JSON dönmesi gerektiğini doğrular.
        """
        self.client.login(username='testuser', password='password')
        url = reverse('film_detay', args=[self.film.id])
        headers = {'HTTP_X_REQUESTED_WITH': 'XMLHttpRequest'}

        # 1. Küfürlü Yorum (AJAX)
        response = self.client.post(url, {'yorum_icerigi': 'Senin amk'}, **headers)
        self.assertEqual(response.status_code, 400)
        self.assertJSONEqual(response.content, {"ok": False, "error": "⛔ Yorumunuz uygunsuz ifade içeriyor. Lütfen saygılı bir dil kullanın."})
        self.assertEqual(Yorum.objects.count(), 0)

        # 2. Anlamsız Yorum (AJAX)
        response = self.client.post(url, {'yorum_icerigi': 'asdasdasd'}, **headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("anlamlı bir metin", response.json()["error"])

        # 3. Çok Kısa Yorum (AJAX)
        response = self.client.post(url, {'yorum_icerigi': 'a'}, **headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Yorum çok kısa", response.json()["error"])

