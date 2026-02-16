import logging
import time
import concurrent.futures
import requests
from django.core.management.base import BaseCommand
from django.conf import settings
from filmler.models import Film

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'TMDB API üzerinden filmleri ve türlerini çeker (Retry & Normalization support).'

    def add_arguments(self, parser):
        parser.add_argument('sayfa', type=int, help='Kaç sayfa veri çekilsin? (her sayfa ~20 film)')

    def handle(self, *args, **kwargs):
        sayfa_sayisi = kwargs['sayfa']
        self.api_key = getattr(settings, "TMDB_API_KEY", "")

        if not self.api_key:
            self.stderr.write(self.style.ERROR("[HATA] TMDB_API_KEY ayarlanmamış! .env dosyasını kontrol edin."))
            return

        # 1. Tür Listesini (Genre Map) Çek
        genre_map = {}
        try:
            g_data = self.fetch_tmdb("https://api.themoviedb.org/3/genre/movie/list", {"language": "tr-TR"})
            if g_data:
                for g in g_data.get("genres", []):
                    genre_map[g["id"]] = g["name"]
        except Exception as e:
             self.stderr.write(self.style.ERROR(f"Tür listesi çekilemedi: {e}"))
             return

        base_url = "https://api.themoviedb.org/3/movie/popular"
        self.stdout.write(self.style.WARNING(f">>> {sayfa_sayisi} sayfa taranıyor... Retry & Rate Limit aktif."))

        # İşlem fonksiyonu (Closure olarak genre_map'e erişir)
        def process_movie_wrapper(movie_data):
            return self.process_movie(movie_data, genre_map)

        # Ana Döngü
        count = 0
        existing_count = 0
        
        for sayfa in range(1, sayfa_sayisi + 1):
            self.stdout.write(f"--- Sayfa {sayfa}/{sayfa_sayisi} çekiliyor ---")
            
            data = self.fetch_tmdb(base_url, {"language": "tr-TR", "page": sayfa})
            if not data:
                self.stderr.write(self.style.ERROR(f"Sayfa {sayfa} alınamadı, atlanıyor."))
                continue

            movies = data.get("results", [])
            
            # Threading ile işlem
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(process_movie_wrapper, movies))

            # DB Kayıt (Main Thread)
            for item in results:
                if not item:
                    continue
                
                title = item['isim']
                defaults = item['defaults']
                
                # Normalizasyon ve Duplicate Kontrolü (Case-insensitive)
                # Önce var mı diye bak (iexact). Varsa güncelle, yoksa oluştur.
                # update_or_create iexact ile çalışmaz (get yapar, dönerse update eder).
                # Ama biz isme göre update etmek istiyoruz.
                
                # Normalize Title (strip)
                title = title.strip()
                
                obj, created = Film.objects.update_or_create(
                    isim__iexact=title,
                    defaults={'isim': title, **defaults} # İsmi de düzgün haliyle kaydet/güncelle
                )
                
                if created:
                    count += 1
                else:
                    existing_count += 1

            # Sayfa arası bekleme (nezaket)
            time.sleep(0.5)

        self.stdout.write(self.style.SUCCESS(f"\n[TAMAM] İşlem Bitti!"))
        self.stdout.write(self.style.SUCCESS(f"Eklenen: {count}, Güncellenen: {existing_count}"))

    def fetch_tmdb(self, url, params=None):
        """
        TMDB API isteği atar. Retry, Backoff ve Rate Limit loglama içerir.
        """
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        
        max_retries = 3
        retry_delay = 1 # saniye
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(url, params=params, timeout=10)
                
                # Rate Limit Loglama (Header kontrolü)
                remaining = response.headers.get('X-RateLimit-Remaining')
                if remaining and int(remaining) < 5:
                    self.stdout.write(self.style.WARNING(f"UYARI: Rate Limit azalıyor! Kalan: {remaining}"))
                
                if response.status_code == 429:
                    # Too Many Requests
                    wait = int(response.headers.get('Retry-After', 5)) + 1
                    logger.warning(f"Rate Limit aşıldı (429). {wait}s bekleniyor...")
                    time.sleep(wait)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                if attempt < max_retries:
                    wait = retry_delay * (2 ** attempt) # Exponential backoff: 1, 2, 4...
                    logger.warning(f"API Hatası: {e}. {wait}s sonra tekrar deneniyor ({attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    logger.error(f"API isteği başarısız oldu: {url} | Hata: {e}")
                    return None
        return None

    def process_movie(self, film_data, genre_map):
        """Tek bir filmi işleyip sözlük döner."""
        try:
            film_id = film_data["id"]
            title = film_data["title"]

            # Türleri Bul
            genre_ids = film_data.get("genre_ids", [])
            film_turleri = ", ".join([genre_map.get(gid, "") for gid in genre_ids if gid in genre_map])

            # Oyuncular
            cast_str = "Bilgi yok"
            c_data = self.fetch_tmdb(f"https://api.themoviedb.org/3/movie/{film_id}/credits")
            if c_data:
                cast_names = [p["name"] for p in c_data.get("cast", [])[:5]]
                if cast_names:
                    cast_str = ", ".join(cast_names)

            # Fragman
            fragman_link = ""
            # Önce TR
            v_data = self.fetch_tmdb(f"https://api.themoviedb.org/3/movie/{film_id}/videos", {"language": "tr-TR"})
            videos = v_data.get("results", []) if v_data else []
            
            if not videos:
                # Sonra EN
                v_data = self.fetch_tmdb(f"https://api.themoviedb.org/3/movie/{film_id}/videos", {"language": "en-US"})
                videos = v_data.get("results", []) if v_data else []
            
            for v in videos:
                if v["site"] == "YouTube" and v["type"] == "Trailer":
                    fragman_link = f"https://www.youtube.com/embed/{v['key']}"
                    break

            poster_path = film_data.get("poster_path")
            
            return {
                'isim': title,
                'defaults': {
                    'konu': film_data.get("overview", "Özet yok."),
                    'yil': film_data.get("release_date", "2024").split("-")[0],
                    'puan': round(film_data.get("vote_average", 0), 1),
                    'oyuncular': cast_str,
                    'turler': film_turleri,
                    'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "",
                    'fragman_url': fragman_link,
                },
            }
        except Exception as e:
            logger.error(f"Film işleme hatası ({film_data.get('title')}): {e}")
            return None