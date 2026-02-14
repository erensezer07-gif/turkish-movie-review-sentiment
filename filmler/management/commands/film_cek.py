import logging
import time
import concurrent.futures

import requests
from django.core.management.base import BaseCommand
from django.conf import settings
from filmler.models import Film

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'TMDB API üzerinden filmleri ve türlerini çeker.'

    def add_arguments(self, parser):
        parser.add_argument('sayfa', type=int, help='Kaç sayfa veri çekilsin? (her sayfa ~20 film)')

    def handle(self, *args, **kwargs):
        sayfa_sayisi = kwargs['sayfa']
        api_key = getattr(settings, "TMDB_API_KEY", "")

        if not api_key:
            self.stderr.write(self.style.ERROR("[HATA] TMDB_API_KEY ayarlanmamış! .env dosyasını kontrol edin."))
            return

        # 1. Tür Listesini (Genre Map) Çek
        genre_map = {}
        try:
            g_res = requests.get(
                "https://api.themoviedb.org/3/genre/movie/list",
                params={"api_key": api_key, "language": "tr-TR"},
                timeout=10,
            )
            g_res.raise_for_status()
            for g in g_res.json().get("genres", []):
                genre_map[g["id"]] = g["name"]
        except requests.RequestException as e:
            self.stderr.write(self.style.ERROR(f"Tür listesi çekilemedi: {e}"))
            return

        base_url = "https://api.themoviedb.org/3/movie/popular"
        self.stdout.write(self.style.WARNING(f">>> {sayfa_sayisi} sayfa taranıyor... Türler güncellenecek."))

        def process_movie(film_data):
            film_id = film_data["id"]
            title = film_data["title"]

            # Türleri Bul
            genre_ids = film_data.get("genre_ids", [])
            film_turleri = ", ".join([genre_map.get(gid, "") for gid in genre_ids if gid in genre_map])

            # Oyuncuları Çek
            cast_str = "Bilgi yok"
            try:
                c_res = requests.get(
                    f"https://api.themoviedb.org/3/movie/{film_id}/credits",
                    params={"api_key": api_key},
                    timeout=5,
                )
                c_res.raise_for_status()
                cast_names = [p["name"] for p in c_res.json().get("cast", [])[:5]]
                if cast_names:
                    cast_str = ", ".join(cast_names)
            except requests.RequestException as e:
                logger.warning("Oyuncu bilgisi alınamadı (film_id=%s): %s", film_id, e)

            # Fragmanı Çek
            fragman_link = ""
            try:
                v_res = requests.get(
                    f"https://api.themoviedb.org/3/movie/{film_id}/videos",
                    params={"api_key": api_key, "language": "tr-TR"},
                    timeout=5,
                )
                v_res.raise_for_status()
                videos = v_res.json().get("results", [])

                if not videos:  # TR yoksa EN dene
                    v_res = requests.get(
                        f"https://api.themoviedb.org/3/movie/{film_id}/videos",
                        params={"api_key": api_key, "language": "en-US"},
                        timeout=5,
                    )
                    v_res.raise_for_status()
                    videos = v_res.json().get("results", [])

                for v in videos:
                    if v["site"] == "YouTube" and v["type"] == "Trailer":
                        fragman_link = f"https://www.youtube.com/embed/{v['key']}"
                        break
            except requests.RequestException as e:
                logger.warning("Fragman bilgisi alınamadı (film_id=%s): %s", film_id, e)

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

        # Ana Döngü
        count = 0
        for sayfa in range(1, sayfa_sayisi + 1):
            try:
                res = requests.get(
                    base_url,
                    params={"api_key": api_key, "language": "tr-TR", "page": sayfa},
                    timeout=10,
                )
                res.raise_for_status()
                movies = res.json().get("results", [])

                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    results = list(executor.map(process_movie, movies))

                for data in results:
                    if data:
                        Film.objects.update_or_create(isim=data['isim'], defaults=data['defaults'])
                        count += 1

                self.stdout.write(f"  [OK] Sayfa {sayfa}/{sayfa_sayisi} tamamlandı.")
            except requests.RequestException as e:
                self.stderr.write(self.style.ERROR(f"  [HATA] Sayfa {sayfa} hatası: {e}"))

            # Rate limiting — API sınırını aşmamak için
            time.sleep(0.25)

        self.stdout.write(self.style.SUCCESS(f"\n[TAMAM] Bitti! {count} film güncellendi/eklendi."))