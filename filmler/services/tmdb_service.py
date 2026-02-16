import logging
import re
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

TMDB_API_KEY = getattr(settings, "TMDB_API_KEY", "")


def fetch_movie_details(movie_name: str, existing_trailer_url: str = None):
    """
    TMDB'den film detaylarını çeker.
    Dönen sözlük:
    {
        'runtime': int | None,
        'genres': str | None,
        'backdrop_url': str | None,
        'cast_list': list,
        'trailer_url': str | None
    }
    """
    runtime = None
    genres = None
    backdrop_url = None
    cast_list = []
    trailer_watch_url = None

    if TMDB_API_KEY:
        try:
            search_res = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={"api_key": TMDB_API_KEY, "language": "tr-TR", "query": movie_name},
                timeout=3,
            ).json()

            if search_res.get("results"):
                tmdb_id = search_res["results"][0]["id"]
                # append_to_response ile krediler, resimler ve videolar
                detay = requests.get(
                    f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                    params={
                        "api_key": TMDB_API_KEY,
                        "language": "tr-TR",
                        "append_to_response": "credits,images,videos"
                    },
                    timeout=3,
                ).json()

                runtime = detay.get("runtime")
                genres = ", ".join([g.get("name") for g in detay.get("genres", [])])

                # Backdrop (Arka plan)
                if detay.get("backdrop_path"):
                    backdrop_url = f"https://image.tmdb.org/t/p/original{detay['backdrop_path']}"

                # Oyuncular (Cast) - İlk 15 kişi
                credits = detay.get("credits", {})
                for person in credits.get("cast", [])[:15]:
                    profile = None
                    if person.get("profile_path"):
                        profile = f"https://image.tmdb.org/t/p/w185{person['profile_path']}"
                    
                    cast_list.append({
                        "name": person.get("name"),
                        "character": person.get("character"),
                        "photo": profile
                    })
                
                # --- FRAGMAN ÖNCELİK MANTIĞI ---
                videos = detay.get("videos", {}).get("results", [])
                best_video = None
                
                # 1. TR Dublaj / Fragman
                for v in videos:
                    if v['site'] == 'YouTube' and v['iso_639_1'] == 'tr' and v['type'] == 'Trailer':
                        best_video = v
                        break
                
                # 2. TR Herhangi Bir Video (Teaser vb.)
                if not best_video:
                    for v in videos:
                        if v['site'] == 'YouTube' and v['iso_639_1'] == 'tr':
                            best_video = v
                            break
                            
                # 3. EN Fragman (Yabancı dilde orijinal fragman)
                if not best_video:
                    for v in videos:
                        if v['site'] == 'YouTube' and v['iso_639_1'] == 'en' and v['type'] == 'Trailer':
                            best_video = v
                            break

                # Eğer TMDB'den bulduysak, onu kullan. Yoksa DB fragman_url kalır.
                if best_video:
                    # YouTube Embed URL formatı (otomatik oynatma ve kontrol parametreleri eklenebilir)
                    # Ancak burada sadece embed URL'i hazırlıyoruz.
                    # Frontend'de iframe src olarak kullanılacak.
                    trailer_watch_url = f"https://www.youtube.com/embed/{best_video['key']}?autoplay=1&rel=0"

        except requests.RequestException as e:
            logger.warning("TMDB API hatası: %s", e)

    # Eğer TMDB'den gelmediyse ve DB'de varsa, DB'dekini embed formatına çevir
    if not trailer_watch_url and existing_trailer_url:
        match = re.search(r"youtube\.com/embed/([a-zA-Z0-9_-]+)", existing_trailer_url)
        if match:
            # ?autoplay=1 ekleyerek tıklanınca başlamasını sağlıyoruz
            trailer_watch_url = f"https://www.youtube.com/embed/{match.group(1)}?autoplay=1&rel=0"

    return {
        "runtime": runtime,
        "genres": genres,
        "backdrop_url": backdrop_url,
        "cast_list": cast_list,
        "trailer_watch_url": trailer_watch_url
    }
