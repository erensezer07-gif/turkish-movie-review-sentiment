import logging
import random


from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.db.models import Q

from .forms import UserRegisterForm
from .models import Film, Yorum

# Services
from .services.moderation_service import kufur_kontrol, anlamsiz_mi
from .services.sentiment_service import analyze_comment, get_sentiment_badge
from .services.tmdb_service import fetch_movie_details






# --- 1. CANLI ARAMA API (SPOTIFY TARZI) ---
def live_search(request):
    """
    Navbar'daki anlık arama kutusu için JSON döner.
    Spotify tarzı hızlı sonuç gösterimi sağlar.
    """
    term = request.GET.get("term", "").strip()
    results = []

    if len(term) > 1:
        # Performans: DB seviyesinde arama yap
        filmler = Film.objects.filter(isim__icontains=term)[:10]

        for film in filmler:
            results.append({
                "id": film.id,
                "isim": film.isim,
                "poster": film.poster_url,
                "yil": film.yil,
                "puan": film.puan,
            })

    return JsonResponse(results, safe=False)


# --- 2. ANA SAYFA (FİLTRELEME & SIRALAMA) ---
@login_required
def anasayfa(request):
    """
    Ana sayfa:
    - Varsayılan: Dashboard Modu (Hero, Popüler, Yeni, Kategoriler)
    - Arama/Filtre: Grid Modu (Mevcut yapı)
    """
    q = request.GET.get("q", "").strip()
    kategori = request.GET.get("kategori", "")
    sirala = request.GET.get("sirala", "")
    mode_param = request.GET.get("mode", "")

    # FİLTRELEME MODU (Eski Grid Yapısı)
    if q or kategori or sirala or mode_param == "liste":
        filmler = Film.objects.all()
        if kategori:
            filmler = filmler.filter(turler__icontains=kategori)
        if sirala == "puan":
            filmler = filmler.order_by("-puan")
        elif sirala == "yeni":
            filmler = filmler.order_by("-yil")
        else:
            filmler = filmler.order_by("-id")
        
        if q:
            filmler = filmler.filter(isim__icontains=q).distinct()

        return render(request, "anasayfa.html", {
            "mode": "search",
            "filmler": filmler,
            "q": q,
            "secili_kategori": kategori,
        })

    # DASHBOARD MODU (Netflix Style)
    else:
        # 1. Hero Section: Fragmanı olan & Puanı > 6.5 olan filmler (Carousel için)
        # Filtre: Fragman var VE Puan > 6.5
        hero_candidates = list(Film.objects.filter(
            Q(fragman_url__isnull=False) & ~Q(fragman_url="") & Q(puan__gt=6.5)
        ))
        
        # Rastgele 5 tane seç (yoksa hepsini al)
        hero_movies = random.sample(hero_candidates, min(len(hero_candidates), 5))

        # HD Poster URL oluştur (w500 vb. -> original)
        import re
        for movie in hero_movies:
            if movie.poster_url:
                # /w500/, /w300/, etc. -> /original/
                movie.poster_url_hd = re.sub(r'/w\d+/', '/original/', movie.poster_url)
            else:
                movie.poster_url_hd = ""

        # 2. Popüler (IMDb Puanına Göre)
        popular_movies = Film.objects.all().order_by("-puan")[:15]

        # 3. Yeni Eklenenler (ID'ye göre)
        new_arrivals = Film.objects.all().order_by("-id")[:15]

        # 4. Kategori Bazlı Listeler
        action_movies = Film.objects.filter(turler__icontains="Aksiyon")[:10]
        comedy_movies = Film.objects.filter(turler__icontains="Komedi")[:10]
        horror_movies = Film.objects.filter(turler__icontains="Korku")[:10]

        return render(request, "anasayfa.html", {
            "mode": "dashboard",
            "hero_movies": hero_movies,
            "popular": popular_movies,
            "newest": new_arrivals,
            "action": action_movies,
            "comedy": comedy_movies,
            "horror": horror_movies,
        })


# --- 3. KAYIT OLMA SİSTEMİ ---
def kayit_ol(request):
    """Yeni kullanıcı kayıt sayfası."""
    if request.user.is_authenticated:
        return redirect("anasayfa")

    if request.method == "POST":
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(
                request,
                f"Hesabınız başarıyla oluşturuldu, {user.username}! Giriş yapabilirsiniz.",
            )
            return redirect("login")
        else:
            messages.error(request, "Kayıt hatası. Bilgileri kontrol edin.")
    else:
        form = UserRegisterForm()

    return render(request, "registration/register.html", {"form": form})


# --- 4. FİLM DETAY VE AI ANALİZ ---
@login_required
def film_detay(request, film_id):
    """Film detay sayfası: bilgiler, fragman, AI duygu analizi."""
    film = get_object_or_404(Film, id=film_id)

    # --- AJAX mi kontrol ---
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    # Yorum Gönderme
    if request.method == "POST":
        gelen_yorum = request.POST.get("yorum_icerigi", "").strip()

        # 🚫 Küfür kontrolü
        if kufur_kontrol(gelen_yorum):
            msg = "⛔ Yorumunuz uygunsuz ifade içeriyor. Lütfen saygılı bir dil kullanın."
            if is_ajax:
                return JsonResponse({"ok": False, "error": msg}, status=400)
            messages.error(request, msg)
            return redirect("film_detay", film_id=film.id)

        if len(gelen_yorum) < 2:
            msg = "Yorum çok kısa."
            if is_ajax:
                return JsonResponse({"ok": False, "error": msg}, status=400)
            messages.warning(request, msg)
            return redirect("film_detay", film_id=film.id)

        # 🚫 Anlamsız metin kontrolü
        if anlamsiz_mi(gelen_yorum):
            msg = "⛔ Yorumunuz anlamlı bir metin içermiyor. Lütfen gerçek bir yorum yazın."
            if is_ajax:
                return JsonResponse({"ok": False, "error": msg}, status=400)
            messages.error(request, msg)
            return redirect("film_detay", film_id=film.id)

        # AI Analiz
        sentiment_result = analyze_comment(gelen_yorum)
        ai_karar = sentiment_result["decision"]
        ai_guven = sentiment_result["confidence"]
        ai_kaynak = sentiment_result["source"]

        # Yorum Kaydet
        yeni_yorum = Yorum.objects.create(
            film=film,
            kullanici_adi=request.user.username,
            icerik=gelen_yorum,
            ai_karari=ai_karar,
            ai_guveni=ai_guven,
            ai_kaynak=ai_kaynak,
        )

        if is_ajax:
            # Badge bilgisi
            badge = get_sentiment_badge(ai_karar)

            # Güncel istatistikler
            yorumlar_qs = film.yorumlar.all()
            total = yorumlar_qs.count()
            pos = yorumlar_qs.filter(ai_karari="OLUMLU").count()
            neg = yorumlar_qs.filter(ai_karari="OLUMSUZ").count()
            neu = yorumlar_qs.filter(ai_karari="NÖTR").count()

            return JsonResponse({
                "ok": True,
                "yorum": {
                    "kullanici": yeni_yorum.kullanici_adi,
                    "tarih": yeni_yorum.tarih.strftime("%d %b %Y"),
                    "icerik": yeni_yorum.icerik,
                    "ai_karari": ai_karar,
                    "badge_text": badge["text"],
                    "badge_cls": badge["cls"],
                    "sent_key": badge["sent"],
                    "avatar": yeni_yorum.kullanici_adi[0].upper(),
                },
                "stats": {
                    "pos": pos, "neg": neg, "neu": neu, "total": total,
                    "pos_pct": round(pos / total * 100) if total else 0,
                    "neg_pct": round(neg / total * 100) if total else 0,
                    "neu_pct": round(neu / total * 100) if total else 0,
                },
            })

        messages.success(request, "Yorumunuz kaydedildi.")
        return redirect("film_detay", film_id=film.id)

    # Yorumlar & İstatistikler
    yorumlar = film.yorumlar.all().order_by("-tarih")
    total = yorumlar.count()
    pos = yorumlar.filter(ai_karari="OLUMLU").count()
    neg = yorumlar.filter(ai_karari="OLUMSUZ").count()
    neu = yorumlar.filter(ai_karari="NÖTR").count()

    stats = {
        "pos": pos,
        "neg": neg,
        "neu": neu,
        "total": total,
        "pos_pct": round(pos / total * 100) if total else 0,
        "neg_pct": round(neg / total * 100) if total else 0,
        "neu_pct": round(neu / total * 100) if total else 0,
    }

    # TMDB Ek Bilgiler
    tmdb_data = fetch_movie_details(film.isim, existing_trailer_url=film.fragman_url)

    return render(
        request,
        "detay.html",
        {
            "film": film,
            "yorumlar": yorumlar,
            "stats": stats,
            "runtime_minutes": tmdb_data["runtime"],
            "genres_text": tmdb_data["genres"],
            "backdrop_url": tmdb_data["backdrop_url"],
            "cast_list": tmdb_data["cast_list"],
            "trailer_watch_url": tmdb_data["trailer_watch_url"],
        },
    )


# --- 5. TOPLU FİLM EKLEME (YEDEK) ---
@staff_member_required
def toplu_film_ekle(request):
    """Admin'i terminaldeki manage.py komutuna yönlendirir."""
    messages.warning(
        request, "Lütfen bu işlem için terminalden 'python manage.py film_cek' komutunu kullanın."
    )
    return redirect("anasayfa")



