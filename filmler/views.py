import logging
import re

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.db.models import Q

from .forms import UserRegisterForm
from .models import Film, Yorum
from .services.moderation_service import kufur_kontrol, anlamsiz_mi
from .services.sentiment_service import analyze_comment, get_sentiment_badge
from .services.tmdb_service import fetch_movie_details

logger = logging.getLogger(__name__)

W_SIZE_RE = re.compile(r'/w\d+/')


# --- 1. CANLI ARAMA API (SPOTIFY TARZI) ---
def live_search(request):
    """Navbar'daki anlık arama kutusu için JSON döner."""
    term = request.GET.get("term", "").strip()
    results = []

    if len(term) > 1:
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
def anasayfa(request):
    """
    Ana sayfa (public):
    - Varsayılan: Dashboard Modu (Hero, Popüler, Yeni, Kategoriler)
    - Arama/Filtre: Grid Modu
    """
    q = request.GET.get("q", "").strip()
    kategori = request.GET.get("kategori", "")
    sirala = request.GET.get("sirala", "")
    mode_param = request.GET.get("mode", "")

    # FİLTRELEME MODU
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

    # DASHBOARD MODU
    hero_movies = list(
        Film.objects.filter(
            Q(fragman_url__isnull=False) & ~Q(fragman_url="") & Q(puan__gt=7.5)
        ).order_by("?")[:5]
    )
    for movie in hero_movies:
        movie.poster_url_hd = (
            W_SIZE_RE.sub('/original/', movie.poster_url) if movie.poster_url else ""
        )

    # Kategori section'ları (sırasıyla)
    sections = [
        ("popular", "En Yüksek Puanlılar", "fa-star",
            Film.objects.filter(puan__gt=0).order_by("-puan")[:18]),
        ("newest", "Yeni Eklenenler", "fa-bolt",
            Film.objects.all().order_by("-id")[:18]),
        ("action", "Aksiyon", "fa-fire",
            Film.objects.filter(turler__icontains="Aksiyon").order_by("-puan")[:18]),
        ("drama", "Dram", "fa-masks-theater",
            Film.objects.filter(turler__icontains="Dram").order_by("-puan")[:18]),
        ("comedy", "Komedi", "fa-face-laugh-squint",
            Film.objects.filter(turler__icontains="Komedi").order_by("-puan")[:18]),
        ("scifi", "Bilim Kurgu", "fa-rocket",
            Film.objects.filter(turler__icontains="Bilim-Kurgu").order_by("-puan")[:18]),
        ("thriller", "Gerilim", "fa-bolt-lightning",
            Film.objects.filter(turler__icontains="Gerilim").order_by("-puan")[:18]),
        ("horror", "Korku", "fa-ghost",
            Film.objects.filter(turler__icontains="Korku").order_by("-puan")[:18]),
        ("animation", "Animasyon", "fa-child-reaching",
            Film.objects.filter(turler__icontains="Animasyon").order_by("-puan")[:18]),
        ("adventure", "Macera", "fa-mountain",
            Film.objects.filter(turler__icontains="Macera").order_by("-puan")[:18]),
        ("romance", "Romantik", "fa-heart",
            Film.objects.filter(turler__icontains="Romantik").order_by("-puan")[:18]),
    ]

    return render(request, "anasayfa.html", {
        "mode": "dashboard",
        "hero_movies": hero_movies,
        "sections": sections,
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
        messages.error(request, "Kayıt hatası. Bilgileri kontrol edin.")
    else:
        form = UserRegisterForm()

    return render(request, "registration/register.html", {"form": form})


# --- 4. FİLM DETAY VE AI ANALİZ ---
def film_detay(request, film_id):
    """Film detay sayfası: bilgiler, fragman, AI duygu analizi.
    Görüntüleme public; yorum gönderme login gerektirir.
    """
    film = get_object_or_404(Film, id=film_id)
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    if request.method == "POST":
        # Yorum gönderme login zorunlu
        if not request.user.is_authenticated:
            msg = "Yorum yapabilmek için giriş yapmalısınız."
            if is_ajax:
                return JsonResponse({"ok": False, "error": msg, "login_required": True}, status=401)
            messages.warning(request, msg)
            return redirect("login")

        gelen_yorum = request.POST.get("yorum_icerigi", "").strip()

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

        if anlamsiz_mi(gelen_yorum):
            msg = "⛔ Yorumunuz anlamlı bir metin içermiyor. Lütfen gerçek bir yorum yazın."
            if is_ajax:
                return JsonResponse({"ok": False, "error": msg}, status=400)
            messages.error(request, msg)
            return redirect("film_detay", film_id=film.id)

        sentiment_result = analyze_comment(gelen_yorum)
        ai_karar = sentiment_result["decision"]
        ai_guven = sentiment_result["confidence"]
        ai_kaynak = sentiment_result["source"]

        yeni_yorum = Yorum.objects.create(
            film=film,
            kullanici_adi=request.user.username,
            icerik=gelen_yorum,
            ai_karari=ai_karar,
            ai_guveni=ai_guven,
            ai_kaynak=ai_kaynak,
        )

        if is_ajax:
            badge = get_sentiment_badge(ai_karar)
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

