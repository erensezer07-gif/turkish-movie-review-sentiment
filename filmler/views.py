import logging
import re
from datetime import datetime

import requests
from django.conf import settings
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

from .forms import UserRegisterForm
from .models import Film, Yorum

# ✅ AI client (Django -> FastAPI)
# Dosya yoksa oluştur: sinema_sitesi/ai_client.py
try:
    from sinema_sitesi.ai_client import analiz_yap
except Exception:
    analiz_yap = None

logger = logging.getLogger(__name__)

# TMDB Ayarları
TMDB_API_KEY = getattr(settings, "TMDB_API_KEY", "")

# AI Ayarları (settings.py içine eklemen önerilir)
AI_API_URL = getattr(settings, "AI_API_URL", "http://127.0.0.1:8001")
AI_API_TIMEOUT = getattr(settings, "AI_API_TIMEOUT", 10)

# =============================================
# 🚫 KÜFÜR FİLTRESİ — yorum kaydetmeden önce kontrol
# =============================================
KUFUR_LISTESI = {
    "yarrak", "yarak", "sik", "sikik", "sikerim", "sikim", "sikeyim", "siktir",
    "amk", "aq", "amq", "amına", "amina", "amcık", "amcik", "orospu", "oruspu",
    "orusbu", "orosbu", "piç", "pic", "pezevenk", "göt", "got", "götveren",
    "gavat", "ibne", "puşt", "pust", "kahpe", "kaltak", "sürtük", "surtuk",
    "fahişe", "fahise", "döl", "taşak", "tasak", "dalyarak",
    "yavşak", "yavsak", "haysiyetsiz", "şerefsiz", "serefsiz", "namussuz",
    "boktan", "osur", "sıç", "anan", "anana", "ananı", "anani",
    "bacını", "bacini", "avradını", "skm", "sg", "s2m", "mk",
}


def kufur_kontrol(text: str) -> bool:
    """Metin küfür/argo içeriyorsa True döner."""
    words = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", text.lower())
    for word in words:
        if word in KUFUR_LISTESI:
            return True
    return False


# Yaygın Türkçe kelimeler — gibberish tespiti için (sadece tek kelimelik yorumlarda kullanılır)
TURKCE_SOZLUK = {
    # yaygın kelimeler
    "bir", "bu", "de", "da", "ve", "çok", "iyi", "güzel", "film", "harika",
    "kötü", "mükemmel", "berbat", "izle", "izledim", "izlenmeli", "süper",
    "muhteşem", "fena", "olmamış", "olmuş", "bence", "ama", "için", "daha",
    "en", "hiç", "gibi", "kadar", "ile", "ben", "sen", "benim", "senin",
    "onun", "ne", "nasıl", "neden", "niye", "evet", "hayır", "yok", "var",
    "olan", "olumlu", "olumsuz", "pek", "gayet", "oldukça", "gerçekten",
    "kesinlikle", "tavsiye", "ederim", "etmem", "izlemeyin", "izleyin",
    "eser", "yapım", "oyunculuk", "senaryo", "müzik", "görsel", "efekt",
    "sahne", "oyuncu", "yönetmen", "başrol", "kadro", "hikaye", "konu",
    "sonuç", "final", "aksiyon", "komedi", "dram", "korku", "gerilim",
    "romantik", "animasyon", "belgesel", "klasik", "yeni", "eski",
    "heyecanlı", "sıkıcı", "eğlenceli", "duygusal", "etkileyici",
    "başarılı", "başarısız", "beğendim", "beğenmedim", "sevdim", "sevmedim",
    "eh", "işte", "fakat", "lakin", "hatta", "bile", "sade", "sadece",
    "her", "bazı", "tüm", "hep", "hiçbir", "şey", "şu", "o",
    "gel", "git", "yap", "al", "ver", "oku", "gör", "bak", "düşün",
    "tamam", "güzel", "kötüydü", "iyiydi", "zorunda", "gerek",
    # ek kelimeler — false positive önleme
    "yani", "yanii", "abi", "baya", "bayağı", "filan", "falan", "yaa", "yahu",
    "valla", "vallahi", "hani", "zaten", "aslında", "tabi", "tabii", "elbette",
    "bidaha", "tekrar", "izlemem", "muhtemelen", "belki", "olabilir", "olmaz",
    "beklenti", "karşıladı", "karşılamadı", "idare", "eder", "orta", "ortalama",
    "seksi", "fesat", "kapak", "efsane", "bomba", "şahane", "süperdi",
    "yetişkin", "çocuk", "herkes", "kimse", "hiçkimse", "biri", "birisi",
    # 2-3-4 harfli yaygın kelimeler (kısa spam koruması için)
    "az", "çok", "cok", "bir", "tek", "ilk", "son", "ön", "arka", "yan", "üst", "alt",
    "iç", "dış", "sol", "sag", "sağ", "boş", "bos", "dolu", "net", "saf", "mat",
    "ak", "al", "at", "ay", "aç", "ad", "ah", "an", "aş", "av", "az",
    "ba", "be", "bu", "ca", "ce", "cu", "çi", "çü", "da", "de", "do", "du",
    "eh", "ek", "el", "em", "en", "er", "es", "et", "ev", "ey", "fa", "fe",
    "ha", "he", "hu", "ıh", "ıl", "ın", "ır", "ıs", "ış", "iz", "il", "im",
    "in", "ir", "is", "iş", "it", "iv", "iy", "je", "ke", "ki", "ko", "ku",
    "la", "le", "li", "lo", "ma", "me", "mi", "mu", "ne", "ni", "no", "nu",
    "od", "of", "oh", "ok", "ol", "om", "on", "op", "or", "os", "ot", "oy",
    "öç", "öd", "of", "ög", "ök", "öl", "ön", "öp", "ör", "os", "öt", "öz",
    "pe", "pi", "pu", "ra", "re", "ro", "ru", "se", "si", "su", "sü",
    "ta", "te", "ti", "tu", "tü", "uc", "uç", "ud", "uf", "ug", "uh", "uj",
    "uk", "ul", "um", "un", "ur", "us", "uş", "ut", "uy", "uz", "üç", "üf",
    "ül", "ün", "ür", "üs", "üş", "üt", "üz", "ve", "ya", "ye", "yi", "yo",
    "yu", "yü", "za", "ze", "zı", "zi", "zo", "zu",
    "ada", "adi", "afi", "aha", "ahi", "ait", "aka", "aki", "aks", "ala",
    "ali", "alo", "alp", "alt", "ama", "ana", "ani", "ant", "ara", "ari",
    "ark", "arp", "art", "arz", "asa", "asi", "ask", "asl", "asu", "ata",
    "ate", "ati", "aut", "ava", "avi", "aya", "ayi", "ayn", "aza", "aze",
    "bad", "bağ", "bal", "ban", "bar", "bas", "baş", "bat", "bay", "baz",
    "bel", "ben", "beş", "bet", "bey", "bez", "bin", "bir", "bit", "biz",
    "boa", "bok", "bol", "bom", "bop", "bor", "boş", "boy", "boz", "bre",
    "bul", "bun", "buş", "bük", "bül", "büre", "büz", "can", "caz", "cem",
    "cep", "cer", "cık", "cıl", "cır", "cıs", "cız", "cik", "cim", "cin",
    "cip", "iri", "isi", "kot", "koy", "koz", "kök", "kör", "kös", "kot",
    "köy", "kuh", "kul", "kum", "kup", "kur", "kus", "kuş", "kut", "kuz",
    "küf", "küh", "kük", "kül", "küm", "kün", "küp", "kür", "küs", "küt",
    "laf", "lak", "lal", "lam", "lan", "lap", "las", "laş", "lav", "laz",
    "leb", "lef", "leh", "lek", "leş", "ley", "lığ", "lık", "lim", "lir",
    "lök", "lop", "lor", "loş", "lot", "lup", "lüp", "maç", "mai", "mal",
    "mas", "maş", "mat", "may", "men", "met", "mey", "mıh", "mık", "mıl",
    "mır", "mis", "mit", "mlk", "mor", "muç", "mum", "muş", "muz", "müç",
    "müd", "mül", "mün", "mür", "müş", "müz", "nal", "nam", "nan", "nar",
    "nas", "naş", "naz", "nem", "net", "nev", "ney", "nış", "niğ", "nil",
    "nim", "niş", "nod", "nof", "nom", "not", "nur", "oda", "odi", "ofl",
    "oje", "ole", "ol", "oma", "ona", "ons", "ora", "org", "ork", "orp",
    "ors", "ort", "oru", "ost", "otl", "oto", "oya", "oyn", "ozu", "öcü",
    "ödü", "öge", "öke", "ökü", "ölü", "önü", "örf", "örs", "ört", "örü",
    "öte", "oto", "ötü", "öyle", "özü", "pah", "pak", "pal", "pan", "pas",
    "pat", "pay", "paz", "pek", "pes", "peş", "pey", "pır", "pıt", "piç",
    "pik", "pil", "pim", "pin", "pip", "pir", "pis", "piş", "pof", "pop",
    "pos", "poş", "pot", "poy", "poz", "pör", "pös", "puf", "puh", "pul",
    "pus", "puş", "put", "püf", "pür", "püs", "püt", "rab", "raf", "rag",
    "rak", "ram", "ran", "rap", "rar", "raş", "ray", "raz", "red", "ref",
    "rej", "rek", "rem", "ren", "ret", "rey", "rez", "rıh", "rlg", "rol",
    "rom", "rop", "rot", "roz", "ruf", "ruh", "ruj", "rum", "run", "rus",
    "ruz", "rüç", "rük", "rüş", "sac", "saç", "saf", "sağ", "sah", "sak",
    "sal", "sam", "san", "sap", "sar", "sat", "sav", "say", "saz", "seç",
    "sed", "sef", "sek", "sel", "sem", "sen", "sep", "ser", "ses", "set",
    "sev", "sey", "sez", "sığ", "sık", "sın", "sır", "siv", "siz", "sis",
    "sit", "skı", "ski", "sof", "sol", "som", "son", "sop", "sor", "sos",
    "soy", "sör", "söz", "suç", "sud", "suf", "suk", "sul", "sum", "sun",
    "sup", "sur", "sus", "suş", "sut", "süt", "süy", "süz", "şad", "şah",
    "şak", "şal", "şam", "şan", "şap", "şar", "şas", "şat", "şaz", "şeb",
    "şef", "şeh", "şek", "şem", "şen", "şer", "şeş", "şet", "şev", "şey",
    "şıh", "şık", "şıp", "şiâ", "şia", "şif", "şii", "şik", "şim", "şip",
    "şiş", "şit", "şiv", "şok", "şom", "şor", "şov", "şoz", "şu", "şua",
    "şuh", "şut", "tab", "taç", "tağ", "tak", "tal", "tam", "tan", "tar",
    "tas", "taş", "tat", "tav", "tay", "taz", "tef", "tek", "tel", "tem",
    "ten", "ter", "tes", "teş", "tez", "tığ", "tık", "tın", "tıp", "tır",
    "tıs", "tiğ", "tik", "tim", "tin", "tip", "tir", "tiş", "tiz", "toğ",
    "tok", "tol", "ton", "top", "tor", "tos", "toy", "toz", "töh", "tör",
    "tös", "töz", "tuğ", "tuh", "tul", "tun", "tup", "tur", "tuş", "tut",
    "tuz", "tüf", "tüh", "tül", "tüm", "tün", "tüp", "tür", "tüs", "tüş",
    "tüy", "tüz", "uca", "uç", "udi", "udu", "ufk", "ufo", "ula", "ulu",
    "umm", "umk", "umu", "uni", "uns", "unu", "ura", "urg", "us", "usc",
    "usl", "usta", "uşk", "uta", "utu", "uy", "uza", "uzi", "uz", "üce",
    "üde", "üfe", "ügi", "üğü", "üke", "üle", "ülü", "üme", "ümi", "ünl",
    "ünü", "üre", "ürk", "ürü", "üse", "usk", "üst", "üşü", "üte", "ütü",
    "üve", "üye", "vab", "vah", "val", "van", "var", "vat", "vay", "vaz",
    "veb", "veç", "ver", "vet", "vey", "veZ", "vız", "viz", "vin", "vir",
    "vol", "vor", "vot", "voy", "voz", "vuh", "vuk", "vul", "vur", "vus",
    "vuz", "vüc", "yad", "yağ", "yah", "yak", "yal", "yam", "yan", "yap",
    "yar", "yas", "yaş", "yat", "yav", "yay", "yaz", "yeğ", "yek", "yel",
    "yem", "yen", "yer", "yes", "yeş", "yet", "yey", "yıl", "yır", "yiv",
    "yiy", "yob", "yod", "yoğ", "yok", "yol", "yom", "yon", "yor", "yos",
    "yoz", "yön", "yör", "yuf", "yuh", "yum", "yun", "yur", "yuş", "yut",
    "yuz", "yük", "yül", "yün", "yür", "yüs", "yüz", "zağ", "zah", "zam",
    "zan", "zar", "zat", "zav", "zay", "zaz", "zek", "zem", "zen", "zer",
    "zev", "zey", "zıh", "zık", "zıl", "zır", "zıt", "zil", "zin", "zir",
    "ziy", "ziz", "zor", "zoş", "zum", "züh", "zük", "zül", "züm", "zür",
}

BLACKLIST = {"asd", "qwe", "zxc", "jkl", "mnb", "ase", "asdf", "qwer", "zxcv", "tyu", "ghj", "bnm", "qaz", "wsx", "edc", "rfv", "as", "sa"}

def anlamsiz_mi(text: str) -> bool:
    """
    Rastgele harf dizilerini ve spam yorumları tespit eder (Gelişmiş).
    Kurallar:
    1. Normalizasyon: "süperrrr" -> "süperr"
    2. Repetitive Substring: "asdasdasd" -> SPAM
    3. Rule A: 4+ Ardışık Ünsüz -> SPAM (dmşkamk)
    4. Rule B: >3 Harf ve HİÇ ÜNLÜ YOKSA -> SPAM
    5. Rule C: Kelime içi noktalama -> SPAM (s.a.l.a.k)
    """
    # Orijinal metni koru (Rule C için)
    # Temizlik (noktalama işaretlerini kaldır, sadece harf ve boşluk bırak)
    # Ancak önce Rule C kontrolü yapmalıyız
    
    # Rule C: Punctuation Abuse (Kelime içi nokta/sembol)
    # Örn: "s.a.l.a.k", "a.mk" (Harf.Harf)
    if re.search(r'[a-zA-ZçğıöşüÇĞİÖŞÜ]\.[a-zA-ZçğıöşüÇĞİÖŞÜ]', text):
        return True

    # Temizlik
    clean = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ ]", "", text.lower()).strip()
    
    if len(clean) < 2:
        return True 

    if clean in BLACKLIST:
        return True

    # 1. Expressive Lengthening Normalization
    norm = re.sub(r'(.)\1{2,}', r'\1\1', clean)

    # 2. Repetitive Substring Check (Klavye Ezme)
    if re.search(r'(.{2,})\1{2,}', norm):
        return True

    words = norm.split()
    unluler = set("aeıioöuü")
    # Rule A Regex (Turkish Consonants 4+)
    # [bcçdfgğhjklmnprsştvyz]
    cons_cluster_re = re.compile(r'[bcçdfgğhjklmnprsştvyz]{4,}')

    for w in words:
        # Rule A: Consonant Clusters
        if cons_cluster_re.search(w):
            return True
            
        # Rule B: Vowel Absence (>3 chars)
        if len(w) > 3 and not any(c in unluler for c in w):
            return True

    # 3. Space/Word Count Heuristic
    if len(words) > 1:
        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len < 15:
            return False 

    # 4. Tek Kelime Analizi (Strict)
    if len(words) == 1:
        w = words[0]
        if len(w) <= 4:
            if w not in TURKCE_SOZLUK and w not in BLACKLIST: 
                return True
        return False
        
    return False


# --- TÜRKÇE KARAKTER DÜZELTİCİ ---
def tr_lower(text):
    """Türkçe karakter sorununu (I-ı, İ-i, Ş-ş) çözen fonksiyon."""
    if not text:
        return ""
    kucuk_harf_tablosu = {
        ord("I"): "ı",
        ord("İ"): "i",
        ord("Ş"): "ş",
        ord("Ğ"): "ğ",
        ord("Ü"): "ü",
        ord("Ö"): "ö",
        ord("Ç"): "ç",
    }
    return text.translate(kucuk_harf_tablosu).lower()


# --- 1. CANLI ARAMA API (SPOTIFY TARZI) ---
def live_search(request):
    """Navbar'daki anlık arama kutusu için JSON döner."""
    term = request.GET.get("term", "").strip()
    results = []

    if len(term) > 1:
        filmler = Film.objects.all().order_by("-id")[:200]  # performans
        term_lower = tr_lower(term)

        for film in filmler:
            if term_lower in tr_lower(film.isim):
                results.append(
                    {
                        "id": film.id,
                        "isim": film.isim,
                        "poster": film.poster_url,
                        "yil": film.yil,
                        "puan": film.puan,
                    }
                )
            if len(results) >= 10:
                break

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
        import random
        from django.db.models import Q
        
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

        # 🚫 Anlamsız metin kontrolü
        if anlamsiz_mi(gelen_yorum):
            msg = "⛔ Yorumunuz anlamlı bir metin içermiyor. Lütfen gerçek bir yorum yazın."
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

        # Default (AI servis yoksa bile yorum kaydolur)
        ai_karar = "NÖTR"
        ai_guven = 0.0
        ai_kaynak = None

        # API label → DB format normalizasyonu
        KARAR_NORMALIZE = {
            "Olumlu": "OLUMLU", "olumlu": "OLUMLU", "OLUMLU": "OLUMLU",
            "Olumsuz": "OLUMSUZ", "olumsuz": "OLUMSUZ", "OLUMSUZ": "OLUMSUZ",
            "Kararsız": "NÖTR", "kararsız": "NÖTR", "NÖTR": "NÖTR", "Nötr": "NÖTR",
        }

        # AI Analiz
        try:
            if analiz_yap is None:
                raise RuntimeError("ai_client import edilemedi.")
            sonuc = analiz_yap(gelen_yorum)
            raw_karar = sonuc.get("karar", "NÖTR")
            ai_karar = KARAR_NORMALIZE.get(raw_karar, "NÖTR")
            ai_guven = float(sonuc.get("guven_skoru", 0.0))
            ai_kaynak = sonuc.get("kaynak")
            logger.info("AI sonuç: raw=%s → normalized=%s güven=%.4f kaynak=%s", raw_karar, ai_karar, ai_guven, ai_kaynak)
        except Exception as e:
            logger.warning("AI servisine bağlanılamadı: %s | AI_API_URL=%s", e, AI_API_URL)
            ai_kaynak = "api_error"

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
            # Badge bilgisi (Standardized)
            BADGE_MAP = {
                "OLUMLU": {"text": "Olumlu", "cls": "badge bg-success text-white p-2", "sent": "pos"},
                "OLUMSUZ": {"text": "Olumsuz", "cls": "badge bg-danger text-white p-2", "sent": "neg"},
            }
            badge = BADGE_MAP.get(ai_karar, {"text": "Nötr", "cls": "badge bg-secondary text-white p-2", "sent": "neu"})

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
    runtime, genres = None, None
    backdrop_url, cast_list, trailer_watch_url = None, [], None

    if TMDB_API_KEY:
        try:
            search_res = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={"api_key": TMDB_API_KEY, "language": "tr-TR", "query": film.isim},
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
    if not trailer_watch_url and film.fragman_url:
        match = re.search(r"youtube\.com/embed/([a-zA-Z0-9_-]+)", film.fragman_url)
        if match:
            # ?autoplay=1 ekleyerek tıklanınca başlamasını sağlıyoruz
            trailer_watch_url = f"https://www.youtube.com/embed/{match.group(1)}?autoplay=1&rel=0"

    return render(
        request,
        "detay.html",
        {
            "film": film,
            "yorumlar": yorumlar,
            "stats": stats,
            "runtime_minutes": runtime,
            "genres_text": genres,
            "backdrop_url": backdrop_url,
            "cast_list": cast_list,
            "trailer_watch_url": trailer_watch_url,
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


def live_search(request):
    """AJAX canlı arama endpoint'i."""
    term = request.GET.get("term", "").strip()
    if len(term) < 2:
        return JsonResponse([], safe=False)

    # İlk 5 eşleşen filmi getir (Sadece Film Adı)
    results_qs = Film.objects.filter(isim__icontains=term)[:5]

    data = []
    for f in results_qs:
        data.append({
            "id": f.id,
            "isim": f.isim,
            "yil": f.yil,
            "puan": f.puan,
            "poster": f.poster_url,
        })
    
    return JsonResponse(data, safe=False)
