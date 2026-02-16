import re

# =============================================
# 🚫 KÜFÜR FİLTRESİ
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
