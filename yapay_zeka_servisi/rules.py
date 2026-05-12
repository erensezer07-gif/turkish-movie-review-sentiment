"""Kural tabanlı sentiment / metin kontrolleri.

Bu modül BERT veya TF-IDF gerektirmez; saf metin işleme yapar.
Guardrail, nötr tespit, ironi/sarcasm split, küfür-vari sınır kontrolleri burada.
"""
import re
import difflib

MAX_TEXT_LENGTH = 5000

# ---------------------------------------------------------------------
# 1) HINT / PHRASE KÜTÜPHANELERİ
# ---------------------------------------------------------------------
NEG_HINTS = [
    "rezalet", "berbat", "iğrenç", "igrenc", "çöp", "cop", "bok", "kaka",
    "sakın", "sakin", "pişman", "pisman", "fiyasko", "zıkkım", "saçma",
    "boş", "bos", "dandik", "izlenilmez", "kaçın", "kacin", "sıkıcı",
    "sikici", "bayık", "bayik", "uykumu", "vakit_kaybi", "zaman_kaybi",
    "yetersiz", "sığ", "sig", "amatör", "amator", "beceriksiz", "tırt",
    "tirt", "leş", "lez", "kusturucu", "işkence", "iskence", "zulüm",
    "zulum", "katlanılmaz", "ucuz", "basit", "facia", "kepaze", "yavan",
]
POS_HINTS = [
    "mükemmel", "mukemmel", "şaheser", "saheser", "efsane", "bayıldım",
    "harika", "başyapıt", "basyapit", "müthiş", "muthis", "harikulade",
    "tapıyorum", "mutlaka", "soluksuz", "sürükleyici", "surukleyici",
    "masterpiece", "şahane", "sahane", "epik", "kült", "kult", "sarsıcı",
    "vurucu", "derinlikli", "vizyoner", "doyurucu", "fevkalade",
    "kusursuz", "enfes", "şaşırtıcı", "sasirtici", "büyüleyici",
    "buyuleyici", "döktürmüş",
]

NEGATION_TOKENS_RAW = {
    "degil", "değil", "yok", "hic", "hiç", "asla", "katiyen", "maalesef",
    "olmaz", "olamaz", "hicbir", "hiçbir",
}
NEGATION_PHRASES_RAW = {"bile degil", "bile değil"}

NEG_EMOJIS = ["💩", "👎", "🤮", "🤬", "😡", "😠", "🤢", "😴", "🤦", "😤", "📉", "🗑️"]
POS_EMOJIS = ["💯", "🔥", "👍", "❤️", "😍", "🥰", "⭐", "✨", "👏", "🙌", "🤩", "🚀", "🍿", "👑"]

NEG_PHRASES = [
    "sakın gitmeyin", "zaman kaybı", "zaman kaybi", "vakit kaybı", "vakit kaybi",
    "param haram olsun", "haram olsun", "izlemeyin", "izlenmez", "uzak durun",
    "yanına bile yaklaşmayın", "berbat ötesi", "rezalet ötesi", "izlemeye değmez",
    "izlemeye degmez", "pişman oldum", "pisman oldum", "sıkıldım izlerken",
    "sikildim izlerken", "vaktimi çaldı", "vaktimi caldi", "berbat bir film",
    "rezil bir film", "en kötü film", "en kotu film", "tam bir hayal kırıklığı",
    "tam bir hayal kirikligi", "beş para etmez", "bes para etmez", "yarıda bıraktım",
    "yarida biraktim", "sonu saçmaydı", "sonu sacmaydi", "senaryo çok kötü",
    "senaryo cok kotu", "hiç beğenmedim", "hic begenmedim", "hayatımdan çalınan",
    "hayatimdan calinan", "tahammül edemedim", "tahammul edemedim", "içim şişti",
    "icim sisti", "ruhum daraldı", "ruhum daraldi", "gözlerim kanadı", "gozlerim kanadi",
    "beyin yakan saçmalık", "beyin yakan sacmalik", "mantık hatası dolu",
    "mantik hatasi dolu", "oyunculuklar yerlerde", "efektler berbat", "kurgu felaket",
    "tavsiye etmem",
]
POS_PHRASES = [
    "kesinlikle izleyin", "mutlaka izleyin", "kaçırmayın", "kacirmayin", "kaçırma",
    "kacirma", "defalarca izlenir", "herkese tavsiye ederim", "şiddetle tavsiye",
    "siddetle tavsiye", "arşivlik", "arsivlik", "tekrar izleyeceğim", "tekrar izleyecegim",
    "şans verin", "sans verin", "sakın kaçırmayın", "sakin kacirmayin", "favorim oldu",
    "efsane olmuş", "efsane olmus", "muhteşem bir film", "muhtesem bir film",
    "hayran kaldım", "hayran kaldim", "en iyi filmlerden", "favorilerime eklendi",
    "şimdiye kadar izlediğim en iyi", "simdiye kadar izledigim en iyi", "oyunculuk harika",
    "senaryo mükemmel", "senaryo mukemmel", "görüntü yönetmeni döktürmüş",
    "goruntu yonetmeni dokturbus", "oyunculuk resitali", "senaryo çok zekice",
    "senaryo cok zekice", "ters köşe", "ters kose", "sonu mükemmeldi", "sonu mukemmeldi",
    "etkisinden çıkamadım", "etkisinden cikamadim", "ağzımız açık izledik", "agzimiz acik izledik",
    "soluksuz izledim", "gözünü kırpmadan", "gozunu kirpmadan", "su gibi aktı", "su gibi akti",
    "10 numara", "yıldızlı pekiyi", "yildizli pekiyi", "tek kelimeyle muazzam",
    "ayakta alkışlanacak", "ayakta alkislanacak",
]

NEUTRAL_STRICT_PHRASES = [
    "ne iyi ne kotu", "ne cok iyi ne cok kotu", "ne iyi ne de kotu", "ne cok iyi ne de kotu",
    "eh iste", "orta karar",
]
NEUTRAL_SOFT_PHRASES = [
    "ortalama", "vasat", "idare eder", "normal", "soyle boyle", "siradan", "standart",
    "kararsizim", "kotu degil ama iyi de degil", "begenmedim ama kotu degil",
    "mukemmel degil ama kotu de degil", "cok beklentiye girmeyin", "beklentimin altinda",
    "beklentimi karsilamadi", "cerezlik", "vakit gecirmelik", "kafa dagitmalik",
    "yoklukta gider", "pazar sinemasi", "tv filmi tadinda", "bos vakitte izlenir",
    "izlenir ama", "tek seferlik", "bir kere izlenir", "etki yaratmiyor", "akilda kalici degil",
    "iz birakmiyor", "iz birakacak bir etki yaratmiyor", "fikir guzel ama uygulama zayif",
    "potansiyeli harcanmis", "guzel basladi kotu bitti", "iyi basladi ama", "sonu haric",
    "biraz sikici", "fena degildi ama", "guzeldi ama", "abartildigi kadar degil",
    "klise dolu", "siradan bir yapim",
]

CONTRAST_TOKENS = {"ama", "fakat", "ancak", "lakin", "ragmen", "rağmen", "halde"}
CONTRAST_PHRASES = {"yine de", "buna ragmen", "buna rağmen"}

MILD_POS_PHRASES = [
    "fena degil", "fena değil", "fena degildi", "fena değildi", "iyiydi", "guzeldi", "güzeldi",
    "iyi sayilir", "kotu degildi", "kötü değildi", "oyunculuklar iyi", "cekimler guzel",
    "çekimler güzel", "muzikler guzel", "müzikler güzel", "goruntu guzel", "görüntü güzel",
    "fikir guzel", "baslangic iyi", "başlangıç iyi", "atmosfer iyi", "potansiyel var",
    "kurgu iyi", "konu guzel", "mekanlar guzel", "kostumler iyi", "kostümler iyi",
    "efektler iyi", "kotu sayilmaz", "izlenir",
]
MILD_NEG_WORDS = {
    "zayif", "zayıf", "eksik", "sikici", "sıkıcı", "uzun", "yorucu", "vasat", "ortalama",
    "kotu", "kötü", "bayik", "bayık", "sacma", "saçma", "kopuk", "yavas", "yavaş",
    "siradan", "sıradan", "klise", "mantiksiz", "mantıksız", "tutarsiz", "tutarsız",
    "basit", "olmamis", "olmamış", "yapay", "donuk", "abarti", "abartı", "zorlama",
    "tahmin edilebilir", "heyecansiz", "heyecansız", "durgun", "sarkmis", "sarkmış",
    "tempo dusuk", "tempo düşük", "inandirici degil", "inandırıcı değil", "finali kotu",
    "finali kötü",
}

FUZZY_STRICT = 0.90
FUZZY_RELAXED = 0.80

# Opsiyonel BERT neutral band (kapalı önerilir)
BERT_NEUTRAL_LOW = 0.45
BERT_NEUTRAL_HIGH = 0.65

# ---------------------------------------------------------------------
# 2) SARCASM
# ---------------------------------------------------------------------
SARCASM_MARKERS = [
    "saka yapiyorum", "saka yapıyorum", "şaka yapıyorum", "şaka yapiyorum",
    "ironi", "ironiydi", "ironiydi ya",
    "tabii ki", "tabi ki", "saka maka",
]

SARCASM_NEG_CUES = [
    "salonu terk", "terk ett", "ciktim", "çıktım", "cikmak",
    "zor tuttum", "dayanamadim", "katlanamadim",
    "berbat", "rezalet", "sıkıcı", "sikici", "iğrenç", "igrenc",
    "vakit kayb", "zaman kayb", "pişman", "pisman", "cop", "çöp",
]

# ---------------------------------------------------------------------
# 3) HELPERS
# ---------------------------------------------------------------------
_TR_MAP = str.maketrans({
    "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "ı": "i", "I": "i", "İ": "i",
    "ö": "o", "Ö": "o", "ş": "s", "Ş": "s", "ü": "u", "Ü": "u",
})


def rule_clean(text: str) -> str:
    if text is None:
        return ""
    t = str(text).lower().translate(_TR_MAP)
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _split_phrases(phrases):
    single = set()
    multi = []
    for ph in phrases:
        ph = (ph or "").strip()
        if not ph:
            continue
        if " " in ph:
            multi.append(ph)
        else:
            single.add(ph)
    return single, multi


def _has_any_phrase(clean: str, toks: list, single_set: set, multi_list: list) -> bool:
    if single_set:
        for t in toks:
            if t in single_set:
                return True
    if multi_list:
        for ph in multi_list:
            if ph in clean:
                return True
    return False


# Precomputed sets
R_NEG_HINTS = [rule_clean(x) for x in NEG_HINTS if rule_clean(x)]
R_POS_HINTS = [rule_clean(x) for x in POS_HINTS if rule_clean(x)]
NEG_SET = set(R_NEG_HINTS)
POS_SET = set(R_POS_HINTS)

R_NEG_PHRASES = [rule_clean(x) for x in NEG_PHRASES if rule_clean(x)]
R_POS_PHRASES = [rule_clean(x) for x in POS_PHRASES if rule_clean(x)]
R_NEUTRAL_STRICT = [rule_clean(x) for x in NEUTRAL_STRICT_PHRASES if rule_clean(x)]
R_NEUTRAL_SOFT = [rule_clean(x) for x in NEUTRAL_SOFT_PHRASES if rule_clean(x)]
R_MILD_POS_PHRASES = [rule_clean(x) for x in MILD_POS_PHRASES if rule_clean(x)]
R_CONTRAST_PHRASES = {rule_clean(x) for x in CONTRAST_PHRASES if rule_clean(x)}

NEG_PH_SINGLE, NEG_PH_MULTI = _split_phrases(R_NEG_PHRASES)
POS_PH_SINGLE, POS_PH_MULTI = _split_phrases(R_POS_PHRASES)
NEU_STR_SINGLE, NEU_STR_MULTI = _split_phrases(R_NEUTRAL_STRICT)
NEU_SFT_SINGLE, NEU_SFT_MULTI = _split_phrases(R_NEUTRAL_SOFT)
MILD_POS_SINGLE, MILD_POS_MULTI = _split_phrases(R_MILD_POS_PHRASES)

NEGATION_TOKENS = {rule_clean(x) for x in NEGATION_TOKENS_RAW if rule_clean(x)}
NEGATION_PHRASES = {rule_clean(x) for x in NEGATION_PHRASES_RAW if rule_clean(x)}

NE_NE_REGEX_1 = re.compile(r"\bne\s+(cok\s+)?iyi\w*\s+ne(\s+(de|da))?\s+(cok\s+)?kotu\w*\b")
NE_NE_REGEX_2 = re.compile(r"\bne\s+(cok\s+)?kotu\w*\s+ne(\s+(de|da))?\s+(cok\s+)?iyi\w*\b")
NE_GENERIC_REGEX = re.compile(r"\bne\b.{0,80}\bne(\s+de)?\b")

NOT_BAD_BUT_RE = re.compile(r"\b(kotu|fena|berbat|rezalet)\b.*?\bdegil\b.*?\b(ama|fakat|ancak|lakin|yine\s+de)\b")
GOOD_BUT_RE = re.compile(
    r"\b(iyi|guzel|harika|basarili|surukleyici|atmosfer|muzik|muzikler|gorsel\w*|oyuncu\w*|efekt\w*)\b.*?\b(ama|fakat|ancak|lakin|yine\s+de|buna\s+ragmen)\b.*?\b(zayif|eksik|sikici|uzun|yavas|kopuk|siradan|vasat|tikan\w*|imkansiz|yoksun|dusuk|dustu|zor|anlamsiz)\b"
)


def negation_near(toks, i, window=2) -> bool:
    left = max(0, i - window)
    right = min(len(toks), i + window + 1)
    window_toks = toks[left:right]
    if any(t in NEGATION_TOKENS for t in window_toks):
        return True
    window_text = " ".join(window_toks)
    if any(ph in window_text for ph in NEGATION_PHRASES):
        return True
    return False


def check_guardrails(text: str, cutoff=0.85):
    """Döndürür: 'neg' | 'pos' | 'neutral' | 'conflict' | None"""
    raw = "" if text is None else str(text)
    neg_e = any(e in raw for e in NEG_EMOJIS)
    pos_e = any(e in raw for e in POS_EMOJIS)
    if neg_e and pos_e:
        return "conflict"
    if neg_e:
        return "neg"
    if pos_e:
        return "pos"

    clean = rule_clean(raw)
    toks = clean.split()

    neu_strict = (
        _has_any_phrase(clean, toks, NEU_STR_SINGLE, NEU_STR_MULTI)
        or bool(NE_NE_REGEX_1.search(clean))
        or bool(NE_NE_REGEX_2.search(clean))
    )
    neu_soft = _has_any_phrase(clean, toks, NEU_SFT_SINGLE, NEU_SFT_MULTI)
    neu_p = neu_strict or neu_soft

    neg_p = _has_any_phrase(clean, toks, NEG_PH_SINGLE, NEG_PH_MULTI)
    pos_p = _has_any_phrase(clean, toks, POS_PH_SINGLE, POS_PH_MULTI)

    hits = int(neu_p) + int(neg_p) + int(pos_p)
    if hits >= 2:
        return "conflict"

    if neg_p:
        return "neg"
    if pos_p:
        return "pos"
    if neu_p:
        return "neutral"

    neg_found = False
    pos_found = False
    for i, w in enumerate(toks):
        if w in NEG_SET and not negation_near(toks, i, window=2):
            neg_found = True
        if w in POS_SET and not negation_near(toks, i, window=2):
            pos_found = True
        if neg_found and pos_found:
            return "conflict"
    if neg_found:
        return "neg"
    if pos_found:
        return "pos"

    strict_thr = max(float(cutoff), FUZZY_STRICT)
    relaxed_thr = min(float(cutoff), FUZZY_RELAXED)
    for i, w in enumerate(toks):
        if len(w) < 4:
            continue
        threshold = strict_thr if len(w) < 6 else relaxed_thr
        for hint in R_NEG_HINTS:
            if difflib.SequenceMatcher(None, w, hint).ratio() >= threshold:
                if not negation_near(toks, i, window=2):
                    neg_found = True
                break
        for hint in R_POS_HINTS:
            if difflib.SequenceMatcher(None, w, hint).ratio() >= threshold:
                if not negation_near(toks, i, window=2):
                    pos_found = True
                break
        if neg_found and pos_found:
            return "conflict"

    if neg_found:
        return "neg"
    if pos_found:
        return "pos"
    return None


def has_soft_neutral_signal(text: str) -> bool:
    clean = rule_clean("" if text is None else str(text))
    toks = clean.split()
    return _has_any_phrase(clean, toks, NEU_SFT_SINGLE, NEU_SFT_MULTI)


def is_neutral_like(text: str, return_reason: bool = False):
    clean = rule_clean("" if text is None else str(text))
    toks = clean.split()
    if _has_any_phrase(clean, toks, NEU_STR_SINGLE, NEU_STR_MULTI):
        return (True, "neutral_strict_phrase") if return_reason else True
    if _has_any_phrase(clean, toks, NEU_SFT_SINGLE, NEU_SFT_MULTI):
        return (True, "soft_neutral_phrase") if return_reason else True
    if NE_GENERIC_REGEX.search(clean) or NE_NE_REGEX_1.search(clean) or NE_NE_REGEX_2.search(clean):
        return (True, "ne_ne") if return_reason else True
    if NOT_BAD_BUT_RE.search(clean):
        return (True, "not_bad_but") if return_reason else True
    if GOOD_BUT_RE.search(clean):
        return (True, "good_but") if return_reason else True

    has_contrast = any(w in toks for w in CONTRAST_TOKENS) or any(ph in clean for ph in R_CONTRAST_PHRASES)
    if not has_contrast:
        return (False, None) if return_reason else False

    pos_hit = _has_any_phrase(clean, toks, MILD_POS_SINGLE, MILD_POS_MULTI) or any(t in toks for t in POS_SET)
    neg_hit = any(t in toks for t in MILD_NEG_WORDS) or any(t in toks for t in NEG_SET)
    ok = bool(pos_hit and neg_hit)
    return (ok, "mixed_pos_neg") if return_reason else ok


def validate_text(text):
    if text is None:
        return None, "Boş metin"
    text = str(text)
    if len(text.strip()) < 2:
        return None, "Boş veya çok kısa metin"
    if len(text) > MAX_TEXT_LENGTH:
        return text[:MAX_TEXT_LENGTH], f"Metin {MAX_TEXT_LENGTH} karaktere kısaltıldı"
    return text, None


def split_on_sarcasm(text: str):
    """Metni ironi marker'ından böler. Returns (head, tail, marker) or (None, None, None)."""
    clean = rule_clean(text)
    for m in SARCASM_MARKERS:
        mc = rule_clean(m)
        if mc in clean:
            parts = clean.split(mc, 1)
            head = parts[0].strip() if len(parts) > 0 else ""
            tail = parts[1].strip() if len(parts) > 1 else ""
            return head, tail, m
    return None, None, None


def has_sarcasm_negative_tail(tail: str) -> bool:
    if not tail:
        return False
    return any(cue in tail for cue in SARCASM_NEG_CUES)
