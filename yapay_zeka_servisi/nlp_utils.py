import re


def temizle_tek(metin: str) -> str:
    metin = str(metin).lower()
    metin = re.sub(r"http\S+|www\.\S+", " ", metin)
    metin = re.sub(r"\d+", " ", metin)
    metin = re.sub(r"[^a-zçğıöşü\s]", " ", metin)
    metin = re.sub(r"\s+", " ", metin).strip()
    return metin


def temizle_liste(X):
    return [temizle_tek(x) for x in X]
