from django.db import models


class Film(models.Model):
    """Veritabanındaki film kayıtları."""

    isim = models.CharField(max_length=200, verbose_name="Film Adı")
    konu = models.TextField(verbose_name="Özet / Konu", blank=True, null=True)
    yil = models.CharField(max_length=10, blank=True, null=True, verbose_name="Yapım Yılı")
    puan = models.FloatField(default=0.0, verbose_name="IMDb Puanı")
    oyuncular = models.CharField(max_length=500, blank=True, default="Bilgi Yok", verbose_name="Oyuncular")
    turler = models.CharField(max_length=250, blank=True, default="", verbose_name="Kategoriler")
    poster_url = models.CharField(
        max_length=500, blank=True, null=True,
        default="https://via.placeholder.com/300x450",
        verbose_name="Poster URL",
    )
    fragman_url = models.CharField(max_length=500, blank=True, null=True, verbose_name="Fragman URL")
    eklenme_tarihi = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Film"
        verbose_name_plural = "Filmler"
        ordering = ["-eklenme_tarihi"]

    def __str__(self):
        return self.isim


class Yorum(models.Model):
    """Kullanıcıların filmlere yaptığı yorumlar ve AI analiz sonuçları."""

    film = models.ForeignKey(Film, on_delete=models.CASCADE, related_name='yorumlar')
    kullanici_adi = models.CharField(max_length=50, default="Misafir")
    icerik = models.TextField(verbose_name="Yorum İçeriği")
    tarih = models.DateTimeField(auto_now_add=True)

    # --- YAPAY ZEKA SONUÇLARI ---
    ai_karari = models.CharField(max_length=20, blank=True, verbose_name="AI Kararı")
    ai_guveni = models.FloatField(default=0.0, verbose_name="Güven Skoru")
    ai_kaynak = models.CharField(max_length=30, blank=True, null=True, verbose_name="AI Kaynak Model")

    class Meta:
        verbose_name = "Yorum"
        verbose_name_plural = "Yorumlar"
        ordering = ["-tarih"]

    def __str__(self):
        return f"{self.kullanici_adi} - {self.film.isim}"

    @property
    def sent_key(self):
        """Template'teki filtreleme için kısa duygu anahtarı döndürür."""
        mapping = {"OLUMLU": "pos", "OLUMSUZ": "neg", "NÖTR": "neu"}
        return mapping.get(self.ai_karari, "neu")