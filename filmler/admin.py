from django.contrib import admin
from .models import Film, Yorum


@admin.register(Film)
class FilmAdmin(admin.ModelAdmin):
    list_display = ("isim", "yil", "puan", "turler", "eklenme_tarihi")
    list_filter = ("yil",)
    search_fields = ("isim", "oyuncular", "turler")
    ordering = ("-eklenme_tarihi",)


@admin.register(Yorum)
class YorumAdmin(admin.ModelAdmin):
    list_display = ("kullanici_adi", "film", "ai_karari", "ai_guveni", "ai_kaynak", "tarih")
    list_filter = ("ai_karari", "tarih")
    search_fields = ("kullanici_adi", "icerik", "film__isim")
    ordering = ("-tarih",)
    readonly_fields = ("ai_karari", "ai_guveni", "ai_kaynak")