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
    list_display = ("kullanici_adi", "film", "ai_karari", "ai_guveni", "get_kaynak_badge", "tarih")
    list_filter = ("ai_karari", "tarih", "ai_kaynak")
    search_fields = ("kullanici_adi", "icerik", "film__isim")
    ordering = ("-tarih",)
    readonly_fields = ("ai_karari", "ai_guveni", "ai_kaynak")

    def get_kaynak_badge(self, obj):
        """AI Kaynağını renkli badge olarak gösterir."""
        if not obj.ai_kaynak:
            return "-"
        
        # Renk belirleme
        color = "gray"
        if "local" in obj.ai_kaynak:
            color = "green"
        elif "api" in obj.ai_kaynak:
            color = "orange"
        elif "error" in obj.ai_kaynak or "exception" in obj.ai_kaynak:
            color = "red"
        
        from django.utils.html import format_html
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 6px; border-radius: 4px;">{}</span>',
            color,
            obj.ai_kaynak
        )
    get_kaynak_badge.short_description = "AI Kaynak"
    get_kaynak_badge.admin_order_field = "ai_kaynak"