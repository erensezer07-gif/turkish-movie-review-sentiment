from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from filmler.views import anasayfa, film_detay, toplu_film_ekle, kayit_ol
from filmler.forms import UserLoginForm
from filmler.views import live_search

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', anasayfa, name='anasayfa'),
    path('yukle/', toplu_film_ekle, name='toplu_film_ekle'),
    path('film/<int:film_id>/', film_detay, name='film_detay'),
    path('register/', kayit_ol, name='register'),
    path('live-search/', live_search, name='live_search'),

    # 🔒 Şifre Sıfırlama Adımları
    path('accounts/password_reset/', auth_views.PasswordResetView.as_view(
        template_name='registration/password_reset_form.html',
        email_template_name='registration/password_reset_email.html',
        subject_template_name='registration/password_reset_subject.txt',
    ), name='password_reset'),

    path('accounts/password_reset/done/', auth_views.PasswordResetDoneView.as_view(
        template_name='registration/password_reset_done.html'
    ), name='password_reset_done'),

    path('accounts/reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(
        template_name='registration/password_reset_confirm.html'
    ), name='password_reset_confirm'),

    path('accounts/reset/done/', auth_views.PasswordResetCompleteView.as_view(
        template_name='registration/password_reset_complete.html'
    ), name='password_reset_complete'),

    # Diğer hazır üyelik yolları (login, logout vb.)
    path('accounts/', include('django.contrib.auth.urls')),
]