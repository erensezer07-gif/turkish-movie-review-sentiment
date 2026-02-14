from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm


class UserRegisterForm(UserCreationForm):
    email = forms.EmailField(
        label="E-posta Adresi",
        help_text="Şifre sıfırlama için gereklidir.",
        widget=forms.EmailInput(attrs={"placeholder": "ornek@email.com"}),
    )

    class Meta:
        model = User
        fields = ['username', 'email']
        labels = {
            'username': 'Kullanıcı Adı',
        }
        help_texts = {
            'username': 'Harf, rakam ve @/./+/-/_ karakterlerinden oluşabilir.',
        }
        widgets = {
            'username': forms.TextInput(attrs={"placeholder": "Kullanıcı adınız"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Şifre alanlarını Türkçeleştir
        self.fields['password1'].label = "Şifre"
        self.fields['password1'].help_text = "En az 8 karakter olmalıdır."
        self.fields['password1'].widget = forms.PasswordInput(attrs={"placeholder": "Şifreniz"})

        self.fields['password2'].label = "Şifre Onayı"
        self.fields['password2'].help_text = "Doğrulama için şifrenizi tekrar girin."
        self.fields['password2'].widget = forms.PasswordInput(attrs={"placeholder": "Şifrenizi tekrar girin"})


class UserLoginForm(AuthenticationForm):
    username = forms.CharField(
        label="Kullanıcı Adı",
        widget=forms.TextInput(attrs={"placeholder": "Kullanıcı adınız"}),
    )
    password = forms.CharField(
        label="Şifre",
        widget=forms.PasswordInput(attrs={"placeholder": "Şifreniz"}),
    )