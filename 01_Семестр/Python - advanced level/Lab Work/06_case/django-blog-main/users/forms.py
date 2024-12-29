# django-blog-main/users/forms.py
# Импорты Django
from django import forms
from django.forms import FileInput
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

# Импорты локальных моделей
from .models import Profile


class UserRegisterForm(UserCreationForm):
    """
    Description:
        Форма для регистрации нового пользователя.
        Расширяет стандартную форму UserCreationForm, добавляя поле email.

    Attributes:
        email: Поле для ввода email (обязательное).
    """

    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

    def __init__(self, *args, **kwargs) -> None:
        """
        Description:
            Инициализация формы. Удаляет поле 'usable_password', если оно присутствует.

        Args:
            *args: Позиционные аргументы.
            **kwargs: Именованные аргументы.
        """
        super().__init__(*args, **kwargs)
        # Удаление поля 'usable_password', если оно есть
        if 'usable_password' in self.fields:
            del self.fields['usable_password']


class UserUpdateForm(forms.ModelForm):
    """
    Description:
        Форма для обновления данных пользователя.
    """

    class Meta:
        model = User
        fields = ['username']


class ProfileUpdateForm(forms.ModelForm):
    """
    Description:
        Форма для обновления профиля пользователя.
        Включает поля для биографии и изображения.

    Attributes:
        image: Поле для загрузки изображения с использованием виджета FileInput.
    """

    image = forms.ImageField(widget=FileInput)

    class Meta:
        model = Profile
        fields = ['bio', 'image']