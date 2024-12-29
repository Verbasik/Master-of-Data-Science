# django-blog-main/users/views.py
# Импорты Django
from django.shortcuts import redirect, render
from django.contrib import messages
from django.contrib.auth.decorators import login_required

# Импорты локальных форм
from .forms import UserRegisterForm, UserUpdateForm, ProfileUpdateForm


def register(request):
    """
    Description:
        Обрабатывает регистрацию нового пользователя.
        Если запрос методом POST, проверяет валидность формы и сохраняет пользователя.
        Если запрос методом GET, отображает пустую форму регистрации.

    Args:
        request: Объект запроса Django.

    Returns:
        Перенаправляет на страницу входа после успешной регистрации
        или отображает форму регистрации.
    """
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()  # Автоматическое хэширование пароля

            username = form.cleaned_data.get('username')
            messages.success(request, 'Your account has been created! You are now able to log in.')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'users/register.html', {'form': form})


@login_required
def profile(request):
    """
    Description:
        Обрабатывает обновление профиля пользователя.
        Если запрос методом POST, проверяет валидность форм и сохраняет изменения.
        Если запрос методом GET, отображает формы с текущими данными пользователя.

    Args:
        request: Объект запроса Django.

    Returns:
        Перенаправляет на страницу профиля после успешного обновления
        или отображает формы обновления.
    """
    if request.method == 'POST':
        user_update_form = UserUpdateForm(request.POST, instance=request.user)
        profile_update_form = ProfileUpdateForm(
            request.POST,
            request.FILES,
            instance=request.user.profile
        )

        if user_update_form.is_valid() and profile_update_form.is_valid():
            if user_update_form.has_changed() or profile_update_form.has_changed():
                user_update_form.save()
                profile_update_form.save()
                messages.success(request, 'Your account has been updated!')
            else:
                messages.info(request, 'No changes detected.')

            return redirect('profile')

    else:
        user_update_form = UserUpdateForm(instance=request.user)
        profile_update_form = ProfileUpdateForm(instance=request.user.profile)

    context = {
        'user_update_form': user_update_form,
        'profile_update_form': profile_update_form
    }

    return render(request, 'users/profile.html', context)