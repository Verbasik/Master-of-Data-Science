# django-blog-main/blog_project/urls.py
# Django импорты
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import include, path

# Локальные импорты
from users import views as users_views

urlpatterns = [
   # Админка
   path('admin/', admin.site.urls),

   # Аутентификация
   path('signup/', users_views.register, name='register'),
   path('profile/', users_views.profile, name='profile'),
   path(
       'login/',
       auth_views.LoginView.as_view(template_name='users/login.html'),
       name='login'
   ),
   path(
       'logout/',
       auth_views.LogoutView.as_view(template_name='users/logout.html'),
       name='logout'
   ),

   # URL блога
   path('', include('blog.urls')),
]

# Медиа файлы в режиме отладки
if settings.DEBUG:
   urlpatterns += static(
       settings.MEDIA_URL,
       document_root=settings.MEDIA_ROOT
   )