# django-blog-main/blog/urls.py
# Django URL маршрутизация
from django.urls import path

# Импорт представлений
from .views import (
   PostListView,
   PostDetailView, 
   PostCreateView,
   PostUpdateView,
   PostDeleteView,
   UserPostListView,
   LikedPostListView,
   like_post
)
from . import views

urlpatterns = [
   # Главная страница блога
   path('', PostListView.as_view(), name='blog-home'),
   
   # Маршруты для операций с постами
   path('post/<int:pk>/', PostDetailView.as_view(), name='post-detail'),
   path('post/new/', PostCreateView.as_view(), name='post-create'),
   path('post/<int:pk>/update', PostUpdateView.as_view(), name='post-update'),
   path('post/<int:pk>/delete', PostDeleteView.as_view(), name='post-delete'),
   
   # Маршруты для пользовательских функций
   path('user/<str:username>', UserPostListView.as_view(), name='user-posts'),
   path('post/<int:pk>/like/', like_post, name='like-post'),
   path('liked-posts/', LikedPostListView.as_view(), name='liked-posts'),
   
   # Дополнительные страницы
   path('about/', views.about, name='blog-about'),
]