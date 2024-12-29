# django-blog-main/blog/views.py
# Django импорты
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.models import User
from django.db import models
from django.shortcuts import get_object_or_404, redirect, render
from django.views.generic import (
   CreateView,
   DeleteView,
   DetailView,
   ListView,
   UpdateView
)

# Локальные импорты 
from .models import Like, Post


@login_required
def like_post(request, pk: int) -> redirect:
   """
   Description:
       Обработка лайков поста.

   Args:
       request: HTTP запрос
       pk (int): Идентификатор поста

   Returns:
       redirect: Перенаправление на страницу поста
   """
   post = get_object_or_404(Post, pk=pk)
   liked = Like.objects.filter(user=request.user, post=post).exists()

   if liked:
       Like.objects.filter(user=request.user, post=post).delete()
   else:
       Like.objects.create(user=request.user, post=post)

   return redirect('post-detail', pk=post.pk)


class PostListView(ListView):
   """
   Description:
       Отображение списка всех постов.
   """
   model = Post
   template_name = 'blog/home.html'
   context_object_name = 'posts'
   ordering = ['-date_posted']
   paginate_by = 10

   def get_context_data(self, **kwargs):
       """Добавление информации о лайках в контекст"""
       context = super().get_context_data(**kwargs)
       if self.request.user.is_authenticated:
           likes = Like.objects.filter(user=self.request.user)
           context['liked_posts'] = likes.values_list('post_id', flat=True)
       return context


class UserPostListView(ListView):
   """
   Description:
       Отображение постов конкретного пользователя.
   """
   model = Post
   template_name = 'blog/user_posts.html'
   context_object_name = 'posts'
   paginate_by = 10

   def get_queryset(self):
       """Фильтрация постов по автору"""
       self.user_profile = get_object_or_404(
           User,
           username=self.kwargs.get('username')
       )
       return Post.objects.filter(
           author=self.user_profile
       ).order_by('-date_posted')

   def get_context_data(self, **kwargs):
       """Добавление профиля пользователя и лайков в контекст"""
       context = super().get_context_data(**kwargs)
       context['user_profile'] = self.user_profile
       if self.request.user.is_authenticated:
           likes = Like.objects.filter(user=self.request.user)
           context['liked_posts'] = likes.values_list('post_id', flat=True)
       return context


class LikedPostListView(LoginRequiredMixin, ListView):
   """
   Description:
       Отображение постов, которые понравились пользователю.
   """
   model = Post
   template_name = 'blog/liked_posts.html'
   context_object_name = 'posts'
   paginate_by = 10

   def get_queryset(self):
       """Получение списка лайкнутых постов"""
       liked_posts_ids = Like.objects.filter(
           user=self.request.user
       ).order_by('-created').values_list('post_id', flat=True)
       
       return Post.objects.filter(
           id__in=liked_posts_ids
       ).order_by(
           models.Case(
               *[models.When(id=pk, then=pos) 
                 for pos, pk in enumerate(liked_posts_ids)]
           )
       )


class PostDetailView(DetailView):
   """
   Description:
       Детальное отображение поста.
   """
   model = Post
   template_name = 'blog/post_detail.html'

   def get_context_data(self, **kwargs):
       """Добавление информации о лайках в контекст"""
       context = super().get_context_data(**kwargs)
       post = self.get_object()
       context['total_likes'] = post.likes.count()
       
       if self.request.user.is_authenticated:
           context['liked'] = Like.objects.filter(
               user=self.request.user,
               post=post
           ).exists()
       else:
           context['liked'] = False
       
       return context


class PostCreateView(LoginRequiredMixin, CreateView):
   """
   Description:
       Создание нового поста.
   """
   model = Post
   fields = ['title', 'subtitle', 'content']

   def form_valid(self, form):
       """Установка автора поста"""
       form.instance.author = self.request.user
       return super().form_valid(form)


class PostUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
   """
   Description:
       Обновление существующего поста.
   """
   model = Post
   fields = ['title', 'subtitle', 'content']

   def form_valid(self, form):
       """Проверка автора поста"""
       form.instance.author = self.request.user
       return super().form_valid(form)

   def test_func(self):
       """Проверка прав на редактирование"""
       post = self.get_object()
       return self.request.user == post.author


class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
   """
   Description:
       Удаление поста.
   """
   model = Post
   success_url = '/'

   def test_func(self):
       """Проверка прав на удаление"""
       post = self.get_object()
       return self.request.user == post.author


def about(request):
   """
   Description:
       Отображение страницы 'О нас'.

   Args:
       request: HTTP запрос

   Returns:
       render: Отрендеренная страница
   """
   return render(request, 'blog/about.html', {'title': 'About'})