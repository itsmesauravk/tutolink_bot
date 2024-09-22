from django.urls import path
from . import views

urlpatterns = [
    path('recommend-tutors/', views.recommend_tutors, name='recommend_tutors'),
]
