from django.contrib import admin
from django.urls import path
from detector import views

urlpatterns = [
    path('', views.home, name='home'),
    path('totalreg/', views.totalreg, name='totalreg'),
    path('add/', views.add, name='add'),
    path('start/', views.start, name='start'),
    path('totalreg', views.totalreg, name='totalreg'),
    path('extract_attendance', views.extract_attendance, name='extract_attendance'),
    path('add_attendance/', views.add_attendance, name='add_attendance'),
    path('download_csv/<path:file_path>/', views.download_csv, name='download_csv'),
]


