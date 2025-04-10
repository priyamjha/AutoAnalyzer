from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Upload form
    path('report/<int:dataset_id>/', views.report, name='report'),  # Report page
    path('download-pdf/<int:dataset_id>/', views.download_pdf, name='download_pdf'),  # PDF download
]
