from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("models",views.listModels, name="listModels"),
    path("generate",views.generate, name="generate")
]