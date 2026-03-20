from django.urls import path
from .views import CoughPredictionAPI, home

urlpatterns = [
    path("", home),
    path("predict-cough/", CoughPredictionAPI.as_view()),
]
