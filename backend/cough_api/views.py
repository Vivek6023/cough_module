from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import render
from .ml_model import run_prediction
from django.core.files.storage import default_storage

# FRONTEND PAGE
def home(request):
    return render(request, "index.html")

# API ENDPOINT
class CoughPredictionAPI(APIView):
    def post(self, request):
        audio = request.FILES.get("audio")
        if not audio:
            return Response({"error": "No audio file"}, status=400)

        file_path = default_storage.save(audio.name, audio)
        result = run_prediction(file_path)

        return Response(result, status=200)
