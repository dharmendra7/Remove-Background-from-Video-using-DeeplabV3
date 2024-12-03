from django.urls import path, include
from .views import *

urlpatterns = [
    path('', home, name = 'home'),
    # path('', uploadTheVideo, name = 'upload-video'),
    path('create-frames/', createFrames, name = 'create-frames'),
    path('merge-frames/', mergeFrames, name = 'merge-frames'),
]