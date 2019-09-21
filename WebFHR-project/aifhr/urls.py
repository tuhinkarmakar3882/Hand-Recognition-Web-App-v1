#aifhr URLs.py

from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

import aifhr.views

urlpatterns = [
                  path('', aifhr.views.home, name='AIFHR_Home'),
                  path('/Recognition',aifhr.views.perform_recognition, name='perform_recognition')
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
