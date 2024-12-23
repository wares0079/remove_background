"""
URL configuration for removebackground project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from bg_removal.views import upload_image, image_result

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', upload_image, name='upload_image'),
    path('result/<int:pk>/', image_result, name='image_result'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)