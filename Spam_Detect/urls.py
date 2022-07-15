
from django.contrib import admin
from django.urls import re_path, include

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import re_path
from app import views


urlpatterns = [
    re_path('admin/', admin.site.urls),
    re_path(r'^$',views.upload_files, name='upload_files'),
    # re_path(r'^Turf_Edit/(?P<id>\d+)/$',views.Turf_Edit, name='Turf_Edit'),
    # re_path(r'^req/$',views.req, name='req'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL,
                          document_root=settings.STATIC_ROOT)
