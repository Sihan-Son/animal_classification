from django.urls import path

from animal_classification.views import IndexView, TestView, UploadView

app_name = 'animal'

urlpatterns = [
    path('', IndexView.as_view(), name="index"),
    path('test/', TestView.as_view(), name="test"),
    path('upload/', UploadView.as_view(), name='upload'),

]
