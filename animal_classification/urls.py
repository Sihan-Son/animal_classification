from django.urls import path

# from animal_classification.views import IndexView, TestView, UploadView, ImageUpload, ImageDetail, ImageClassify
from animal_classification.views import upload_image, classify_image, ClassificationResultView

app_name = 'animal'

urlpatterns = [
    # path('', IndexView.as_view(), name="index"),
    # path('upload/', ImageUpload.as_view(), name='upload'),
    path('upload/', upload_image, name='upload'),
    path('classify/<pk>/', classify_image, name='classify'),
    path('result/<pk>/', ClassificationResultView.as_view(), name='result'),
    # path('detail/<pk>/', ImageDetail.as_view(), name='detail'),
    # path('result/', )
]
