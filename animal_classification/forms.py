from django import forms
from .models import Photo


class PhotoForm(forms.ModelForm):
    class Meta:
        model = Photo
        fields = ('image', 'comment')
        exclude = ('processed_image', 'uploaded_at',)
