from django import forms

from animal_classification.models import Image


class UploadImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['content', 'comment']
