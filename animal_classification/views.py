from django.shortcuts import render, redirect
from django.views import View

from animal_classification.forms import PhotoForm


class IndexView(View):
    pass


class TestView(View):
    pass


class UploadView(View):
    def get(self, request):
        form = PhotoForm()
        ctx = {
            'form': form,
        }

        return render(request, 'upload_photo.html', ctx)

    def post(self, request):
        form = PhotoForm(request.POST, request.FILES)

        if form.is_valid():
            obj = form.save()
            return redirect(obj)

        ctx = {
            'form': form,
        }

        return render(request, 'edit.html', ctx)
