import os

from django.shortcuts import redirect, render
from django.views.generic import DetailView

from animal_classification.forms import UploadImageForm
from animal_classification.models import Image, ClassificationResult
from classification.AnimalClassification.classification import AnimalClassification


def upload_image(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            print("valid")
            image = form.save()
            return redirect('animal:classify', pk=image.pk)
        else:
            print("not valid")
            form = UploadImageForm()
    else:
        form = UploadImageForm()
    return render(request, 'animal/image_upload_form.html', {'form': form})


def classify_image(request, pk):
    # origin image's variables
    origin_image = Image.objects.get(pk=pk)
    origin_path = os.path.join(r'', origin_image.content.path)

    # [START check_is_already_classified]
    try:
        result = ClassificationResult.objects.get(pk=pk)
        return redirect(result)
    except Exception as e:
        pass

    # [END check_is_already_classified]

    # [START classify_image]
    classifier = AnimalClassification(origin_path)
    label, probability = classifier.run_graph()
    # [END classify_image]

    # [START save_result_to_model]
    result = ClassificationResult()
    result.origin = origin_image
    result.label = str(label)
    probability = max(probability[0])
    result.probability = float(probability)

    try:
        result.save()
    except Exception as e:
        print('result saving failed: ' + e)
        return redirect(result)
    else:
        print('result saved!')
    # [END save_result_to_model]

    print(origin_path)

    return redirect(result)


# class ImageDetail(DetailView):

class ClassificationResultView(DetailView):
    model = ClassificationResult
    template_name = 'animal/classification_result.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        print(context)
        return context
