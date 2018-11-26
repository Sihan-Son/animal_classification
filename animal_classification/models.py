from django.db import models
from django.shortcuts import resolve_url


class Image(models.Model):
    content = models.FileField(upload_to='image/origin/%Y/%m/%d')
    comment = models.CharField(max_length=200, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)  # 레코드 생성시 현재 시간으로 자동 생성

    def __str__(self):
        return self.content.url

    def get_absolute_url(self):
        url = resolve_url('animal:detail', self.pk)
        return url


class ClassificationResult(models.Model):
    origin = models.OneToOneField(Image, on_delete=models.CASCADE)
    label = models.CharField(default='?', max_length=200)
    probability = models.FloatField(default=0.0)

    def __str__(self):
        return "('{}', {:f}%)".format(self.label, (self.probability * 100))

    def get_absolute_url(self):
        url = resolve_url('animal:result', self.pk)
        return url
