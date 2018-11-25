from datetime import datetime

from django.db import models
from django.urls import reverse_lazy

from _django import settings


def user_path(instance, filename):
    """
    출처: https://wayhome25.github.io/django/2017/03/14/django-07-kilogram-04-photo-model/
    :param instance: The name of the model.
    :param filename: Random generated 8 characters.
    :type instance: str
    :type filename: str
    :return: instance owner's username, file's name, extension (e.g., jho/test_image.png)
    :rtype: str
    """
    from random import choice
    import string  # string.ascii_letters : ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
    arr = [choice(string.ascii_letters) for _ in range(8)]
    date = datetime.strftime(instance.uploaded_at, "%y%m%d")
    pid = date.join('_').join(arr)
    extension = filename.split('.')[-1]  # 배열로 만들어 마지막 요소를 추출하여 파일확장자로 지정
    # file will be uploaded to MEDIA_ROOT/date/<random>
    return '%s/%s.%s' % (date, pid, extension)


class Photo(models.Model):
    image = models.ImageField(upload_to='%Y/%m/%d/orig')
    processed_image = models.ImageField(blank=True, upload_to='%Y/%m/%d/processed')
    # owner = models.ForeignKey(settings.AUTH_USER_MODEL)  # 로그인 한 사용자, many to one relation
    comment = models.CharField(max_length=255, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)  # 레코드 생성시 현재 시간으로 자동 생성

    def get_absolute_url(self):
        url = reverse_lazy('detail', kwargs={'pk': self.pk})
        return url
