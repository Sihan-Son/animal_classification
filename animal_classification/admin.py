from django.contrib import admin

from .models import Image


class ImageAdmin(admin.ModelAdmin):
    fieldsets = [
        (
            'Image',
            {
                'fields': ['image', 'processed_image', 'comment', ]
            }
        ),
    ]


admin.site.register(Image, ImageAdmin)
