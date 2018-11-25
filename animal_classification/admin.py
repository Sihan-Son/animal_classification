from django.contrib import admin

from .models import Photo


class PhotoAdmin(admin.ModelAdmin):
    fieldsets = [
        (
            'Photo',
            {
                'fields': ['image', 'processed_image', 'comment', ]
            }
        ),
    ]


admin.site.register(Photo, PhotoAdmin)
