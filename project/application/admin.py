from django.contrib import admin
from .models import PostureLog

@admin.register(PostureLog)
class PostureLogAdmin(admin.ModelAdmin):
    list_display = ("user", "posture", "ear_shoulder_distance", "timestamp")
    list_filter = ("posture", "timestamp")
    search_fields = ("user__username",)