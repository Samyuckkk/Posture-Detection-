from django.db import models
from django.contrib.auth.models import User

class PostureLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    posture = models.CharField(max_length=20)  # "upright", "slouching", "unknown"
    ear_shoulder_distance = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.user} - {self.posture} @ {self.timestamp.strftime('%H:%M:%S')}"
