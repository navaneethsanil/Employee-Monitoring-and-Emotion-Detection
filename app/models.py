from django.db import models

class EmployeeLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20)
    emotion = models.CharField(max_length=20, null=True, blank=True)

    def __str__(self):
        return f"{self.timestamp} - {self.status} - {self.emotion}"

