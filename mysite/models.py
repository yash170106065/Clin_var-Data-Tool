from django.db import models

class predicting_mod (models.Model):
    sequence = models.FileField(upload_to='text_files')





