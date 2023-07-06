from django.db import models


class GenerativeModel(models.Model):
    name=models.CharField(max_length=256)
    description_en=models.CharField(max_length=1024)
    description_pt=models.CharField(max_length=1024)
    log_path=models.CharField(max_length=512)
    sugested_inputs=models.CharField(max_length=26)
    eta=models.FloatField()
    is_diffusion=models.BooleanField(default=False)
    aux_diff_path=models.CharField(max_length=512, default=None, null=True )
    diff_path=models.CharField(max_length=512, default=None, null=True)