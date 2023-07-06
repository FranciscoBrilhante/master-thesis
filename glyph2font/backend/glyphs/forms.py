from django import forms
from django.core.exceptions import ValidationError
from.models import GenerativeModel
from django.contrib.auth.models import User
from django.utils.translation import gettext as _
import string


class Generate(forms.Form):
    format=forms.CharField(max_length=32)
    poligons=forms.IntegerField(max_value=20,min_value=1, required=False)
    segments=forms.IntegerField(max_value=20,min_value=1, required=False)
    colorscheme=forms.CharField(max_length=32)
    model=forms.CharField(max_length=32)
    labels=forms.CharField(max_length=32)
    width=forms.IntegerField()
    height=forms.IntegerField()
    images=forms.ImageField()
    

    def clean_format(self):
        data=self.cleaned_data['format']
        if data not in ['png','svg']:
            raise ValidationError(_('Image format to be returned must be either .png or .svg'), code='invalid')
        return data
    
    def clean_colorscheme(self):
        data=self.cleaned_data['colorscheme']
        if data not in ['white','black']:
            raise ValidationError(_('Color scheme of input images must be either black or white'), code='invalid')
        return data
    
    def clean_labels(self):
        data=self.cleaned_data['labels']
        valid_characters=list(string.ascii_uppercase)
        aux=list(data)
        for elem in aux:
            if elem not in valid_characters:
                raise ValidationError(_('labels must be a sequence of uppercase characters of the input image in order of appearance'), code='invalid')
        return data

    def clean_model(self):
        data=self.cleaned_data['model']
        if not GenerativeModel.objects.filter(name=data).exists():
            raise ValidationError(_('Model name must be one of the available list at /models'),code='invalid')
        return data
