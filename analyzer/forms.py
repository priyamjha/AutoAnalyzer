from django import forms
from .models import UploadedDataset

class UploadForm(forms.ModelForm):
    class Meta:
        model = UploadedDataset
        fields = ['file']
