from django import forms


class ImageUploadForm(forms.Form):
    image = forms.ImageField(required=False)
    caption = forms.CharField(required=False, label = 'Original Caption')