import base64

from django.shortcuts import render
from urllib import request

from .forms import ImageUploadForm

from static.inference import get_prediction

# Create your views here.
def index(request):

	image_uri = None 
	Predicted_caption = None

	if request.method == 'POST':

		# get the form
		form = ImageUploadForm(request.POST, request.FILES)

		# check if form is valid
		if form.is_valid():
			
			# get the image
			image = form.cleaned_data['image']
			image_bytes = image.file.read()

			# encode the image
			encoded_img = base64.b64encode(image_bytes).decode('ascii')
			image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)

			try:
				Predicted_caption = get_prediction(image_bytes)
			except RuntimeError as re:
				print(re)			

	else:
		form = ImageUploadForm()

	context = {
		'form': form,
		'image_uri': image_uri,
		'predicted_caption': Predicted_caption,
	}

	return render(request, 'index.html', context)

