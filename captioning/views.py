from django.shortcuts import render
from urllib import request

# Create your views here.
def index(request):

	return render(request, 'index.html')