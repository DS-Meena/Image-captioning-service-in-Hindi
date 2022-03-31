"""
WSGI config for ICS_hindi project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

# for the text vocab error
# from static.text_pre_processing import textVocab

application = get_wsgi_application()

def main():
	os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ICS_hindi.settings')

if __name__ == '__main__':

    from static.text_pre_processing import textVocab

    main()
