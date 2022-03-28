import torch
import torchvision
from fastai.text.all import *
import torchtext
import pickle

import matplotlib.pyplot as plt
import PIL
from PIL import Image

from .model_implementation import *
from .text_pre_processing import *
from .image_processing import *

# # extract features
# features = model.encoder(input_batch)

# # get predictions
# pred_caps, alphas = model.decoder.generate_caption(features, vocab=vocab)
# pred_caps = pred_caps[1:len(pred_caps)-1]

# # make it printable
# caption = ' '.join(pred_caps)
# print("Predicted:", caption)

vocab = 1 
model = 1

def load_models():
    print("loading models.....")

    # load the vocab
    with open('static/vocab.pkl', 'rb') as file:
        vocab = pickle.load(file)   

    # initialize model
    model = EncoderDecoder(
        embed_sz = 300,
        vocab_sz = len(vocab),
        att_dim = 128,
        enc_dim = 2048,
        dec_dim = 256
    )

    # load trained model
    PATH = 'Unclean5Sentences.pth'

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['state_dict'])



# function to predict the caption for the given image
def get_prediction(image_bytes):
    
    # transform the give image first
    input_batch = transform_image(image_bytes)

    print(input_batch.size())

    return "Hey it's working hurray"