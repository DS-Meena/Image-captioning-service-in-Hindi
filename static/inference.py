import torch
import torchvision
# from fastai.text.all import *
import torchtext
import pickle

import matplotlib.pyplot as plt
import PIL
from PIL import Image

from .model_implementation import *
from .text_pre_processing import textVocab
from .image_processing import *

# ------------------------------------------
print("loading models.....")

with open('static/vocab_stoi.pkl', 'rb') as handle:
    stoi = pickle.load(handle)
with open('static/vocab_itos.pkl', 'rb') as handle:
    itos = pickle.load(handle)

# create text vocab object
vocab = textVocab()
vocab.stoi = stoi 
vocab.itos = itos

# load the vocab (Was showing error so i recreated a new)
# with open('static/vocab.pkl', 'rb') as file:
#     vocab = pickle.load(file)   

# initialize model
model = EncoderDecoder(
    embed_sz = 300,
    vocab_sz = len(vocab),
    att_dim = 128,
    enc_dim = 2048,
    dec_dim = 256
)

criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# load trained model
PATH = 'static/Unclean5Sentences.pth'

checkpoint = torch.load(PATH, map_location ='cpu')
model.load_state_dict(checkpoint['state_dict'])

# ------------------------------------------

# function to predict the caption for the given image
def get_prediction(image_bytes):
    
    model.eval()
    with torch.no_grad():
        # transform the give image first
        img = transform_image(image_bytes)
        
        print("Encoding..........")
        # extract features
        print(img.shape)
        features = model.encoder(img)

        print("Predicting..........")
        # get predictions
        pred_caps, alphas = model.decoder.generate_caption(features, vocab=vocab)
        pred_caps = pred_caps[1:len(pred_caps)-1]

        # make it printable
        caption = ' '.join(pred_caps)
        print("Predicted:", caption)

    return caption

# function to optimize the caption for the given image
def optimize_prediction(captions, image_bytes):

    print("Optimizing......")

    # optimizing the model
    model.train()
    # Image
    
    # transform the give image first
    img = transform_image(image_bytes)

    print(img.shape)

    # original caption
    # convert original caption into vector
    captions = [vocab.stoi[word] for word in captions.split(' ')]
    captions.insert(0, 1)
    captions.append(2)
    captions = torch.tensor(captions)
    captions = captions.unsqueeze(0)

    print(captions.shape)

    pred_caps, attentions = model(img, captions)

    # calculate batch loss
    targets = captions[:, 1:]

    print("Calculating loss......")
    loss = criterion(pred_caps.view(-1, len(vocab)), targets.reshape(-1))
    print(loss.item())

    # # optimize the model
    loss.backward()
    optimizer.step()

    print("Saving optimized mode.....")
    save_model(model, 13, len(vocab))
    