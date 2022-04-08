import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.models as models

from .image_processing import *

# pretrained inception model
inception = models.inception_v3(pretrained=True)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.my_inception = MyInceptionFeatureExtractor(inception)
        
    def forward(self, images):
   
        # get the feature maps
        features = self.my_inception(images) 
        
        features = features.permute(0, 2, 3, 1)
        
        features = features.view(features.size(0), -1, features.size(-1))
        
        return features

class Attention(nn.Module):
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        
        # initialize the shapes
        self.attention_dim = attention_dim
        
        # create linear layer's to transform following
        # (input shape, output shape)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # decoder's output
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # encoded image
        self.full_att = nn.Linear(attention_dim, 1)  # attention's output
    
    # input - features of image and hidden state value
    def forward(self, features, hidden_states):
        # pass the tensor's through linear layers
        att1 = self.encoder_att(features)   
        att2 = self.decoder_att(hidden_states)
                
        # combine both attentions
        combined_states = torch.tanh(att1 + att2.unsqueeze(1))
        
        # pass combined state through last linear layer
        attention_scores = self.full_att(combined_states)
        
        attention_scores = attention_scores.squeeze(2)        

        # calculate alpha 
        alpha = F.softmax(attention_scores, dim=1)
        
        # get attention weights
        weighted_encoding = features * alpha.unsqueeze(2)   # torch.Size([bs, 64, 1])
        weighted_encoding = weighted_encoding.sum(dim=1)    # sum all weights at dim 1
        
        # return alpha and attention weights (both are tensors)
        return alpha, weighted_encoding
        
class Decoder(nn.Module):
    def __init__(self, embed_sz, vocab_sz, att_dim, enc_dim, dec_dim, drop_prob=0.3):
        super().__init__()
        
        # initialize the model parameters
        self.vocab_sz = vocab_sz
        self.att_dim = att_dim
        self.dec_dim = dec_dim
        
        # initialize embedding model and attention model
        self.embedding = nn.Embedding(vocab_sz, embed_sz)
        self.attention = Attention(enc_dim, dec_dim, att_dim)
        
        # create the hidden and cell state
        self.init_h = nn.Linear(enc_dim, dec_dim)
        self.init_c = nn.Linear(enc_dim, dec_dim)
        
        # create lstm cell
        self.lstm_cell = nn.LSTMCell(embed_sz + enc_dim, dec_dim, bias=True)
        
        # create other nn layers
        self.f_beta = nn.Linear(dec_dim, enc_dim)
        self.fcn = nn.Linear(dec_dim, vocab_sz)
        self.drop = nn.Dropout(drop_prob)
    
    def forward(self, features, captions):
        
        # vectorize the captions(tokenized):
        embeds = self.embedding(captions)
        
        # initialize hidden and cell state
        h, c = self.init_hidden_state(features)
        
        # get the captions length in current batch
        cap_len = len(captions[0]) - 1
        
        # get batch size and features size
        batch_sz = captions.size(0)
        num_features = features.size(1)
        
        # create tensor of zeros for predictions and alpha
        preds = torch.zeros(batch_sz, cap_len, self.vocab_sz)
        alphas = torch.zeros(batch_sz, cap_len, num_features)
        
        for i in range(cap_len):
            # get alpha and attention weights
            alpha, att_weights = self.attention(features, h)
            
            # create lstm input
            lstm_input = torch.cat((embeds[:,i], att_weights), dim=1)
            
            # pass through lstm cell
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            # pass through linear layer
            output = self.fcn(self.drop(h))
            
            # store the output and alpha
            preds[:, i] = output
            alphas[:, i] = alpha
            
        return preds, alphas
    
    # create method to generate captions
    def generate_caption(self, features, max_len=20, vocab=None):
        batch_sz = features.size(0)
        
        # hidden and cell state
        h, c = self.init_hidden_state(features)
        
        alphas = []
        captions = [vocab.stoi['<start>']]
        
        # starting input
        word = torch.tensor(vocab.stoi['<start>']).view(1, -1)
        embeds = self.embedding(word)
        
        # get next 20 words
        for i in range(max_len):
            alpha, weighted_encoding = self.attention(features, h)
            
            # store alpha score
            alphas.append(alpha.cpu().detach().numpy())
            
            # update hidden and cell state
            lstm_input = torch.cat((embeds[:, 0], weighted_encoding), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            # pass through layers
            output = self.fcn(self.drop(h))
            output = output.view(batch_sz, -1)
            
            # select the best word
            pred_word_idx = output.argmax(dim=1)
            
            # save the word
            captions.append(pred_word_idx.item())
        
            # stop when end of sentence
            if vocab.itos[pred_word_idx.item()] == '<end>':
                break
                
            # next input
            embeds = self.embedding(pred_word_idx.unsqueeze(0))
            
        # return sentence
        return [vocab.itos[idx] for idx in captions], alphas  # if idx != 0 and idx != 1 and idx != 2
    
    # method to get hidden and cell state value
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        
        # return hidden and cell state
        return h, c

class EncoderDecoder(nn.Module):
    def __init__(self, embed_sz, vocab_sz, att_dim, enc_dim, dec_dim, drop_prob=0.3):
        super().__init__()
        
        # crete the encoder and decoder models
        self.encoder = Encoder()
        self.decoder = Decoder(
            embed_sz = embed_sz,
            vocab_sz = vocab_sz,
            att_dim = att_dim,
            enc_dim = enc_dim,
            dec_dim = dec_dim
        )
    
    def forward(self, images, captions):
        # extract image features
        features = self.encoder(images)
        
        # generate captions
        outputs = self.decoder(features, captions)
        
        # return predicted caption, attention alphas
        return outputs