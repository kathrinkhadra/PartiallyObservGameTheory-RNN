import torch
from torch import nn
import numpy as np

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, input,device):

        batch_size=input.size(0)


        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size,device)

        # Passing in the input and hidden state into the model and obtaining outputs
        #print(input.size())
        out, hidden = self.rnn(input, hidden)#[:, :,:6]

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        fc_input=[out]#[out,input[:, :,-1]]
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size,device):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden
