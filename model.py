import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet34(pretrained=True)
        # Freeze the backbone
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.embed.weight.data.normal_(-0.1, 0.1)
        self.embed.bias.data.fill_(0)


    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout_prob=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.init_weights()


    def forward(self, features, captions):
        # Features: Hidden states from the CNN
        # Captions: Input sentence
        embeddings = torch.cat([features[:, None, :], self.embed(captions[:, :-1])], axis=1)  # Exclude the <end> token
        states = self.make_initial_states(batch_size=features.shape[0])
        out, states = self.rnn(embeddings, states)
        lin = self.linear(out)
        return lin


    def make_initial_states(self, batch_size):
        st1 = torch.normal(0, 0.5, size=(self.num_layers, batch_size, self.hidden_size))
        st2 = torch.normal(0, 0.5, size=(self.num_layers, batch_size, self.hidden_size))
        if torch.cuda.is_available():
            st1 = st1.cuda()
            st2 = st2.cuda()
        return st1, st2


    def init_weights(self):
        """Initialize the weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)


    def sample(self, inputs, states=None, max_len=20):
        "accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"
        predicted_sentence = []
        for i in range(max_len):
            hiddens, states = self.rnn(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            print(outputs.shape)
            _, predicted = outputs.max(1)
            predicted_sentence.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
        return predicted_sentence
