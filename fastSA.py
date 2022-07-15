'''
Estimated run time (W/ CUDA)

Epoch: 01 | Epoch Time: 0m 7s
	Train Loss: 0.688 | Train Acc: 61.31%
	 Val. Loss: 0.637 |  Val. Acc: 72.46%
Epoch: 02 | Epoch Time: 0m 6s
	Train Loss: 0.651 | Train Acc: 75.04%
	 Val. Loss: 0.507 |  Val. Acc: 76.92%
Epoch: 03 | Epoch Time: 0m 6s
	Train Loss: 0.578 | Train Acc: 79.91%
	 Val. Loss: 0.424 |  Val. Acc: 80.97%
Epoch: 04 | Epoch Time: 0m 6s
	Train Loss: 0.501 | Train Acc: 83.97%
	 Val. Loss: 0.377 |  Val. Acc: 84.34%
Epoch: 05 | Epoch Time: 0m 6s
	Train Loss: 0.435 | Train Acc: 86.96%
	 Val. Loss: 0.363 |  Val. Acc: 86.18%

Estimated run time (NO CUDA)

'''


import torchtext
import torch
import torch.nn as nn
import torch.optim as optim
# https://stackoverflow.com/questions/66854921/legacy-torchtext-0-9-0
from torchtext.legacy import data
from torchtext.legacy import datasets
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F

import random
import time
import numpy as np
import spacy

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):

        #text = [sent len, batch size]

        embedded = self.embedding(text)

        #embedded = [sent len, batch size, emb dim]

        embedded = embedded.permute(1, 0, 2)

        #embedded = [batch size, sent len, emb dim]

        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

        #pooled = [batch size, embedding_dim]

        return self.fc(pooled)

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  preprocessing = generate_bigrams)

LABEL = data.LabelField(dtype = torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state = random.seed(SEED))

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = "glove.6B.100d",
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)

print(f'The model has {count_parameters(model):,} trainable parameters')

pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

model.load_state_dict(torch.load('tut3-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

nlp = spacy.load('en_core_web_sm')

def predict_sentiment(model, sentence):
    model.eval()
    tokenized = generate_bigrams([tok.text for tok in nlp.tokenizer(sentence)])
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


print(predict_sentiment(model, "This film is terrible"))
print(predict_sentiment(model, "This film is great"))

'''
The model has 2,500,301 trainable parameters
Epoch: 01 | Epoch Time: 0m 52s
	Train Loss: 0.686 | Train Acc: 61.58%
	 Val. Loss: 0.630 |  Val. Acc: 71.75%
Epoch: 02 | Epoch Time: 0m 53s
	Train Loss: 0.646 | Train Acc: 74.07%
	 Val. Loss: 0.511 |  Val. Acc: 76.08%
Epoch: 03 | Epoch Time: 0m 50s
	Train Loss: 0.572 | Train Acc: 79.89%
	 Val. Loss: 0.428 |  Val. Acc: 80.54%
Epoch: 04 | Epoch Time: 0m 53s
	Train Loss: 0.496 | Train Acc: 84.13%
	 Val. Loss: 0.389 |  Val. Acc: 83.80%
Epoch: 05 | Epoch Time: 0m 57s
	Train Loss: 0.431 | Train Acc: 87.00%
	 Val. Loss: 0.372 |  Val. Acc: 85.89%
Test Loss: 0.385 | Test Acc: 85.38%
1.2877721111692608e-09
1.0
Process finished with exit code 0

'''