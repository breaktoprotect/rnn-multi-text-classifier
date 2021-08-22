'''
Adapted from Janani Ravi's Pluralsight Course 'Natural Language Processing with PyTorch'
'''
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, nn_type="rnn"):
        super().__init__()

        self.type = nn_type # Normal RNN or LSTM
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        if self.type.lower() == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        else:
            self.rnn = nn.RNN(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)

        if self.type.lower() == "lstm":
            output, (hidden, _) = self.rnn(embedded) # Additional last cell state from LTSM cell
        else:
            output, hidden = self.rnn(embedded)
            
        hidden_1D = hidden.squeeze(0)
        assert torch.equal(output[-1, :, :], hidden_1D)

        return self.fc(hidden_1D)

    # Helper function to perform (batched) training
    @staticmethod
    def train_once(model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0

        model.train()

        for batch in iterator:
            optimizer.zero_grad()
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            #rounded_preds = torch.round(torch.sigmoid(predictions))
            predicted_label = predictions.argmax(1)
            

            #debug
            #print("[DEBUG] predictions=",predictions)
            #print("[DEBUG] batch.label=", batch.label)
            #print("[DEBUG] rounded_preds=",rounded_preds)
            #print("[DEBUG] predicted_label=", predicted_label)


            correct = (predicted_label == batch.label).float()
            

            acc = correct.sum() / len(correct)

            loss.backward() # backward propagation?

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    # Helper function to evaluate model
    @staticmethod
    def evaluate(model, test_iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0
        model.eval()

        with torch.no_grad():
            for batch in test_iterator:
                predictions = model(batch.text).squeeze(1)
                loss = criterion(predictions, batch.label)
                #rounded_preds = torch.round(torch.sigmoid(predictions))
                predicted_label = predictions.argmax(1)
                correct = (predicted_label == batch.label).float()
                acc = correct.sum() / len(correct)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        test_loss = epoch_loss / len(test_iterator)
        test_acc = epoch_acc / len(test_iterator)

        return test_loss, test_acc 


    #? EXPERIMENTAL
    @staticmethod
    def predict(model, TEXT, LABEL, sentence):
        import spacy
        nlp = spacy.load('en_core_web_sm')
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence 
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
        length = [len(indexed)]                                    #compute no. of words
        tensor = torch.LongTensor(indexed)                         #convert to tensor
        tensor = tensor.unsqueeze(1)                             #reshape in form of batch,no. of words
        length_tensor = torch.LongTensor(length)                   #convert to tensor
        with torch.no_grad():
            prediction = torch.sigmoid(model(tensor))                  #prediction 

        label_class_list = LABEL.vocab.itos

        #debug
        print("[DEBUG] prediction.argmax(1) = ", prediction)

        return LABEL.vocab.itos[prediction.argmax(1)]        