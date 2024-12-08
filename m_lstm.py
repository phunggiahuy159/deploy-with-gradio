
import torch
import torch.nn as nn
from transformers import AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aspect2idx = {
    'CAMERA': 0, 'FEATURES': 1, 'BATTERY': 2, 'PERFORMANCE': 3,
    'DESIGN': 4, 'GENERAL': 5, 'PRICE': 6, 'SCREEN': 7, 'SER&ACC': 8, 'STORAGE': 9
}
sentiment2idx = {
    'Positive': 2, 'Neutral': 1, 'Negative': 0
}
num_aspect = len(aspect2idx)

idx2aspect = dict(zip(aspect2idx.values(), aspect2idx.keys()))
idx2sentiment = dict(zip(sentiment2idx.values(),sentiment2idx.keys()))

tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')

class Cae(nn.Module):
    def __init__(self, word_embedder, categories, polarities):
        super().__init__()
        self.word_embedder = word_embedder
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss(ignore_index=-1)

        embed_dim = word_embedder.embedding_dim
        
        self.lstm = nn.LSTM(embed_dim, embed_dim // 2, batch_first=True, bidirectional=True)
        self.dropout_after_lstm = nn.Dropout(0.5)

        self.category_fcs = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in range(self.category_num)])
        self.sentiment_fc = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, self.polarity_num)) for _ in range(self.category_num)])


    def forward(self, tokens, labels, mask, threshold=0.4):
        word_embeddings = self.word_embedder(tokens)

        embeddings = word_embeddings
        

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm(lstm_result)


         # Pool the LSTM output (mean pooling) for each token's hidden states
        pooled_output = lstm_result.mean(dim=1) # batch_size * embed_size
        # print(pooled_output)
        # print(pooled_output.shape)
        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            # Category and sentiment predictions
            category_output = self.category_fcs[i](pooled_output)
            sentiment_output = self.sentiment_fc[i](pooled_output)
            final_category_outputs.append(category_output)
            final_sentiment_outputs.append(sentiment_output)

        loss = 0
        if labels is not None:
            category_labels = labels[:, :self.category_num]
            polarity_labels = labels[:, self.category_num:]

            for i in range(self.category_num):
                category_mask = (category_labels[:, i] != -1)  # Mask out ignored labels
                sentiment_mask = (polarity_labels[:, i] != -1)

                if category_mask.any():  # Only calculate if there are valid labels
                    category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(-1)[category_mask], category_labels[:, i][category_mask])
                    loss += category_temp_loss

                if sentiment_mask.any():  # Only calculate if there are valid labels
                    sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i][sentiment_mask], polarity_labels[:, i][sentiment_mask].long())
                    loss += sentiment_temp_loss

        # formatting output
        final_category_outputs = [torch.sigmoid(e) for e in final_category_outputs]
        final_sentiment_outputs = [torch.softmax(e, dim=-1) for e in final_sentiment_outputs]
        final_sentiment_outputs = [torch.argmax(e, dim=-1) for e in final_sentiment_outputs]
        
        final_categories = []
        final_sentiments = []

        for i in range(len(final_category_outputs)):
            batch_category = []
            batch_sentiment = []
            for j, category_score in enumerate(final_category_outputs[i]):
                # Apply threshold for aspect detection
                if category_score >= threshold:
                    batch_category.append(1)  # Aspect detected
                    batch_sentiment.append(final_sentiment_outputs[i][j].item())
                else:
                    batch_category.append(0)  # Aspect not detected
                    batch_sentiment.append(-1)  # Set sentiment to -1 for undetected aspect
            final_categories.append(batch_category)
            final_sentiments.append(batch_sentiment)
        final_categories = torch.tensor(final_categories)
        final_sentiments = torch.tensor(final_sentiments)
        
        output = {
            'pred_category': torch.transpose(final_categories, 0, 1), # batch_size*10
            'pred_sentiment': torch.transpose(final_sentiments, 0, 1) # batch_size*10
        }

        return output


def infer_LSTM_model(comment):
    model.eval()
    # Tokenize input text
    encoding = tokenizer(comment, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # No labels provided in inference
    with torch.no_grad():
        output = model(input_ids, labels=None, mask=attention_mask)

    pred_category = output['pred_category']
    pred_sentiment = output['pred_sentiment']

    res = ''
    pred_category = pred_category.squeeze()
    pred_sentiment = pred_sentiment.squeeze()
    for i, v in enumerate(pred_category):
        if v!=0:
            res += f'{idx2aspect[i]}: {idx2sentiment[int(pred_sentiment[i])]}'
            res += '\n'

    return res

embedding_dim = 150
vocab = tokenizer.get_vocab()
vocab_size = len(vocab)

embedding_layer = nn.Embedding(vocab_size, embedding_dim)

categories = aspect2idx.keys()
polarities = sentiment2idx.keys()
model = Cae(embedding_layer, categories, polarities)

model.load_state_dict(torch.load("D:\code\intro_ai_ABSA\CAE_checkpoint50.pth",map_location=torch.device('cpu')))
model.eval()

