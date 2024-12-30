import torch
import torch.nn as nn
from transformers import AutoTokenizer
import math 
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aspect2idx = {
    'BATTERY': 0, 'CAMERA': 1, 'DESIGN': 2, 'FEATURES': 3,
    'GENERAL': 4, 'PERFORMANCE': 5, 'PRICE': 6, 'SCREEN': 7, 'SER&ACC': 8, 'STORAGE': 9
}
sentiment2idx = {
    'Positive': 0, 'Negative': 1, 'Neutral': 2
}
num_aspect = len(aspect2idx)

idx2aspect = dict(zip(aspect2idx.values(), aspect2idx.keys()))
idx2sentiment = dict(zip(sentiment2idx.values(),sentiment2idx.keys()))

tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
EMBEDDING_DIM = 300

class Attention(nn.Module):
    def __init__(self, embed_dim=EMBEDDING_DIM, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        if score_function == 'nl':
            self.weight = nn.parameter(torch.Tensor(hidden_dim, hidden_dim))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2: 
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  
        k_len = k.shape[1]
        q_len = q.shape[1]
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        # in sentiment analysis, they focus to the importance of k, so maybe we dont have V  value (intuitively, V is k...)
        output = torch.bmm(score, kx) 
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  
        output = self.proj(output)  
        output = self.dropout(output)
        return output, score


class NoQueryAttention(Attention):
    '''q is a parameter'''
    def __init__(self, embed_dim=EMBEDDING_DIM, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1, dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)
    
class ATAE_LSTM(nn.Module):
    def __init__(self, embed_matrix, hidden_dim=128, embedding_dim=EMBEDDING_DIM, polarities_dim=3, num_aspects=10):
        super(ATAE_LSTM, self).__init__()
        
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embed_matrix, dtype=torch.float))
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_aspects = num_aspects
        self.polarities_dim = polarities_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(hidden_dim, score_function='bi_linear')

        self.category_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(num_aspects)
        ])

        self.sentiment_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, polarities_dim)
            ) for _ in range(num_aspects)
        ])

        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, text_indices, labels=None, threshold=0.25):
        x = self.embed(text_indices)
        h, _ = self.lstm(x)
        _, score = self.attention(h)
        pooled_output = torch.bmm(score, h).squeeze(dim=1)  # Shape: (batch_size, hidden_dim)

        final_category_outputs = []
        final_sentiment_outputs = []

        for i in range(self.num_aspects):
            # Category and sentiment predictions
            category_output = self.category_fcs[i](pooled_output)
            sentiment_output = self.sentiment_fc[i](pooled_output)
            final_category_outputs.append(category_output)
            final_sentiment_outputs.append(sentiment_output)

        loss = 0
        if labels is not None:
            category_labels = labels[:, :self.num_aspects]
            polarity_labels = labels[:, self.num_aspects:]

            for i in range(self.num_aspects):
                if polarity_labels.size(1) <= i:  
                    continue

                category_mask = (category_labels[:, i] != -1)  
                sentiment_mask = (polarity_labels[:, i] != -1)

                if category_mask.any():  
                    category_temp_loss = self.category_loss(
                        final_category_outputs[i].squeeze(-1)[category_mask],
                        category_labels[:, i][category_mask]
                    )
                    loss += category_temp_loss

                if sentiment_mask.any():  
                    sentiment_temp_loss = self.sentiment_loss(
                        final_sentiment_outputs[i][sentiment_mask],
                        polarity_labels[:, i][sentiment_mask].long()
                    )
                    loss += sentiment_temp_loss

        final_category_outputs = [torch.sigmoid(e) for e in final_category_outputs]
        final_sentiment_outputs = [torch.softmax(e, dim=-1) for e in final_sentiment_outputs]
        
        
        final_categories = []
        final_sentiments = []

        for i in range(len(final_category_outputs)):
            batch_category = []
            batch_sentiment = []
            for j, category_score in enumerate(final_category_outputs[i]):
                if category_score >= threshold:
                    batch_category.append(1)  
                    batch_sentiment.append(torch.argmax(final_sentiment_outputs[i][j]).item())
                else:
                    batch_category.append(0)  
                    batch_sentiment.append(-1)  
            final_categories.append(batch_category)
            final_sentiments.append(batch_sentiment)

        final_categories = torch.tensor(final_categories)
        final_sentiments = torch.tensor(final_sentiments)

        output = {
            'pred_category': torch.transpose(final_categories, 0, 1),  # (batch_size, num_aspects)
            'pred_sentiment': torch.transpose(final_sentiments, 0, 1)  # (batch_size, num_aspects)
        }

        return output
    
def infer_LSTM_attention_model(comment):
    model.eval()
    # Tokenize input text
    encoding = tokenizer(comment, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)

    # No labels provided in inference
    with torch.no_grad():
        output = model(input_ids)

    pred_category = output['pred_category'].cpu().numpy()
    pred_sentiment = output['pred_sentiment'].cpu().numpy()

    res = ''
    for i, v in enumerate(pred_category[0]):  # [0] for batch size 1
        if v == 1:  # Aspect detected
            res += f'{idx2aspect[i]}: {idx2sentiment[int(pred_sentiment[0][i])]}'
            res += '\n'

    return res

# Load model checkpoint
""""
checkpoint = torch.load("UI/ATAE_checkpoint20.pth", map_location=device)
embed_matrix = checkpoint['embed_matrix']
model = ATAE_LSTM(embed_matrix)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
"""
# Placeholder for the embedding matrix (shape doesn't matter for loading weights)
vocab_size = len(tokenizer)
dummy_embed_matrix = torch.empty(vocab_size-1, EMBEDDING_DIM)  # Replace vocab_size and EMBEDDING_DIM as needed

# Initialize the model with the dummy matrix
model = ATAE_LSTM(dummy_embed_matrix)

# Load the saved state_dict
checkpoint = torch.load("D:\code\deploy gradio\deploy-with-gradio\ATAE_checkpoint20.pth", torch.device('cpu'))
model.load_state_dict(checkpoint)

# Move the model to the appropriate device
model.to(device)
model.eval()

# Example Inference
# comment = "Điện thoại có camera đẹp, giá cả hợp lý"
# result = infer_LSTM_attention_model(comment)
# print(result)