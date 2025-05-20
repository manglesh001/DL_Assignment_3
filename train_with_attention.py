import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import math


!wandb login 58a0b576fd5221cd0d63b154deaabbe535e853c6

# =======================
# Vocabulary
# =======================
class Vocab:
    def __init__(self, level='char'):
        self.level = level
        self.char2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        self.idx2char = {0: "<pad>", 1: "<sos>", 2: "<eos>"}
        self.size = 3

    def build(self, texts):
        for text in texts:
            if self.level == 'char':
                tokens = list(text)
            else:
                tokens = text.split()
            for token in tokens:
                if token not in self.char2idx:
                    self.char2idx[token] = self.size
                    self.idx2char[self.size] = token
                    self.size += 1

    def encode(self, text):
        if self.level == 'char':
            tokens = list(text)
        else:
            tokens = text.split()
        return [self.char2idx.get(token, 2) for token in tokens]  # Fallback to <eos> for unknown tokens

    def decode(self, idxs):
        tokens = []
        for i in idxs:
            if i > 2:
                tokens.append(self.idx2char.get(i, ''))
        if self.level == 'char':
            return ''.join(tokens)
        else:
            return ' '.join(tokens)

# =======================
# Dataset
# =======================
class TransliterationDataset(Dataset):
    def __init__(self, filepath, inp_vocab, out_vocab):
        self.pairs = []
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) < 2:
                    continue
                lat, dev = fields[0], fields[1]
                self.pairs.append((lat, dev))
        inp_vocab.build([p[0] for p in self.pairs])
        out_vocab.build([p[1] for p in self.pairs])
        self.inp_vocab = inp_vocab
        self.out_vocab = out_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lat, dev = self.pairs[idx]
        x = self.inp_vocab.encode(lat)
        y = [self.out_vocab.char2idx["<sos>"]] + self.out_vocab.encode(dev) + [self.out_vocab.char2idx["<eos>"]]
        return torch.tensor(x), torch.tensor(y)

def collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    x_lens = [len(x) for x in x_batch]
    y_lens = [len(y) for y in y_batch]
    x_pad = nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=0)
    y_pad = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=0)
    return x_pad, y_pad, torch.tensor(x_lens), torch.tensor(y_lens)

# =======================
# Attention Mechanism
# =======================
class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_type='general'):
        super().__init__()
        self.attention_type = attention_type
        
        if attention_type == 'general':
            self.attn = nn.Linear(hidden_dim, hidden_dim)
        elif attention_type == 'concat':
            self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)
            
    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size, src_len, hidden_dim = encoder_outputs.shape
        
        if self.attention_type == 'dot':
            energy = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'general':
            energy = torch.bmm(encoder_outputs, self.attn(hidden).unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'concat':
            hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
            energy = self.v(torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), dim=2)))).squeeze(2)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention_weights = F.softmax(energy, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights

# =======================
# Encoder and Decoder
# =======================
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, cell_type, dropout):
        super().__init__()
        self.cell_type = cell_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_class = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}[cell_type]
        self.rnn = rnn_class(emb_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, cell_type, dropout, attention_type):
        super().__init__()
        self.cell_type = cell_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_class = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}[cell_type]
        self.rnn = rnn_class(emb_dim + hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_dim, attention_type)
        self.out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, encoder_outputs, mask=None):
        if self.cell_type == "LSTM":
            attn_hidden = hidden[0][-1]
        else:
            attn_hidden = hidden[-1]
        
        context, attention_weights = self.attention(attn_hidden, encoder_outputs, mask)
        embedded = self.embedding(input_token)
        rnn_input = torch.cat((embedded, context), dim=1).unsqueeze(1)
        output, hidden = self.rnn(rnn_input, hidden)
        
        if self.cell_type == "LSTM":
            output_hidden = hidden[0][-1]
        else:
            output_hidden = hidden[-1]
        
        output = torch.cat((output_hidden, context), dim=1)
        output = self.dropout(output)
        prediction = self.out(output)
        return prediction, hidden, attention_weights

# =======================
# Seq2Seq Model with Beam Search
# =======================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, enc_layers, dec_layers, cell_type, device, beam_size=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type = cell_type
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.device = device
        self.beam_size = beam_size

    def create_mask(self, src, src_lens):
        batch_size = src.size(0)
        max_len = src.size(1)
        mask = torch.zeros(batch_size, max_len, device=self.device)
        for i, length in enumerate(src_lens):
            mask[i, :length] = 1
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        src_data, src_lens = src
        batch_size, trg_len = trg.size()
        vocab_size = self.decoder.out.out_features
        
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src_data.size(1)).to(self.device)
        
        encoder_outputs, enc_hidden = self.encoder(src_data, src_lens)
        mask = self.create_mask(src_data, src_lens)
        
        if self.cell_type == "LSTM":
            h, c = enc_hidden
            h = self._match_layers(h)
            c = self._match_layers(c)
            dec_hidden = (h, c)
        else:
            dec_hidden = self._match_layers(enc_hidden)
        
        input_token = trg[:, 0]
        
        for t in range(1, trg_len):
            output, dec_hidden, attn_weights = self.decoder(
                input_token, dec_hidden, encoder_outputs, mask
            )
            outputs[:, t] = output
            attentions[:, t] = attn_weights
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
            
        return outputs, attentions

    def _match_layers(self, hidden):
        if self.enc_layers == self.dec_layers:
            return hidden
        elif self.enc_layers > self.dec_layers:
            return hidden[:self.dec_layers]
        else:
            pad = hidden.new_zeros((self.dec_layers - self.enc_layers, *hidden.shape[1:]))
            return torch.cat([hidden, pad], dim=0)

# =======================
# Train & Eval
# =======================
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_count = 0, 0, 0
    for src, trg, src_lens, _ in loader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output, _ = model((src, src_lens), trg)
        output_dim = output.shape[-1]
        loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
        pred = output.argmax(2)
        correct = ((pred[:, 1:] == trg[:, 1:]) & (trg[:, 1:] != 0)).sum().item()
        total_correct += correct
        total_count += (trg[:, 1:] != 0).sum().item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    acc = 100.0 * total_correct / total_count
    print(f"Train Loss: {total_loss / len(loader):.4f}, Acc: {acc:.2f}%")
    return total_loss / len(loader), acc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0, 0, 0
    with torch.no_grad():
        for src, trg, src_lens, _ in loader:
            src, trg = src.to(device), trg.to(device)
            output, _ = model((src, src_lens), trg, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            pred = output.argmax(2)
            correct = ((pred[:, 1:] == trg[:, 1:]) & (trg[:, 1:] != 0)).sum().item()
            total_correct += correct
            total_count += (trg[:, 1:] != 0).sum().item()
            total_loss += loss.item()
    acc = 100.0 * total_correct / total_count
    print(f"Val Loss: {total_loss / len(loader):.4f}, Acc: {acc:.2f}%")
    return total_loss / len(loader), acc

# =======================
# Main
# =======================
def main(args):
    wandb.init(project="dakshina-transliteration-attention", config=args)
    device = torch.device(args.device)

    # Initialize vocab with specified level
    inp_vocab = Vocab(level=args.level)
    out_vocab = Vocab(level=args.level)
    
    # Load datasets
    train_data = TransliterationDataset(args.train_path, inp_vocab, out_vocab)
    dev_data = TransliterationDataset(args.dev_path, inp_vocab, out_vocab)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_fn)

    # Initialize model
    encoder = Encoder(inp_vocab.size, args.embedding_dim, args.hidden_dim,
                     args.enc_layers, args.cell_type, args.dropout)
    decoder = AttentionDecoder(out_vocab.size, args.embedding_dim, args.hidden_dim,
                              args.dec_layers, args.cell_type, args.dropout,
                              args.attention_type)
    
    model = Seq2Seq(encoder, decoder, args.enc_layers, args.dec_layers,
                   args.cell_type, device, beam_size=args.beam_size).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, dev_loader, criterion, device)
        wandb.log({
            "train_loss": train_loss, 
            "train_acc": train_acc, 
            "val_loss": val_loss, 
            "val_acc": val_acc, 
            "epoch": epoch+1
        })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seq2Seq Transliteration with Attention')
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--enc_layers', type=int, default=1)
    parser.add_argument('--dec_layers', type=int, default=1)
    parser.add_argument('--cell_type', choices=['GRU', 'LSTM', 'RNN'], default='LSTM')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--attention_type', choices=['dot', 'general', 'concat'], default='general')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--level', choices=['char', 'word'], default='char',
                       help='Processing level: character or word')
    parser.add_argument('--train_path', type=str, 
                       default='/kaggle/input/devnagiridata/hi.translit.sampled.train.tsv')
    parser.add_argument('--dev_path', type=str,
                       default='/kaggle/input/devnagiridata/hi.translit.sampled.dev.tsv')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    
    args = parser.parse_args()
    main(args)
