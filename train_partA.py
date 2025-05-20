# =======================
# Imports
# =======================
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import json
import numpy as np

!wandb login 58a0b576fd5221cd0d63b154deaabbe535e853c6

# =======================
# Sweep & Default Configs
# =======================
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'embedding_dim': {'values': [64, 128, 256]},
        'hidden_dim': {'values': [64, 128, 256]},
        'enc_layers': {'values': [1, 2, 3]},
        'dec_layers': {'values': [1, 2, 3]},
        'cell_type': {'values': ['GRU', 'LSTM', 'RNN']},
        'dropout': {'values': [0.2, 0.3, 0.5]},
        'epochs': {'values': [15, 20]},
        'beam_size': {'values': [1, 3, 5]},
        'batch_size': {'values': [64, 128]},
        'learning_rate': {'values': [0.001, 0.0005, 0.0001]}
    }
}

default_config = {
    'embedding_dim': 128,
    'hidden_dim': 128,
    'enc_layers': 2,
    'dec_layers': 2,
    'cell_type': 'LSTM',
    'dropout': 0.2,
    'epochs': 10,
    'beam_size': 1,
    'batch_size': 64,
    'learning_rate': 0.001
}

# =======================
# Vocabulary Class
# =======================
class Vocab:
    def __init__(self, level='char'):
        self.level = level
        self.token2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        self.idx2token = {0: "<pad>", 1: "<sos>", 2: "<eos>"}
        self.size = 3

    def build(self, texts):
        for text in texts:
            tokens = list(text) if self.level == 'char' else text.strip().split()
            for token in tokens:
                if token not in self.token2idx:
                    self.token2idx[token] = self.size
                    self.idx2token[self.size] = token
                    self.size += 1

    def encode(self, text):
        tokens = list(text) if self.level == 'char' else text.strip().split()
        return [self.token2idx[token] for token in tokens]

    def decode(self, indices):
        return ' '.join([self.idx2token[idx] for idx in indices if idx > 2])

# =======================
# Dataset Class
# =======================
class TransliterationDataset(Dataset):
    def __init__(self, filepath, inp_vocab, out_vocab):
        self.pairs = []
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) < 2:
                    continue
                src, tgt = fields[0], fields[1]
                self.pairs.append((src, tgt))
        inp_vocab.build([p[0] for p in self.pairs])
        out_vocab.build([p[1] for p in self.pairs])
        self.inp_vocab = inp_vocab
        self.out_vocab = out_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        x = self.inp_vocab.encode(src)
        y = [self.out_vocab.token2idx["<sos>"]] + self.out_vocab.encode(tgt) + [self.out_vocab.token2idx["<eos>"]]
        return torch.tensor(x), torch.tensor(y)

def collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    x_lens = [len(x) for x in x_batch]
    y_lens = [len(y) for y in y_batch]
    x_pad = nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=0)
    y_pad = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=0)
    return x_pad, y_pad, torch.tensor(x_lens), torch.tensor(y_lens)

# =======================
# Encoder / Decoder
# =======================
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_class = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}[cell_type]
        self.rnn = rnn_class(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_class = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}[cell_type]
        self.rnn = rnn_class(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden):
        embedded = self.embedding(input_token).unsqueeze(1)
        output, hidden = self.rnn(embedded, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

# =======================
# Seq2Seq Wrapper
# =======================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, enc_layers, dec_layers, cell_type, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type = cell_type
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.size()
        vocab_size = self.decoder.out.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        enc_hidden = self.encoder(src[0], src[1])

        if self.cell_type == "LSTM":
            h, c = enc_hidden
            h = self._match_layers(h)
            c = self._match_layers(c)
            dec_hidden = (h, c)
        else:
            dec_hidden = self._match_layers(enc_hidden)

        input_token = trg[:, 0]
        for t in range(1, trg_len):
            output, dec_hidden = self.decoder(input_token, dec_hidden)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_token = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1
        return outputs

    def _match_layers(self, hidden):
        if self.enc_layers == self.dec_layers:
            return hidden
        elif self.enc_layers > self.dec_layers:
            return hidden[:self.dec_layers]
        else:
            pad = hidden.new_zeros((self.dec_layers - self.enc_layers, *hidden.shape[1:]))
            return torch.cat([hidden, pad], dim=0)

# =======================
# Train & Eval Functions
# =======================
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_count = 0, 0, 0
    for src, trg, src_lens, _ in loader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model((src, src_lens), trg)
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
    return total_loss / len(loader), acc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0, 0, 0
    with torch.no_grad():
        for src, trg, src_lens, _ in loader:
            src, trg = src.to(device), trg.to(device)
            output = model((src, src_lens), trg, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            pred = output.argmax(2)
            correct = ((pred[:, 1:] == trg[:, 1:]) & (trg[:, 1:] != 0)).sum().item()
            total_correct += correct
            total_count += (trg[:, 1:] != 0).sum().item()
            total_loss += loss.item()
    acc = 100.0 * total_correct / total_count
    return total_loss / len(loader), acc

# =======================
# Main Function
# =======================
def main(args):
    wandb.init(config=default_config, project="dakshina-transliteration")
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inp_vocab = Vocab(level=args.level)
    out_vocab = Vocab(level=args.level)

    train_data = TransliterationDataset(args.train_path, inp_vocab, out_vocab)
    dev_data = TransliterationDataset(args.dev_path, inp_vocab, out_vocab)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(inp_vocab.size, config.embedding_dim, config.hidden_dim, config.enc_layers, config.cell_type, config.dropout)
    decoder = Decoder(out_vocab.size, config.embedding_dim, config.hidden_dim, config.dec_layers, config.cell_type, config.dropout)
    model = Seq2Seq(encoder, decoder, config.enc_layers, config.dec_layers, config.cell_type, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, dev_loader, criterion, device)
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch + 1})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='char', choices=['char', 'word'], help='Tokenization level')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--dev_path', type=str, required=True, help='Path to dev data')
    parser.add_argument('--sweep', action='store_true', help='Enable WandB hyperparameter sweep')

    # Add model/training hyperparameters with default values from default_config
    parser.add_argument('--embedding_dim', type=int, default=default_config['embedding_dim'])
    parser.add_argument('--hidden_dim', type=int, default=default_config['hidden_dim'])
    parser.add_argument('--enc_layers', type=int, default=default_config['enc_layers'])
    parser.add_argument('--dec_layers', type=int, default=default_config['dec_layers'])
    parser.add_argument('--cell_type', type=str, default=default_config['cell_type'], choices=['GRU', 'LSTM', 'RNN'])
    parser.add_argument('--dropout', type=float, default=default_config['dropout'])
    parser.add_argument('--epochs', type=int, default=default_config['epochs'])
    parser.add_argument('--beam_size', type=int, default=default_config['beam_size'])
    parser.add_argument('--batch_size', type=int, default=default_config['batch_size'])
    parser.add_argument('--learning_rate', type=float, default=default_config['learning_rate'])

    args = parser.parse_args()

    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, project="dakshina-transliteration")
        wandb.agent(sweep_id, function=lambda: main(args), count=30)
    else:
        # Use argparse values to override default_config before calling main
        wandb.init(
            config={
                k: getattr(args, k)
                for k in default_config.keys()
            },
            project="dakshina-transliteration"
        )
        main(args)


