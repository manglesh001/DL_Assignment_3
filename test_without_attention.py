import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
import random
import numpy as np

# =======================
# Fixed Best Configuration
# =======================
best_config = {
    'embedding_dim': 128,
    'hidden_dim': 256,
    'enc_layers': 3,
    'dec_layers': 3,
    'cell_type': 'LSTM',
    'dropout': 0.2,
    'epochs': 15,
    'beam_size': 5,
    'batch_size': 128,  # Added
    'learning_rate': 0.001  # Added
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.init(project="dakshina-transliteration", config=config)
config = wandb.config

# =======================
# Vocabulary
# =======================
class Vocab:
    def __init__(self):
        self.char2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        self.idx2char = {0: "<pad>", 1: "<sos>", 2: "<eos>"}
        self.size = 3

    def build(self, texts):
        for text in texts:
            for char in text:
                if char not in self.char2idx:
                    self.char2idx[char] = self.size
                    self.idx2char[self.size] = char
                    self.size += 1

    def encode(self, text):
        return [self.char2idx[c] for c in text]

    def decode(self, idxs):
        return ''.join([self.idx2char[i] for i in idxs if i > 2])

# =======================
# Dataset
# =======================
class TransliterationDataset(Dataset):
    def __init__(self, filepath, inp_vocab=None, out_vocab=None):
        self.pairs = []
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) < 2:
                    continue
                lat, dev = fields[0], fields[1]
                self.pairs.append((lat, dev))
        if inp_vocab and out_vocab:
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
# Encoder / Decoder / Seq2Seq
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
        outputs, hidden = self.rnn(packed)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, cell_type, dropout):
        super().__init__()
        self.cell_type = cell_type
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_class = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}[cell_type]
        self.rnn = rnn_class(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden):
        embedded = self.embedding(input_token).unsqueeze(1)
        output, hidden = self.rnn(embedded, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

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
            input_token = trg[:, t] if random.random() < teacher_forcing_ratio else top1
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
# Train / Evaluate
# =======================
def compute_accuracy(preds, targets):
    preds = preds.argmax(-1)
    correct = ((preds == targets) & (targets != 0)).sum().item()
    total = (targets != 0).sum().item()
    return correct / total

def train_eval(model, loader, criterion, optimizer, is_train):
    model.train() if is_train else model.eval()
    total_loss, total_acc = 0, 0
    with torch.set_grad_enabled(is_train):
        for src, trg, src_lens, _ in loader:
            src, trg = src.to(device), trg.to(device)
            if is_train: optimizer.zero_grad()
            output = model((src, src_lens), trg)
            loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))
            acc = compute_accuracy(output[:, 1:], trg[:, 1:])
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_acc += acc
    return total_loss / len(loader), total_acc / len(loader)

# =======================
# Train and Save Best
# =======================
inp_vocab, out_vocab = Vocab(), Vocab()
train_set = TransliterationDataset("/kaggle/input/dataset-01/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv", inp_vocab, out_vocab)
dev_set = TransliterationDataset("/kaggle/input/dataset-01/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv", inp_vocab, out_vocab)
test_set = TransliterationDataset("/kaggle/input/dataset-01/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv", inp_vocab, out_vocab)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

encoder = Encoder(inp_vocab.size, config.embedding_dim, config.hidden_dim, config.enc_layers, config.cell_type, config.dropout)
decoder = Decoder(out_vocab.size, config.embedding_dim, config.hidden_dim, config.dec_layers, config.cell_type, config.dropout)
model = Seq2Seq(encoder, decoder, config.enc_layers, config.dec_layers, config.cell_type, device).to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)

best_val_acc = 0.0
for epoch in range(config.epochs):
    train_loss, train_acc = train_eval(model, train_loader, criterion, optimizer, is_train=True)
    val_loss, val_acc = train_eval(model, dev_loader, criterion, optimizer, is_train=False)
    wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch})
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.4f}, Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")

# =======================
# Test Evaluation
# =======================
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

os.makedirs("predictions_vanilla", exist_ok=True)
with open("predictions_vanilla/preds.txt", "w", encoding="utf-8") as f:
    correct, total = 0, 0
    samples = []
    for src, trg, src_lens, _ in test_loader:
        src, trg = src.to(device), trg.to(device)
        output = model((src, src_lens), trg, teacher_forcing_ratio=0.0)
        pred_idxs = output.argmax(-1)[0].tolist()
        true_idxs = trg[0].tolist()
        pred_str = out_vocab.decode(pred_idxs)
        true_str = out_vocab.decode(true_idxs)
        input_str = inp_vocab.decode(src[0].tolist())
        f.write(f"{input_str}\t{true_str}\t{pred_str}\n")
        if pred_str == true_str:
            correct += 1
        total += 1
        samples.append((input_str, true_str, pred_str))
    test_acc = correct / total
    wandb.log({"test_accuracy": test_acc})
    print("Test Accuracy:", test_acc)

# Sample Grid (for visualization)
print("\nSample Predictions:")
print("{:<20} | {:<20} | {:<20}".format("Input", "Reference", "Prediction"))
print("=" * 65)
for s in random.sample(samples, 10):
    print("{:<20} | {:<20} | {:<20}".format(*s))

