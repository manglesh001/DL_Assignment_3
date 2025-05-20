import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

# =======================
# Best Configuration
# =======================
best_config = {
    'embedding_dim': 256,
    'hidden_dim': 256,
    'enc_layers': 2,
    'dec_layers': 2,
    'cell_type': 'LSTM',
    'dropout': 0.5,
    'epochs': 15,
    'beam_size': 5,
    'attention_type': 'concat',
    'batch_size': 256,
    'learning_rate': 0.001
}

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
        return ''.join([self.idx2char[i] for i in idxs if i not in [0, 1, 2]])

# =======================
# Dataset
# =======================
class TransliterationDataset(Dataset):
    def __init__(self, filepath, inp_vocab, out_vocab, is_test=False):
        self.pairs = []
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) < 2:
                    continue
                lat, dev = fields[0], fields[1]
                self.pairs.append((lat, dev))
        if not is_test:
            inp_vocab.build([p[0] for p in self.pairs])
            out_vocab.build([p[1] for p in self.pairs])
        self.inp_vocab = inp_vocab
        self.out_vocab = out_vocab
        self.is_test = is_test

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lat, dev = self.pairs[idx]
        x = self.inp_vocab.encode(lat)
        if self.is_test:
            return torch.tensor(x), lat, dev
        y = [self.out_vocab.char2idx["<sos>"]] + self.out_vocab.encode(dev) + [self.out_vocab.char2idx["<eos>"]]
        return torch.tensor(x), torch.tensor(y), lat, dev

def collate_fn(batch):
    if len(batch[0]) == 3:  # Test batch
        x_batch, lat, dev = zip(*batch)
        x_lens = [len(x) for x in x_batch]
        x_pad = nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=0)
        return x_pad, lat, dev, torch.tensor(x_lens)
    else:  # Train/val batch
        x_batch, y_batch, lat, dev = zip(*batch)
        x_lens = [len(x) for x in x_batch]
        y_lens = [len(y) for y in y_batch]
        x_pad = nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=0)
        y_pad = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=0)
        return x_pad, y_pad, torch.tensor(x_lens), torch.tensor(y_lens), lat, dev

# =======================
# Model Components
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
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden

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
        batch_size, src_len, hidden_dim = encoder_outputs.size()
        
        if self.attention_type == 'general':
            energy = torch.bmm(encoder_outputs, self.attn(hidden).unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'concat':
            hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
            concat = torch.cat((hidden_expanded, encoder_outputs), dim=2)
            energy = self.v(torch.tanh(self.attn(concat))).squeeze(2)
        else:  # dot
            energy = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention_weights = F.softmax(energy, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, cell_type, dropout, attention_type):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_class = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}[cell_type]
        self.rnn = rnn_class(emb_dim + hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_dim, attention_type)
        self.out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, encoder_outputs, mask=None):
        if isinstance(hidden, tuple):  # LSTM
            attn_hidden = hidden[0][-1]
        else:  # GRU/RNN
            attn_hidden = hidden[-1]
        
        context, attn_weights = self.attention(attn_hidden, encoder_outputs, mask)
        embedded = self.embedding(input_token)
        rnn_input = torch.cat((embedded, context), dim=1).unsqueeze(1)
        output, hidden = self.rnn(rnn_input, hidden)
        
        if isinstance(hidden, tuple):
            output_hidden = hidden[0][-1]
        else:
            output_hidden = hidden[-1]
        
        output = torch.cat((output_hidden, context), dim=1)
        output = self.dropout(output)
        prediction = self.out(output)
        return prediction, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src_lens, max_len):
        batch_size = len(src_lens)
        mask = torch.zeros(batch_size, max_len, device=self.device)
        for i, length in enumerate(src_lens):
            mask[i, :length] = 1
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        src_data, src_lens = src
        encoder_outputs, enc_hidden = self.encoder(src_data, src_lens)
        batch_size, trg_len = trg.size()
        vocab_size = self.decoder.out.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        
        src_len = encoder_outputs.size(1)
        mask = self.create_mask(src_lens, src_len)

        if isinstance(enc_hidden, tuple):
            dec_hidden = enc_hidden
        else:
            dec_hidden = enc_hidden

        input_token = trg[:, 0]
        for t in range(1, trg_len):
            output, dec_hidden, _ = self.decoder(input_token, dec_hidden, encoder_outputs, mask)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
            
        return outputs

    def predict(self, src, src_lens, max_len=30):
        self.eval()
        with torch.no_grad():
            encoder_outputs, enc_hidden = self.encoder(src, src_lens)
            src_len = encoder_outputs.size(1)
            mask = self.create_mask(src_lens.tolist(), src_len)
            
            if isinstance(enc_hidden, tuple):
                dec_hidden = enc_hidden
            else:
                dec_hidden = enc_hidden
            
            input_token = torch.tensor([1], device=self.device)  # <sos> token
            output_seq = []
            attention_weights = []
            
            for _ in range(max_len):
                output, dec_hidden, attn_weights = self.decoder(input_token, dec_hidden, encoder_outputs, mask)
                top1 = output.argmax(1)
                if top1.item() == 2:  # <eos> token
                    break
                output_seq.append(top1.item())
                attention_weights.append(attn_weights.cpu().numpy())
                input_token = top1
                
        return output_seq, attention_weights

# =======================
# Training and Evaluation with Word-Level Accuracy
# =======================
def train(model, loader, criterion, optimizer, device, out_vocab):
    model.train()
    total_loss = 0
    total_correct_chars = 0
    total_chars = 0
    
    # Word-level tracking
    total_correct_words = 0
    total_words = 0
    
    for batch in loader:
        src, trg, src_lens, _, lat, dev = batch
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        output = model((src, src_lens), trg)
        
        # Calculate loss
        output_dim = output.shape[-1]
        loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        
        # Calculate character-level accuracy
        pred = output.argmax(dim=2)
        mask = (trg[:, 1:] != 0)  # Ignore padding
        correct_chars = ((pred[:, 1:] == trg[:, 1:]) & mask).sum().item()
        total_correct_chars += correct_chars
        total_chars += mask.sum().item()
        
        # Calculate word-level accuracy
        batch_size = trg.size(0)
        for i in range(batch_size):
            # Get predicted sequence without padding, sos, eos
            pred_seq = [idx.item() for idx in pred[i, 1:] if idx.item() not in [0, 1, 2]]
            # Convert to string
            pred_word = out_vocab.decode(pred_seq)
            
            # Get true word
            true_word = dev[i]
            
            # Increment counts
            total_words += 1
            if pred_word == true_word:
                total_correct_words += 1
                
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    char_accuracy = (total_correct_chars / total_chars) * 100 if total_chars > 0 else 0
    word_accuracy = (total_correct_words / total_words) * 100 if total_words > 0 else 0
    
    return avg_loss, char_accuracy, word_accuracy

def evaluate(model, loader, criterion, device, out_vocab):
    model.eval()
    total_loss = 0
    total_correct_chars = 0
    total_chars = 0
    
    # Word-level tracking
    total_correct_words = 0
    total_words = 0
    
    with torch.no_grad():
        for batch in loader:
            src, trg, src_lens, _, lat, dev = batch
            src, trg = src.to(device), trg.to(device)
            
            output = model((src, src_lens), trg, teacher_forcing_ratio=0)
            
            # Calculate loss
            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            
            # Calculate character-level accuracy
            pred = output.argmax(dim=2)
            mask = (trg[:, 1:] != 0)  # Ignore padding
            correct_chars = ((pred[:, 1:] == trg[:, 1:]) & mask).sum().item()
            total_correct_chars += correct_chars
            total_chars += mask.sum().item()
            
            # Calculate word-level accuracy
            batch_size = trg.size(0)
            for i in range(batch_size):
                # Get predicted sequence without padding, sos, eos
                pred_seq = [idx.item() for idx in pred[i, 1:] if idx.item() not in [0, 1, 2]]
                # Convert to string
                pred_word = out_vocab.decode(pred_seq)
                
                # Get true word
                true_word = dev[i]
                
                # Increment counts
                total_words += 1
                if pred_word == true_word:
                    total_correct_words += 1
                    
            total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    char_accuracy = (total_correct_chars / total_chars) * 100 if total_chars > 0 else 0
    word_accuracy = (total_correct_words / total_words) * 100 if total_words > 0 else 0
    
    return avg_loss, char_accuracy, word_accuracy

# =======================
# Attention Visualization with Custom Font
# =======================
def show_attention_grid(samples, font_path):
    # Load custom Devanagari font
    try:
        if font_path and os.path.exists(font_path):
            font_prop = FontProperties(fname=font_path, size=12)
            print(f"Loaded Devanagari font from {font_path}")
        else:
            font_prop = FontProperties(size=12)
            print("Using default font; Devanagari may not render.")
    except Exception as e:
        print(f"Font error: {e}")
        font_prop = FontProperties(size=12)

    # Create a 3x4 grid for 10 samples (adjust as needed)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for idx, (input_seq, true_output, predicted_output, attentions) in enumerate(samples[:12]):
        ax = axes[idx]
        input_chars = list(input_seq)
        output_chars = list(predicted_output)
        
        # Handle cases where attention might be empty
        if len(attentions) == 0:
            print(f"Skipping sample {idx} (no attention weights)")
            continue
        
        attn_matrix = np.array(attentions).squeeze(1)
        if attn_matrix.ndim == 1:
            attn_matrix = attn_matrix.reshape(1, -1)
        
        # Ensure the matrix matches the lengths
        max_input_len = len(input_chars)
        max_output_len = len(output_chars)
        attn_matrix = attn_matrix[:max_output_len, :max_input_len]
        
        sns.heatmap(
            attn_matrix,
            xticklabels=input_chars,
            yticklabels=output_chars,
            cmap='viridis',
            ax=ax,
            cbar=False
        )
        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_prop, rotation=0)
        ax.set_title(f"In: {input_seq}\nGT: {true_output}\nPred: {predicted_output}", 
                     fontsize=8, fontproperties=font_prop)
        ax.set_xlabel('Input (Latin)', fontsize=8)
        ax.set_ylabel('Output (Devanagari)', fontproperties=font_prop, fontsize=8)

    # Hide unused subplots
    for j in range(len(samples), 12):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('attention_heatmap.png', dpi=300, bbox_inches='tight')
    wandb.log({"attention_heatmap": wandb.Image('attention_heatmap.png')})
    plt.close()
    
    
# Function to decode with detailed attention debugging
def decode_with_attention_debug(input_seq, true_output, predicted_output, attentions):
    print("\n Correct prediction found. Showing decoder attention steps:")
    print(f"Input:         {input_seq}")
    print(f"Target Output: {true_output}")
    print(f"Prediction:    {predicted_output}\n")
    
    for i, attn in enumerate(attentions):
        if i >= len(predicted_output):
            break
        
        attn_input_char_idx = np.argmax(attn)
        if attn_input_char_idx < len(input_seq):
            focused_char = input_seq[attn_input_char_idx]
        else:
            focused_char = "?"
            
        print(f"Decoder step {i+1} (predicting '{predicted_output[i]}') is focused on input char: '{focused_char}' (position {attn_input_char_idx})")

# Enhanced visualizations

def create_connectivity_visualization(input_seq, output_seq, attention_weights, title=None, figsize=(10, 6)):
    """
    Create a detailed connectivity visualization showing the attention flow between input and output.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert attention weights to numpy array and flatten as needed
    attn_matrix = np.array([w.flatten() for w in attention_weights])
    
    # Make sure matrix dimensions match our character sequences
    if len(attn_matrix) > len(output_seq):
        attn_matrix = attn_matrix[:len(output_seq)]
    
    # Make sure width matches input length
    max_input_len = len(input_seq)
    if attn_matrix.shape[1] > max_input_len:
        attn_matrix = attn_matrix[:, :max_input_len]
    
    # Plot matrix
    im = ax.imshow(attn_matrix, cmap='Blues')
    
    # Setup axes
    ax.set_xticks(np.arange(len(input_seq)))
    ax.set_yticks(np.arange(len(output_seq)))
    
    # Label axes
    ax.set_xticklabels(list(input_seq))
    ax.set_yticklabels(list(output_seq))
    
    # Label tick marks to show direction of sequence
    plt.xlabel('Input Sequence →')
    plt.ylabel('Output Sequence →')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Attention Weight')
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title(f"Attention Mapping: '{input_seq}' → '{output_seq}'")
    
    # Add grid to make the alignment clearer
    ax.set_xticks(np.arange(-.5, len(input_seq), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(output_seq), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    return fig

def create_attention_flow_diagram(input_seq, output_seq, attention_weights, threshold=0.3, figsize=(12, 8)):
    """
    Create a diagram showing the flow of attention from input to output characters.
    Only connections with attention weight above threshold are shown.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert attention weights to numpy array and flatten
    attn_matrix = np.array([w.flatten() for w in attention_weights])
    
    # Adjust dimensions
    attn_matrix = attn_matrix[:len(output_seq), :len(input_seq)]
    
    # Heights for input and output characters
    in_height = 1.0
    out_height = 0.0
    
    # Plot input characters
    for i, char in enumerate(input_seq):
        ax.text(i, in_height, char, ha='center', va='center', 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Plot output characters
    for i, char in enumerate(output_seq):
        ax.text(i, out_height, char, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    # Draw connections between input and output based on attention
    cmap = plt.cm.Blues
    
    for i, out_char in enumerate(output_seq):
        if i >= len(attn_matrix):
            continue
            
        for j, in_char in enumerate(input_seq):
            if j >= len(attn_matrix[i]):
                continue
                
            weight = attn_matrix[i, j]
            if weight > threshold:
                # Draw line with alpha and width based on attention weight
                ax.plot([j, i], [in_height, out_height], 
                       alpha=min(1.0, weight + 0.3),
                       linewidth=weight*5,
                       color=cmap(weight))
    
    # Add legend for the connection strength
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=threshold, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Attention Weight')
    
    # Setting the axis limits and removing actual axis
    ax.set_xlim(-1, max(len(input_seq), len(output_seq)))
    ax.set_ylim(out_height - 0.5, in_height + 0.5)
    ax.axis('off')
    
    plt.title(f"Attention Flow: '{input_seq}' → '{output_seq}'")
    plt.tight_layout()
    return fig

def visualize_character_focus(input_seq, output_seq, attention_weights, figsize=(12, 6)):
    """
    Create a visualization showing which input character is most attended to
    for each output character.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Process attention weights
    attn_matrix = np.array([w.flatten() for w in attention_weights])
    attn_matrix = attn_matrix[:len(output_seq), :len(input_seq)]
    
    # Find max attention index for each output character
    max_attention_indices = np.argmax(attn_matrix, axis=1)
    
    # Set up positions
    input_pos = {char: idx for idx, char in enumerate(input_seq)}
    input_y = 1
    output_y = 0
    
    # Add background shading
    ax.axhspan(input_y-0.4, input_y+0.4, color='lightblue', alpha=0.3)
    ax.axhspan(output_y-0.4, output_y+0.4, color='lightgreen', alpha=0.3)
    
    # Plot input characters
    for idx, char in enumerate(input_seq):
        ax.text(idx, input_y, char, ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Plot output characters with arrows to their focused input
    arrow_props = dict(arrowstyle='->', linewidth=1.5, color='red')
    
    for idx, char in enumerate(output_seq):
        if idx >= len(max_attention_indices):
            continue
            
        # Plot output character
        ax.text(idx, output_y, char, ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Draw arrow from output to most attended input
        in_idx = max_attention_indices[idx]
        if in_idx < len(input_seq):
            ax.annotate('', 
                      xy=(in_idx, input_y-0.3),  # head at input
                      xytext=(idx, output_y+0.3),  # tail at output
                      arrowprops=arrow_props)
    
    # Add labels
    ax.text(-0.5, input_y, 'Input:', fontsize=12, ha='right', va='center', fontweight='bold')
    ax.text(-0.5, output_y, 'Output:', fontsize=12, ha='right', va='center', fontweight='bold')
    
    # Set up axes
    ax.set_xlim(-1, max(len(input_seq), len(output_seq)))
    ax.set_ylim(-0.8, 1.8)
    ax.axis('off')
    
    plt.title(f"Character Attention Focus: '{input_seq}' → '{output_seq}'")
    plt.tight_layout()
    return fig

# Main execution function to test on specific examples
def test_and_visualize(model, inp_vocab, out_vocab, test_examples, device):
    """
    Run transliteration on specific test examples and create visualizations.
    
    Args:
        model: Trained Seq2Seq model
        inp_vocab: Input vocabulary
        out_vocab: Output vocabulary
        test_examples: List of input strings to test
        device: Device to run model on
    """
    model.eval()
    visualizations = []
    
    for input_text in test_examples:
        # Encode input
        input_ids = inp_vocab.encode(input_text)
        input_tensor = torch.tensor([input_ids]).to(device)
        input_lens = torch.tensor([len(input_ids)])
        
        # Get model prediction
        output_ids, attention_weights = model.predict(input_tensor, input_lens)
        output_text = out_vocab.decode(output_ids)
        
        print(f"\nInput: {input_text}")
        print(f"Output: {output_text}")
        
        # Run attention debugging
        decode_with_attention_debug(input_text, "Unknown", output_text, attention_weights)
        
        # Create visualizations
        fig1 = create_connectivity_visualization(input_text, output_text, attention_weights)
        fig2 = create_attention_flow_diagram(input_text, output_text, attention_weights)
        fig3 = visualize_character_focus(input_text, output_text, attention_weights)
        
        visualizations.append({
            'input': input_text,
            'output': output_text,
            'attention': attention_weights,
            'figures': [fig1, fig2, fig3]
        })
        
        # Display only the first visualization for example
        plt.figure(fig1.number)
        plt.show()
        
        # Save figures if needed
        fig1.savefig(f'connectivity_{input_text}.png')
        fig2.savefig(f'attention_flow_{input_text}.png')
        fig3.savefig(f'char_focus_{input_text}.png')
        
    return visualizations

# For testing with actual model:
def load_and_visualize(model_path, config, inp_vocab_path, out_vocab_path, test_examples):
    """
    Load a trained model and vocabularies, then visualize test examples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load vocabularies
    inp_vocab = torch.load(inp_vocab_path)
    out_vocab = torch.load(out_vocab_path)
    
    # Initialize model
    encoder = Encoder(inp_vocab.size, config['embedding_dim'], config['hidden_dim'], 
                     config['enc_layers'], config['cell_type'], config['dropout'])
    
    decoder = Decoder(out_vocab.size, config['embedding_dim'], config['hidden_dim'],
                     config['dec_layers'], config['cell_type'], config['dropout'],
                     config['attention_type'])
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Run visualization
    test_and_visualize(model, inp_vocab, out_vocab, test_examples, device)

# =======================
# Main Execution
# =======================
def main():
    wandb.init(config=best_config, project="dakshina-translit-test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use the specific devanagiri.ttf file
    font_path = '/kaggle/input/devnagirihindi/devanagari.ttf'
    
    # Check if the font file exists
    if os.path.exists(font_path):
        print(f"Using Devanagari font file: {font_path}")
    else:
        # Try alternative paths
        alternative_paths = [
            '/kaggle/input/devnagiridata/devanagiri.ttf',
            '/kaggle/working/devanagiri.ttf',
            # Add more potential paths if needed
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                font_path = alt_path
                print(f"Found Devanagari font file at alternative path: {font_path}")
                break
        
        if not os.path.exists(font_path):
            print(f"Warning: Font file not found at {font_path}")
            print("Searching for any TTF file in Kaggle input directories...")
            
            # Fallback: search for any TTF file
            found_font = False
            for root, dirs, files in os.walk('/kaggle/input/devnagirihindi/devanagari.ttf'):
                for file in files:
                    if file.endswith('.ttf'):
                        font_path = os.path.join(root, file)
                        print(f"Found alternative font file: {font_path}")
                        found_font = True
                        break
                if found_font:
                    break
            
            if not found_font:
                print("No TTF font file found in Kaggle input, will use system fonts")
                font_path = None
        
    # Initialize vocabularies
    inp_vocab = Vocab()
    out_vocab = Vocab()

    # Load datasets
    train_data = TransliterationDataset("/kaggle/input/devnagiridata/hi.translit.sampled.train.tsv", inp_vocab, out_vocab)
    dev_data = TransliterationDataset("/kaggle/input/devnagiridata/hi.translit.sampled.dev.tsv", inp_vocab, out_vocab)
    test_data = TransliterationDataset("/kaggle/input/devnagiridata/hi.translit.sampled.test.tsv", inp_vocab, out_vocab, is_test=True)

    print(f"Train dataset size: {len(train_data)}")
    print(f"Dev dataset size: {len(dev_data)}")
    print(f"Test dataset size: {len(test_data)}")

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=best_config['batch_size'], 
                            shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=best_config['batch_size'],
                          shuffle=False, collate_fn=collate_fn)
    # Use batch size of 1 for test to get accurate per-word results
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    encoder = Encoder(inp_vocab.size, best_config['embedding_dim'], 
                     best_config['hidden_dim'], best_config['enc_layers'], 
                     best_config['cell_type'], best_config['dropout'])
    
    decoder = Decoder(out_vocab.size, best_config['embedding_dim'],
                     best_config['hidden_dim'], best_config['dec_layers'],
                     best_config['cell_type'], best_config['dropout'],
                     best_config['attention_type'])
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Print model information
    print(f"Encoder vocabulary size: {inp_vocab.size}")
    print(f"Decoder vocabulary size: {out_vocab.size}")
    
    # Count model parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model has {count_parameters(model):,} trainable parameters")

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(best_config['epochs']):
        train_loss, train_char_acc, train_word_acc = train(model, train_loader, criterion, optimizer, device, out_vocab)
        val_loss, val_char_acc, val_word_acc = evaluate(model, dev_loader, criterion, device, out_vocab)
        
        print(f"\nEpoch {epoch+1}/{best_config['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Train Char Acc: {train_char_acc:.2f}% | Train Word Acc: {train_word_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Char Acc: {val_char_acc:.2f}% | Val Word Acc: {val_word_acc:.2f}%")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_char_acc": train_char_acc,
            "train_word_acc": train_word_acc,
            "val_loss": val_loss,
            "val_char_acc": val_char_acc,
            "val_word_acc": val_word_acc
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")

    # Test evaluation
    print("\nLoading best model for testing...")
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
        print("Loaded best model from best_model.pth")
    else:
        print("No saved model found. Using current model state.")
    
    model.eval()
    
    total_char_correct = 0
    total_chars = 0
    total_word_correct = 0
    total_words = 0
    predictions = []
    attention_samples = []
    
    print("\nEvaluating on full test dataset...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            src, lat, dev, src_lens = batch
            src = src.to(device)
            pred_ids, attention_weights = model.predict(src, src_lens)
            pred_str = out_vocab.decode(pred_ids)
            true_str = dev[0]
            
            # Word-level accuracy
            total_words += 1
            word_correct = pred_str == true_str
            if word_correct:
                total_word_correct += 1
            
            # Character-level accuracy
            min_len = min(len(pred_str), len(true_str))
            for i in range(min_len):
                total_chars += 1
                if pred_str[i] == true_str[i]:
                    total_char_correct += 1
            # Count remaining characters in longer string as errors
            total_chars += abs(len(pred_str) - len(true_str))
            
            predictions.append({
                'input': lat[0],
                'true': true_str,
                'pred': pred_str,
                'correct': word_correct
            })
            
            # Collect attention weights for first 10 samples for visualization
            if len(attention_samples) < 10:
                attention_samples.append((lat[0], true_str, pred_str, attention_weights))
            
            # Show progress for large test sets
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(test_loader)} test samples")

    # Calculate accuracies
    word_accuracy = 100 * total_word_correct / total_words if total_words > 0 else 0
    char_accuracy = 100 * total_char_correct / total_chars if total_chars > 0 else 0
    
    print(f"\nFull Test Results:")
    print(f"Test Word Accuracy: {word_accuracy:.2f}% ({total_word_correct}/{total_words})")
    print(f"Test Character Accuracy: {char_accuracy:.2f}% ({total_char_correct}/{total_chars})")
    
    wandb.log({
        "test_word_acc": word_accuracy,
        "test_char_acc": char_accuracy,
        "total_words_correct": total_word_correct,
        "total_words": total_words
    })

    # Visualize attention for 10 samples with custom font
    if font_path:
        print(f"Visualizing attention with custom font: {font_path}")
    else:
        print("Visualizing attention with system fonts")
    show_attention_grid(attention_samples, font_path)

    
    
    # Create and log a table of predictions
    table = wandb.Table(columns=["Input", "True", "Predicted", "Correct"])
    for p in predictions[:100]:  # Log first 100 predictions
        table.add_data(p['input'], p['true'], p['pred'], p['correct'])
    
    wandb.log({"predictions": table})
    
    # Save detailed results
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Test Word Accuracy: {word_accuracy:.2f}% ({total_word_correct}/{total_words})\n")
        f.write(f"Test Character Accuracy: {char_accuracy:.2f}% ({total_char_correct}/{total_chars})\n\n")
        f.write("Sample Predictions:\n\n")
        
        # Save first 20 and a random sample of 10 other predictions
        random_samples = random.sample(predictions[20:], min(10, max(0, len(predictions) - 20)))
        for p in predictions[:20] + random_samples:
            f.write(f"Input: {p['input']}\n")
            f.write(f"True: {p['true']}\n")
            f.write(f"Pred: {p['pred']}\n")
            f.write(f"Correct: {p['correct']}\n\n")

    # Print exactly 10 random samples
    print("\n10 Random Test Samples:")
    # Get 10 random indices for consistent set of samples
    sample_indices = random.sample(range(len(predictions)), min(10, len(predictions)))
    samples = [predictions[i] for i in sample_indices]
    
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}:")
        print(f"Input (Latin): {sample['input']}")
        print(f"True (Devanagari): {sample['true']}")
        print(f"Predicted: {sample['pred']}")
        print(f"Correct: {sample['correct']}\n")

if __name__ == "__main__":
    main()

