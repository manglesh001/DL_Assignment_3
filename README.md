# DL_Assignment_3
Here is a complete `README.md` file for your **Sequence-to-Sequence Transliteration** project using TensorFlow, with beam search, attention, and WandB support:

---

# Sequence-to-Sequence Transliteration (Fundamentals of DL – Assignment 3)

This repository contains an implementation of a sequence-to-sequence (Seq2Seq) transliteration model using TensorFlow and Keras, with support for attention mechanisms and beam search. The model is configurable via `argparse` and is integrated with [Weights & Biases (WandB)](https://wandb.ai/) for hyperparameter tuning.

---



##  Model Overview

The transliteration model is built using a custom `MyRNN` class implementing an encoder-decoder architecture with:

* Configurable RNN cell types (`LSTM`, `GRU`, `RNN`)
* Attention mechanisms (`dot`, `general`, `concat`)
* Beam search decoding
* Multi-layer encoder and decoder support

---



## WandB Configuration for Hyperparameter Sweeps

To use WandB for tuning, define your sweep configuration as follows:

```python
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'embedding_dim': {'values': [32, 64, 128, 256]},
        'hidden_dim': {'values': [32, 64, 128, 256]},
        'enc_layers': {'values': [1, 2, 3]},
        'dec_layers': {'values': [1, 2, 3]},
        'cell_type': {'values': ['GRU', 'LSTM', 'RNN']},
        'dropout': {'values': [0.2, 0.3, 0.5]},
        'epochs': {'values': [10, 15]},
        'beam_size': {'values': [1, 3, 5]},
        'attention_type': {'values': ['dot', 'general', 'concat']},
        'batch_size': {'values': [64, 128, 256]},
        'learning_rate': {'values': [0.001, 0.0005, 0.0001]}
    }
}
```

Then run the sweep using:

```bash
wandb sweep sweep_config.yaml
wandb agent <your-entity>/<project-name>/<sweep-id>
```

---

## 🔧 CLI Arguments

You can run the training with the following arguments:

```python
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
```

---



## 📁 Dataset

Use the [Dakshina dataset](https://huggingface.co/datasets/dakshina) for training. You may also place your `hi.translit.sampled.train.tsv` and `hi.translit.sampled.dev.tsv` files in a `data/` directory and point to them via `--train_path` and `--dev_path`.

## Wandb Report Link:   https://api.wandb.ai/links/manglesh_dl_ass3/wqvjcsxd

Question 2 &4: https://github.com/manglesh001/DL_Assignment_3/blob/main/dl-ass-3-without-attention.ipynb
or 
Question 5  :   https://github.com/manglesh001/DL_Assignment_3/blob/main/dl-ass3-with-attention.ipynb
Question 6:  https://github.com/manglesh001/DL_Assignment_3/blob/main/test_with_attention.py  

