# Philo

Type up to 1,000 characters about anything and get back one quote from a philosopher. It tries to match the vibe of what you wrote, not just its topic: questions get an answer, requests for advice get advice, venting gets something that resonates, and nonsense gets an absurd quote.

![Philo UI: "I'm nervous about my exam tomorrow." answered with Seneca's "The mind that is anxious about the future is miserable."](docs/screenshot.png)

## How it works

Off-the-shelf sentence embeddings match by topic, so "I'm nervous about my exam" pulls up quotes about knowledge or testing. Philo fine-tunes `sentence-transformers/all-MiniLM-L6-v2` so that "close" means "a good reply" instead.

- **Training data.** Claude Haiku worked backwards from 4,000 real quotes and wrote messages each quote would be a good reply to, across the four modes. After filtering, that gave 38,600 pairs, split 90/10 by quote.
- **Training.** Contrastive (InfoNCE) loss over batches of 64 pairs with no repeated quotes, using AdamW with linear warmup and decay.
- **Retrieval.** Inputs and quotes are mean-pooled and normalised into the same 384-dimensional space. A match is one matrix multiply and a top-k lookup.

The pooling, loss, batching and training loop are written by hand as a learning exercise. See [CLAUDE.md](CLAUDE.md) for the full write-up.

## Running it

```bash
uv sync
python train.py        # writes the fine-tuned model to models/v1
python server.py       # http://127.0.0.1:8765
```

The quote index is built on first launch and rebuilt automatically if the model or quotes change. For a terminal version, run `python main.py`.
