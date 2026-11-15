# Philo

Type up to 1,000 characters about anything and get back one quote from a philosopher. It tries to match the vibe of what you wrote, not just its topic: questions get an answer, requests for advice get advice, venting gets something that resonates, and nonsense gets an absurd quote.

![Philo UI: "I'm nervous about my exam tomorrow." answered with Seneca's "The mind that is anxious about the future is miserable."](docs/screenshot.png)

## How it works

Off-the-shelf sentence embeddings match by topic, so "I'm nervous about my exam" pulls up quotes about knowledge or testing. Philo fine-tunes `sentence-transformers/all-MiniLM-L6-v2` so that "close" means "a good reply" instead.

- **Training data.** Claude Haiku worked backwards from real quotes. For each quote it wrote messages the quote would be a good reply to, in each of the four modes, and it rejected quotes that didn't fit a mode. Two generation rounds used 7,721 different quotes and, after filtering, gave 90,378 pairs for under $5 in total. Each round is split 90/10 by quote, so no quote appears in both training and validation.
- **Training.** Contrastive (InfoNCE) loss over batches of 64 pairs with no repeated quotes, using AdamW with linear warmup and decay.
- **Retrieval.** Inputs and quotes are mean-pooled and normalised into the same 384-dimensional space. A match is one matrix multiply and a top-k lookup.
- **Intent routing.** Before retrieval, a frozen `Qwen/Qwen2.5-0.5B-Instruct` reads the message, and a small head trained on its last hidden state predicts the mode. When the head is at least 80% sure a message is nonsense, Philo picks at random from the 50 closest absurd quotes, so gibberish doesn't always get the same reply. Every other message goes through normal retrieval.

The pooling, loss, batching and training loop are written by hand as a learning exercise. See [CLAUDE.md](CLAUDE.md) for the full write-up.

## Running it

```bash
uv sync
python train.py                   # fine-tunes MiniLM, saving each epoch to models/v2-e1 ... v2-e4
python eval.py --model base --model models/v2-e2   # retrieval scores on held-out pairs, vs plain MiniLM
python build_intent_features.py   # runs Qwen over every training message (about an hour on a 4 GB GPU)
python train_intent_head.py       # trains the intent head in seconds -> models/intent_head.pt
python server.py --model models/v2-e2   # http://127.0.0.1:8765
```

Serve whichever epoch scores best in `eval.py`. The server needs `models/intent_head.pt`. Qwen runs on the CPU and adds about 0.1 s per request; `--intent-device cuda` moves it to the GPU.

The quote index is built on first launch and rebuilt automatically if the model or quotes change. For a terminal version without intent routing, run `python main.py --model models/v2-e2`.
