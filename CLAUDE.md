Philosopher Quote Matcher

Type up to 1,000 characters about anything, and get back one quote and the philosopher who said it. The goal is to match the vibe of what you wrote, not just its topic:

Questions get an answer.
Requests for advice get advice.
Venting gets something that resonates.
Nonsense gets an absurd quote.
The problem

Off-the-shelf sentence embeddings match by topic. Ask "I'm nervous about my exam" and a pretrained model returns quotes about knowledge or testing, when the right reply is about courage or calm. This project fine-tunes an embedding model so that “close” means “a good reply,” not “about the same thing.”

How it works

Retrieval. Input text and every quote are embedded into the same 384-dimensional space with sentence-transformers/all-MiniLM-L6-v2. Vectors are mean-pooled over real tokens (using the attention mask) and normalised to unit length. Search is then a single matrix multiply, which gives cosine similarity against every quote at once, followed by a top-k lookup.

Synthetic training data. There was no dataset of (message → ideal quote) pairs, so one was generated. Claude Haiku was given real quotes and asked to work backwards: write realistic user messages that a given quote would be a perfect reply to, separately for each of the four modes. It also had to judge whether each quote actually fit the mode and reject poor fits. The outputs were filtered for length, word overlap with the quote, and duplicates. The result was 38,600 pairs from 4,000 quotes, split 90/10 into train and validation by quote, so no quote appears in both sets. The whole generation run cost under $2.

Contrastive fine-tuning. The model is trained with an InfoNCE loss. For each batch of 64 pairs, it computes a 64 × 64 similarity matrix between inputs and quotes, and cross-entropy pushes each input toward its own quote and away from the other 63. Batches are built so that no quote appears twice, which would otherwise create false negatives. Both inputs and quotes are encoded by the model during training, so it can separate quotes that share a topic but differ in tone. Training uses AdamW with linear warmup and decay.

Built by hand

The core ML pipeline is written from scratch for learning purposes: tokenization handling, attention-masked mean pooling, the contrastive loss, batching and the training loop. The hand-written pooling was checked against the sentence-transformers library output (cosine > 0.9999), and the training loop is benchmarked against the library’s built-in trainer.
