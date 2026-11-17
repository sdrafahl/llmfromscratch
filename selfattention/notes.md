# Making a Context Vector (Self-Attention)

Goal: turn one token's embedding `x_i` into a **context vector** `z_i`, a blend of every
token in the sequence, weighted by how relevant each one is to `x_i`.

1. **Tokenize + embed.** Text → token IDs (BPE) → embedding vectors (+ positional embeddings).
   Attention works on the vectors `x`, never on the raw IDs.
2. **Create weight matrices** `W_q`, `W_k`, `W_v`, each `(d_in, d_out)`. They start random and
   get learned during training.
3. **Project** into query, key, value:
   - `q_i = x_i @ W_q`: just the ONE token we're building a context vector for
   - `K = x @ W_k`: a key for EVERY token
   - `V = x @ W_v`: a value for EVERY token
4. **Attention scores:** dot `q_i` with every key (including its own): `scores = q_i @ K.T`.
   Query vs. keys, never query vs. query.
5. **Attention weights:** scale, then softmax: `weights = softmax(scores / sqrt(d_k))`.
   The weights sum to 1. Scaling keeps softmax from getting too peaky (tiny gradients).
6. **Context vector:** weighted sum of the values: `z_i = weights @ V`
   (= `w_1*v_1 + w_2*v_2 + ... + w_n*v_n`).

All tokens at once: `Z = softmax(Q @ K.T / sqrt(d_k)) @ V`

### Key vs. Value
- **Key** = the catalog label. Only used to decide HOW MUCH a token gets attended to.
- **Value** = the book's contents. WHAT actually gets mixed into the context vector.
- Keys vanish after the scores are computed; only values end up in `z_i`.
