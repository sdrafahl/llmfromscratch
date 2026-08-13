# Embedding Matrix Shapes & Concepts

When the dataloader passes a batch of words through the Embedding layer, it outputs a **3-Dimensional Tensor** (a cube of numbers). 

The exact shape of this 3D matrix is:
**`[batch_size, context_length, embedding_dimension]`**

Using our current configurations, the matrix shape is **`[4, 256, 256]`**. 

Here is exactly what each of those 3 dimensions represents when pointing to a specific piece of data:

1. **Dimension 1 (Batch Size = 4):** 
   This tells you *which specific sentence/sequence* you are looking at in the current batch. You have 4 of them, indexed `0` through `3`. 

2. **Dimension 2 (Context Length = 256):** 
   This tells you *which specific word's location* you are looking at within that sentence. You have up to 256 words in the sequence, indexed `0` through `255`. 

3. **Dimension 3 (Embedding Dimension = 256):**
   This holds the actual data! It is the list of 256 floating-point numbers that contain the "meaning" of the word you pointed to.

### PyTorch Example
If you want to find the mathematical meaning of the very first word, in the very first sentence of your batch, you point to it like this:

`my_matrix[0, 0, :]`
*(Translation: Batch 0, Word Position 0, Give me all the embedding numbers)*

If you want to see the embedding vector for the 50th word in the 3rd sentence of the batch:

`my_matrix[2, 49, :]`
*(Translation: Batch 2, Word Position 49, Give me all the embedding numbers)*

The matrix contains a dedicated "slot" for every single position in the context length, for every single sentence in the batch!

---

### The 3 "Sizes" to Remember
1. **Max Context Size (DataLoader):** The maximum number of tokens the AI can look at at one time (the context window). Dictates the size of the Positional Embedding Layer. (Currently: 256)
2. **Max Vocabulary Size (Tokenizer):** The total number of unique words/sub-words in the dictionary. Dictates the size of the Token Embedding Layer. (Currently: 100,277)
3. **Embedding Dimension Size:** How many numbers make up a single vector. The "brain capacity" of a single word. (Currently: 256)
