# Word-Embeddings
Explore the benefits of word embeddings for NLP tasks

### Task 1: Word Similarity
This task measures how well word embeddings can reproduce human judgements regarding the similarity of words. Using 500 dimension GloVe vectors, we calculated cosine similarity between word embeddings. The embeddings similarity score and the human similarity score had a Pearson correlation of 0.71. See output [here](https://github.com/nancywen25/Word-Embeddings/blob/master/output/word_similarity.txt).

### Task 2: Analogy
This task demonstrates that word embeddings can be used to find the fourth word in an analogy e.g. `king:man::queen:woman`. If each word A, B, C, D in the relationship `A:B::C:D` are represented as word embeddings, D can be calculated from `D = B - A + C`. See output [here](https://github.com/nancywen25/Word-Embeddings/blob/master/output/analogy.txt) for other analogies that can be found with word embeddings.
