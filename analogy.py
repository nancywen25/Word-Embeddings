import numpy as np
import en_core_web_lg   # a spacy module
from wordsim import get_embedding

def get_word(we, nlp):
    """
    Given an word embedding, find the word with the most similar vector
    and return the word and the cosine similarity score
    Args:
        we:

    Returns:
        word: str
        score: the cosine similarity score
    """
    query = np.reshape(we, (1, 300))
    keys, best_rows, scores = nlp.vocab.vectors.most_similar(query, n=5)
    words = [nlp.vocab.strings[key] for key in keys[0]]
    scores = list(scores[0])

    return words, scores

def get_fourth_word(A, B, C, nlp):
    """
    Given three words A, B, C,
    Return word D such that it completes the analogy A:B::C:D
    e.g. king:man::queen:woman

    Args:
        A:
        B:
        C:

    Returns:
        D: word that completes the analogy
    """
    # B - A + C = D
    we_A = get_embedding(A, nlp)
    we_B = get_embedding(B, nlp)
    we_C = get_embedding(C, nlp)

    we_D = we_B - we_A + we_C
    words, scores = get_word(we_D, nlp)
    return words[0] # return the top word

def print_analogy(A, B, C, nlp):
    D = get_fourth_word(A, B, C, nlp)
    return "{}:{}::{}:{}".format(A, B, C, D)


def main():
    # load the GloVe model with 300 dimensions
    nlp = en_core_web_lg.load()
    with open("output/analogy.txt", 'w') as f:
        f.write(print_analogy("king", "man", "queen", nlp) + '\n')
        f.write(print_analogy("London", "England", "Paris", nlp) + '\n')
        f.write(print_analogy("Dog", "Puppy", "Cat", nlp) + '\n')
        f.write(print_analogy("Sister", "Brother", "Aunt", nlp) + '\n')
        f.write(print_analogy("Slow", "Slower", "Fast", nlp) + '\n')

if __name__ == "__main__":
    main()
