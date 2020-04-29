import pandas as pd
import en_core_web_lg   # a spacy module
from scipy import spatial
from scipy.stats import pearsonr


def get_embedding(word, nlp):
    """
    Given a word, returns the word embedding
    Args:
        word:

    Returns:
        a numpy array of 300 floats
    """
    token = nlp(word)
    return token.vector

def read_human_judgements(fname):
    """
    Read in the tab-separated file containing a corpus of
    human judgements on the similarity between two words

    Each row contains: relationship word1 word2 score_human
    Returns:
        df: dataframe containing file content
    """

    df = pd.read_csv(fname,
                     sep='\t',
                     skiprows=11, # skip the rows that start with pound sign
                     header=None,
                     names=['relationship', 'word1', 'word2', 'score_human'])
    return df

def similarity_score(word1, word2, nlp):
    """
    Given two words, return the cosine similarity of the embeddings
    Args:
        word1:
        word2:

    Returns:
        score: cosine similarity
    """

    we1 = get_embedding(word1, nlp)
    we2 = get_embedding(word2, nlp)

    # calculate the cosine similarity btw we1 and we2
    return 1 - spatial.distance.cosine(we1, we2)

def generate_output(df, outfile):
    """
    Write output to file that contains the correlation between the human
    similarity scores and the embedding similarity scores

    Args:
        df:

    Returns:
        None
    """
    # find the pearson correlation
    corr, _ = pearsonr(df['score_human'], df['score_embeddings']) # second element is p-value

    # write the word pair, human score, and word embedding score
    df.to_csv(outfile,
              sep='\t',
              index=False,
              columns=['word1', 'word2', 'score_human', 'score_embeddings'])

    # also write the overall correlation to the file
    with open(outfile, 'a') as f:
        f.write("Pearson correlation btw human score and embedding score: {}".format(corr))

def main():
    # read in human judgement scores
    fname = "data/wordsim-353.txt"
    df = read_human_judgements(fname)

    # load the GloVe model with 300 dimensions
    nlp = en_core_web_lg.load()

    # determine similarity scores using cosine similarity of embeddings
    df['score_embeddings'] = df.apply(lambda row: similarity_score(row['word1'],
                                                                   row['word2'],
                                                                   nlp), axis=1)
    # write results to file
    outfile = "output/word_similarity.txt"
    generate_output(df, outfile)

if __name__ == "__main__":
    main()