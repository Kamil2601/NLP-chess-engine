import pandas as pd
import re
import spacy
from spacy.symbols import ORTH
from sklearn.model_selection import train_test_split

move_regex = re.compile("([Oo0](-[Oo0]){1,2}|[KQRBN]?[a-h]?[1-8]?[x-]?[a-h][1-8](\=[QRBN])?[+#]?[!?]*(\s(1-0|0-1|1\/2-1\/2))?)")
numbered_move_regex = re.compile("\d+\.{1,3}\s*(__move__\s*){0,1}(__move__)")
full_regex = re.compile("\d{1,}\.{1,3}\s?(([Oo0]-[Oo0](-[Oo0])?|[KQRBN]?[a-h]?[1-8]?[x-]?[a-h][1-8]?(\=[QRBN])?[+#]?[!?]*)(\s?\{.+?\})?(\s(1-0|0-1|1\/2-1\/2))?\s){1,2}")

punctuation_regex = r"[\"#\$%&\'\(\)\*\+,-\./:;<=>\@\[\\\]\^_`{\|}~ 0-9]+"

nlp = spacy.load("en_core_web_sm", disable = ['parser','ner'])


notation_token = "__notation__"
move_token = "__move__"

nlp.tokenizer.add_special_case(notation_token, [{ORTH: notation_token}])
nlp.tokenizer.add_special_case(move_token, [{ORTH: move_token}])


def replace_notation(comment: str):
    com = comment

    com = re.sub("\n", " ", com)
    com = re.sub(full_regex, f" {notation_token} ", com)
    com = re.sub(move_regex, notation_token, com)
    com = re.sub(f"\W*({notation_token}(\d|\W)*)+", f" {notation_token} ", com)

    com = re.sub("[+-]?\d+[.]\d+([/]\d*[.]?\d*)", "", com)
    # com = re.sub("\d+", "", com)
    return com

def remove_notation(comment: str):
    comment = replace_notation(comment)
    comment = re.sub(notation_token, " ", comment)

    return comment


def remove_useless(comment: str):
    comment = remove_notation(comment)
    comment = re.sub("[^a-zA-z!?'.]+", " ", comment)
    comment = re.sub("\s+", " ", comment)
    comment = comment.lower()
    return comment


def token_filter(token):
    return not re.search(punctuation_regex, token.lemma_) or re.match(notation_token, token.lemma_)

def get_lemma(token):
    return token.lemma_.lower()


def token_to_text_or_lemma(token, vocab):
    if token.text in vocab:
        return token.text
    elif token.lemma_ in vocab:
        return token.lemma_
    return None

def doc_to_text_or_lemma_tokens(doc, vocab):
    tokens = [token_to_text_or_lemma(token, vocab) for token in doc]
    return [token for token in tokens if token]

def doc_to_lemma_tokens(doc, vocab):
    return [token.lemma_ for token in doc if token.lemma_ in vocab]

def doc_to_tokens(doc, vocab):
    return [token.text for token in doc if token.text in vocab]


def add_comments_with_removed_quality_marks(df_comments: pd.DataFrame, comment_col="comment"):
    """
        For every comment with quality mark, add the same comment with removed quality mark, and the same sentiment
        e.g. "! good move" 1 -> "good move" 1
    """

    def remove_quality_mark(comment):
        return re.sub("^[!?]{1,2}[ ]","", comment)

    df_comments_copy = df_comments[df_comments[comment_col].str.match("^[!?]{1,2}[ ]")].copy()
    df_comments_copy[comment_col] = df_comments_copy[comment_col].map(remove_quality_mark)

    return pd.concat([df_comments, df_comments_copy], axis=0, ignore_index=True)


def spacy_tokenize(comments, vocab, doc_to_tokens):
    nlp = spacy.load("en_core_web_sm", disable = ['parser','ner'])
    docs = [t for t in nlp.pipe(comments, n_process=16, batch_size=128)]
    tokenized_comments = [doc_to_tokens(doc, vocab) for doc in docs]

    return tokenized_comments

def shuffle_and_sort(df: pd.DataFrame, comment_col):
    df = df.sample(frac=1)
    return df.sort_values(by=comment_col, key=lambda x: x.str.len())

def df_train_test_split(comments_df, comment_col, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    train_df, test_df = train_test_split(comments_df, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)

    train_df = train_df.sort_values(by=comment_col, key=lambda x: x.str.len())
    test_df = test_df.sort_values(by=comment_col, key=lambda x: x.str.len())

    return train_df, test_df


def prepare_data_for_sentiment_analysis_training(df_moves: pd.DataFrame, vocab, comment_col="comment", sentiment_col="sentiment", doc_to_tokens = doc_to_lemma_tokens, min_len = 2, max_len = 100):
    #Filter moves, leave only with 0/1 sentiment
    df_comments = df_moves[df_moves[sentiment_col].isin([0,1])][[comment_col, sentiment_col]]
    df_comments = add_comments_with_removed_quality_marks(df_comments, comment_col)

    # remove chess notation and some punctuation
    # df_comments[comment_col] = [remove_useless(com) for com in df_comments[comment_col]]
    df_comments[comment_col] = [com.lower() for com in df_comments[comment_col]]

    # tokenize with Spacy
    df_comments['preprocessed_comment'] = spacy_tokenize(df_comments[comment_col], vocab, doc_to_tokens)

    df_comments = df_comments[df_comments['preprocessed_comment'].map(len).between(min_len, max_len)]

    df_comments = shuffle_and_sort(df_comments, 'preprocessed_comment')

    return df_comments


def prepare_data_for_sentiment_analysis_prediction(df_moves: pd.DataFrame, vocab, comment_col="comment", prep_comment_col="preprocessed_comment", doc_to_tokens = doc_to_lemma_tokens, min_len = 2, max_len = 100):
    # remove chess notation and some punctuation
    df_moves[prep_comment_col] = [remove_useless(com) for com in df_moves[comment_col]]

    # tokenize with Spacy
    df_moves[prep_comment_col] = spacy_tokenize(df_moves[prep_comment_col], vocab, doc_to_tokens)

    df_moves = df_moves[df_moves[prep_comment_col].map(len).between(min_len, max_len)]

    df_moves = shuffle_and_sort(df_moves, prep_comment_col)

    return df_moves
