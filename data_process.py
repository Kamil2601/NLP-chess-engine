import pandas as pd
import re
import spacy
from spacy.symbols import ORTH

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


def remove_unknown_tokens(tokens, known_tokens):
    return [t for t in tokens if t in known_tokens]


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

def prepare_data_for_sentiment_analysis_training(df_moves: pd.DataFrame, vocab, comment_col="comment", sentiment_col="sentiment", doc_to_tokens = doc_to_lemma_tokens, min_len = 2, max_len = 100):
    #Filter moves, leave only with 0/1 sentiment
    df_comments = df_moves[df_moves[sentiment_col].isin([0,1])][[comment_col, sentiment_col]]
    df_comments = add_comments_with_removed_quality_marks(df_comments, comment_col)

    # remove chess notation and some punctuation
    df_comments[comment_col] = [remove_useless(com) for com in df_comments[comment_col]]

    # tokenize with Spacy
    nlp = spacy.load("en_core_web_sm", disable = ['parser','ner'])
    docs = [t for t in nlp.pipe(df_comments[comment_col], n_process=16, batch_size=128)]
    df_comments[comment_col] = [doc_to_tokens(doc, vocab) for doc in docs]

    df_comments = df_comments[df_comments[comment_col].map(len).between(min_len, max_len)]

    return df_comments