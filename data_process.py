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
    # move_regex = "([Oo0](-[Oo0]){1,2}|[KQRBN]?[a-h]?[1-8]?[x-]?[a-h][1-8](\=[QRBN])?[+#]?[!?]*(\s(1-0|0-1|1\/2-1\/2))?)"
    # numbered_move_regex = "\d+\.{1,3}\s*(__move__\s*){0,1}(__move__)"
    

    com = comment

    com = re.sub("\n", " ", com)
    # com = re.sub(move_regex, move_token, com)
    # com = re.sub("__move__-__move__", move_token, com)
    # com = re.sub(numbered_move_regex, "__num-move__", com)
    # com = re.sub("(__num-move__\s*)*(__num-move__)", f" {notation_token} ", com)
    # com = re.sub("(__move__\s*)*(__move__)", f" {notation_token} ", com)
    # com = re.sub("\W*({notation_token}\W*)+", f" {notation_token} ", com)
    # print(re.findall(full_regex, com))

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


def token_filter(token):
    return not re.search(punctuation_regex, token.lemma_) or re.match(notation_token, token.lemma_)

def get_lemma(token):
    return token.lemma_.lower()


def process_comment(comment: str, word_process = get_lemma, word_filter = token_filter):
    com = replace_notation(comment)

    com = re.sub(r'[^\w\s!!]', ' ', com)
    # com = re.sub("\d+", " ", com)
    # com = re.sub("[ ]+", " ", com)

    # com = com.lower()

    # print(com)

    doc = nlp(com)

    # for token in doc:
    #     print(token, "|", token.lemma_, "|", token.pos_, "|", token.is_oov)


    # print([token.lemma_ for token in doc])

    res = [word_process(token) for token in doc if word_filter(token)]
    # res = com
    return res


def preprocess_comment(comment: str):
    com = replace_notation(comment)
    com = re.sub(r'[^\w\s!!]', ' ', com)
    return com

def proces_tokenized_comment(comment):
    return [word_process(token) for token in doc if word_filter(token)]

def process_comments(comments):
    coms = [preprocess_comment(com) for com in comments]

    spacy_comments = [proces_tokenized_comment(t) for t in nlp.pipe()]