import pandas as pd
import re

def remove_notation(comment: str):
    move_regex = "([Oo0](-[Oo0]){1,2}|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](\=[QRBN])?[+#]?[!?]*(\s(1-0|0-1|1\/2-1\/2))?)"
    numbered_move_regex = "\d+\.{1,3}\s*(__move__\s*){0,1}(__move__)"

    com = re.sub(move_regex, "__move__", comment)
    com = re.sub(numbered_move_regex, "__num-move__", com)
    com = re.sub("(__num-move__\s*)*(__num-move__)", "__variant__", com)

    return com