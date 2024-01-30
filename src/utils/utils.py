import chess
import chess.pgn
import pandas as pd
import io
import numpy as np
import sqlite3 as db
import matplotlib.pyplot as plt
import re
from datasets import DatasetDict, Dataset

def load_sql_to_df(sql, db_filename):
    con = db.connect(db_filename)
    data_frame = pd.read_sql_query(sql, con)
    con.close()

    return data_frame

def save_to_sql(df, db_filename, table_name, if_exists='fail', index=False):
    con = db.connect(db_filename)
    df.to_sql(table_name, con, if_exists=if_exists, index=index)
    con.close()
    return


def plot_history(history: dict):
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.legend(['Train loss', 'Validation loss'], fontsize=10)
    ax.set_xlabel('Epochs', size=15)
    ax = fig.add_subplot(1, 2, 2)
    plt.plot(history['train_accuracy'])
    plt.plot(history['val_accuracy'])
    plt.legend(['Train acc.', 'Validation acc.'], fontsize=10)
    ax.set_xlabel('Epochs', size=15)
    plt.show()



def safe_strip(text):
    try:
        return text.strip()
    except:
        return text
    

white_advantage_regex = "(14)|(16)|(18)|(20)"
black_advantage_regex = "(15)|(17)|(19)|(21)"
    
def get_sentiment(row):
    if row.sentiment == -1 and " w " in row.fen and re.search(white_advantage_regex, row.nags):
        return 1
    elif row.sentiment == -1 and " b " in row.fen and re.search(black_advantage_regex, row.nags):
        return 1
    else:
        return row.sentiment
    

def split_moves(table_name, db_path):
    all_moves = load_sql_to_df(f"SELECT * FROM {table_name}", {db_path})
    all_moves.drop("index", axis=1, inplace=True)
    all_moves.comment = all_moves.comment.apply(safe_strip)
    all_moves.nags = all_moves.nags.apply(repr)
    all_moves.sentiment = all_moves.apply(get_sentiment, axis=1)

    
    moves_with_comments = all_moves[all_moves.comment != ""]
    save_to_sql(moves_with_comments, db_path, "moves_with_comments", if_exists="replace")

    labeled_moves_with_comments = moves_with_comments[moves_with_comments.sentiment != -1]
    save_to_sql(labeled_moves_with_comments, db_path, "labeled_moves_with_comments", if_exists="replace")

    unlabeled_moves_with_comments = moves_with_comments[moves_with_comments.sentiment == -1]
    save_to_sql(unlabeled_moves_with_comments, db_path, "unlabeled_moves_with_comments", if_exists="replace")

    nags_without_comments = all_moves[(all_moves.comment == "") & (all_moves.sentiment != -1)]
    save_to_sql(nags_without_comments, db_path, "nags_without_comments", if_exists='replace')
    bad_nags_without_comments = all_moves[(all_moves.comment == "") & (all_moves.sentiment == -1)]
    save_to_sql(bad_nags_without_comments, db_path, "bad_nags_without_comments", if_exists='replace')


def train_val_test_balanced_datasets(
    df, train_size_per_class=100000, val_test_size_per_class=2500
):
    # Assuming your DataFrame is named 'df'
    # Replace 'df' with the actual DataFrame name

    # Filter rows with positive sentiment (1) and negative sentiment (0)
    positive_sentiment = df[df["sentiment"] == 1]
    negative_sentiment = df[df["sentiment"] == 0]

    if (
        len(positive_sentiment) < train_size_per_class + 2 * val_test_size_per_class
        or len(negative_sentiment) < train_size_per_class + 2 * val_test_size_per_class
    ):
        raise ValueError("Not enough data for the requested train and test sizes")

    # Shuffle the rows randomly
    positive_sentiment = positive_sentiment.sample(frac=1).reset_index(drop=True)
    negative_sentiment = negative_sentiment.sample(frac=1).reset_index(drop=True)

    # Define the sizes for train, validation, and test sets
    train_size = train_size_per_class
    val_test_size = val_test_size_per_class

    # Create train set with 100k positive and 100k negative
    train_positive = positive_sentiment[:train_size]
    train_negative = negative_sentiment[:train_size]
    train_data = pd.concat([train_positive, train_negative])
    train_data = train_data[["fen", "move", "comment", "sentiment", "color_comment"]]

    # Create validation set with 2500 positive and 2500 negative
    val_positive = positive_sentiment[train_size : train_size + val_test_size]
    val_negative = negative_sentiment[train_size : train_size + val_test_size]
    validation_data = pd.concat([val_positive, val_negative])
    validation_data = validation_data[
        ["fen", "move", "comment", "sentiment", "color_comment"]
    ]

    # Create test set with 2500 positive and 2500 negative
    test_positive = positive_sentiment[
        train_size + val_test_size : train_size + 2 * val_test_size
    ]
    test_negative = negative_sentiment[
        train_size + val_test_size : train_size + 2 * val_test_size
    ]
    test_data = pd.concat([test_positive, test_negative])
    test_data = test_data[["fen", "move", "comment", "sentiment", "color_comment"]]

    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_data, preserve_index=False)
    validation_dataset = Dataset.from_pandas(validation_data, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_data, preserve_index=False)

    data_dict = DatasetDict(
        {"train": train_dataset, "validation": validation_dataset, "test": test_dataset}
    )

    return data_dict


def add_color_to_comment(row):
    if " w " in row.fen:
        return "white [SEP] " + row.comment
    else:
        return "black [SEP] " + row.comment