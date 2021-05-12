import os
import pandas as pd
import numpy as np
from numpy.random import default_rng

from recsys.utils import pickle_load, pickle_save, absolute_filename, create_path

rng = default_rng(123)

RECSYS15_PATH = "data/recsys15/"
RECSYS15_FILENAME = "yoochoose-clicks.dat"

ECOMM_PATH = "data/ecomm/"
ECOMM_FILENAME = "OnlineRetail.csv"

AOTM_PATH = "data/aotm/"
AOTM_FILENAME = "aotm_list_ids.txt"
AOTM_NUMPYFILENAME = "aotm_sessions.pkl"


def load_recsys15(filename=None):
    """
    Checks to see if the processed recsys15 session sequence file exists
        If True: loads and returns the session sequences
        If False: creates and returns the session sequences constructed from the original data file
    """
    original_filename = absolute_filename(RECSYS15_PATH, RECSYS15_FILENAME)
    if filename is None:
        processed_filename = original_filename.replace(".dat", "_sessions.pkl")
        if os.path.exists(processed_filename):
            return pickle_load(processed_filename)
    else:
        if os.path.exists(absolute_filename(filename)):
            return pickle_load(absolute_filename(filename))

    df = load_original_recsys15(original_filename)
    session_sequences = preprocess_recsys15(df)
    return session_sequences


def load_original_recsys15(pathname=RECSYS15_PATH):
    """
    Reads in the original RecSys15 Challenge "clicks" data file and returns as a Pandas DF
    """
    df = pd.read_csv(
        absolute_filename(pathname, RECSYS15_FILENAME),
        names=["sessionID", "timestamp", "itemID", "category"],
        date_parser=["timestamp"],
        dtype={"category": str, "itemID": str, "sessionID": str},
    )
    return df


def preprocess_recsys15(df, min_session_count=3):
    """
    Given the recsys15 data in pandas df format, clean and sample to only those 
    sessions that contain at least min_session_count number of interactions
    """
    session_counts = df.groupby(["sessionID"]).count()
    df = df[
        df["sessionID"].isin(
            session_counts[session_counts["itemID"] >= min_session_count].index
        )
    ].reset_index(drop=True)

    # TODO: track preprocessed version by appending the filename with min_session_count
    filename = absolute_filename(
        RECSYS15_PATH, RECSYS15_FILENAME.replace(".dat", f"_sessions.pkl")
    )
    sessions = construct_session_sequences(
        df, "sessionID", "itemID", save_filename=filename
    )
    return sessions


def load_ecomm(filename=None):
    """
    Checks to see if the processed Online Retail ecommerce session sequence file exists
        If True: loads and returns the session sequences
        If False: creates and returns the session sequences constructed from the original data file
    """
    original_filename = absolute_filename(ECOMM_PATH, ECOMM_FILENAME)
    if filename is None:
        processed_filename = original_filename.replace(".csv", "_sessions.pkl")
        if os.path.exists(processed_filename):
            return pickle_load(processed_filename)
    else:
        if os.path.exists(absolute_filename(filename)):
            return pickle_load(absolute_filename(filename))

    df = load_original_ecomm(original_filename)
    session_sequences = preprocess_ecomm(df)
    return session_sequences


def load_original_ecomm(pathname=ECOMM_PATH):
    df = pd.read_csv(
        absolute_filename(pathname, ECOMM_FILENAME),
        encoding="ISO-8859-1",
        parse_dates=["InvoiceDate"],
    )
    return df


def preprocess_ecomm(df, min_session_count=3):
    df.dropna(inplace=True)
    item_counts = df.groupby(["CustomerID"]).count()["StockCode"]
    df = df[
        df["CustomerID"].isin(item_counts[item_counts >= min_session_count].index)
    ].reset_index(drop=True)

    # TODO: track preprocessed version by appending the filename with min_session_count
    filename = absolute_filename(
        ECOMM_PATH, ECOMM_FILENAME.replace(".csv", "_sessions.pkl")
    )
    sessions = construct_session_sequences(
        df, "CustomerID", "StockCode", save_filename=filename
    )
    return sessions


def load_aotm(filename=None):
    """
    Checks to see if the processed aotm session sequence file exists
        If True: loads and returns the session sequences
        If False: creates and returns the session sequences constructed from the original data file
    """
    processed_filename = absolute_filename(AOTM_PATH, AOTM_NUMPYFILENAME)

    if os.path.exists(processed_filename):
        return pickle_load(processed_filename)

    original_filename = absolute_filename(AOTM_PATH, AOTM_FILENAME)
    df = load_original_aotm(original_filename)
    session_sequences = preprocess_aotm(df, save_path=AOTM_PATH)
    # session_sequences = construct_session_sequences(df, save_path=processed_filename)
    return session_sequences


def load_original_aotm(pathname=AOTM_PATH):
    """
    Reads in the original AOTM file with all 29,164 playlists in numerical format. 
    Each line defines a playlist in the form #num# artnum: songnum artnum: songnum ... where num is the playlist index 
    Returns a Pandas DF
    """
    df = pd.read_csv(
        absolute_filename(pathname, AOTM_FILENAME),
        delimiter="# ",
        header=None,
        names=["list_id", "artists_tracks"],
        dtype={"category": int, "artists_tracks": str},
        engine="python",
    )
    # some sessions have missing entries...
    df.dropna(inplace=True)
    return df


def preprocess_aotm(df, min_session_count=3, save_path=None):
    """
    Given the aotm data in pandas df format, clean and sample to only those 
    sessions that contain at least min_session_count number of interactions
    """

    # separate out the artists and tracks within the sessions and use only the tracks for word2vec modeling
    artists_tracks = df["artists_tracks"].tolist()
    artists_tracks_tokens = [a.split() for a in artists_tracks]
    track_tokens = [
        [x for x in token if not x.endswith(":")] for token in artists_tracks_tokens
    ]

    # exclude tracks with only one entry, remove if there are tokens with len < min_session_count
    track_tokens = [token for token in track_tokens if len(token) >= min_session_count]

    if save_path:
        create_path(save_path)
        pickle_save(
            track_tokens, filename=absolute_filename(save_path, AOTM_NUMPYFILENAME)
        )
    return track_tokens


def construct_session_sequences(df, sessionID, itemID, save_filename):
    """
    Given a dataset in pandas df format, construct a list of lists where each sublist
    represents the interactions relevant to a specific session, for each sessionID. 
    These sublists are composed of a series of itemIDs (str) and are the core training 
    data used in the Word2Vec algorithm. 

    This is performed by first grouping over the SessionID column, then casting to list
    each group's series of values in the ItemID column. 

    INPUTS
    ------------
    df:                 pandas dataframe
    sessionID: str      column name in the df that represents invididual sessions
    itemID: str         column name in the df that represents the items within a session
    save_filename: str  output filename 
  
    Example:
    Given a df that looks like 

    SessionID |   ItemID 
    ----------------------
        1     |     111
        1     |     123
        1     |     345
        2     |     045 
        2     |     334
        2     |     342
        2     |     8970
        2     |     345
    
    Retrun a list of lists like this: 

    sessions = [
            ['111', '123', '345'],
            ['045', '334', '342', '8970', '345'],
        ]
    """
    grp_by_session = df.groupby([sessionID])

    session_sequences = []
    for name, group in grp_by_session:
        session_sequences.append(list(group[itemID].values))

    filename = absolute_filename(save_filename)
    create_path(filename)
    pickle_save(session_sequences, filename=save_filename)
    return session_sequences


def train_test_split(session_sequences, test_size: int = 10000, rng=rng):
    """
    Next Event Prediction (NEP) does not necessarily follow the traditional train/test split. 

    Instead training is perform on the first n-1 items in a session sequence of n items. 
    The test set is constructed of (n-1, n) "query" pairs where the n-1 item is used to generate 
    recommendation predictions and it is checked whether the nth item is included in those recommendations. 

    Example:
        Given a session sequence ['045', '334', '342', '8970', '128']
        Training is done on ['045', '334', '342', '8970']
        Testing (and validation) is done on ['8970', '128']
    
    Test and Validation sets are constructed to be disjoint. 
    """
    #np.random.seed(123)
    #rng = np.random.default_rng(123)

    ### Construct training set
    # use (1 st, ..., n-1 th) items from each session sequence to form the train set (drop last item)
    train = [sess[:-1] for sess in session_sequences]

    if test_size > len(train):
        print(
            f"Test set cannot be larger than train set. Train set contains {len(train)} sessions."
        )
        return

    ### Construct test and validation sets
    # sub-sample 10k sessions, and use (n-1 th, n th) pairs of items from session_squences to form the
    # disjoint validaton and test sets
    test_validation = [sess[-2:] for sess in session_sequences]
    # TODO: set numpy random seed! NM: added it at the top
    index = rng.choice(range(len(test_validation)), test_size * 2, replace=False)
    test = np.array(test_validation)[index[:test_size]].tolist()
    validation = np.array(test_validation)[index[test_size:]].tolist()

    return train, test, validation


#"""

if __name__ == "__main__":
    # load data
    sessions = load_ecomm()

    #df = load_original_ecomm()
    #sessions = preprocess_ecomm(df)
    #print(sessions[0])

    print(len(sessions))
    #train, test, valid = train_test_split(sessions)

    train, test, valid = train_test_split(sessions, test_size=1000)
    #print(train[0])
    print("validation set:", valid[0])
    print("test set", test[0])
#"""

