# -- coding: utf-8 --
"""
Includes various utility functions.
"""

import numpy as np
 
def slidingWindow(sequence,winSize,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable.
    Thanks to https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/"""
 
    # Verify the inputs
    try: 
        it = iter(sequence)
    except TypeError:
        raise Exception("ERROR sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("ERROR type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("ERROR step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("ERROR winSize must not be larger than sequence length.")
 
    # Pre-compute number of chunks to emit
    numOfChunks = int((len(sequence)-winSize)/step)+1
 
    # Do the work
    for i in range(0,numOfChunks*step,step):
        yield i, sequence[i:i+winSize]

READ_LIMIT = 400
db_readings = np.zeros(READ_LIMIT)

read_counter = 0
aggDB = 0

def reset_vars():
    """
    Resets the variables used in normalization. Since they are global 
    variables, we need to make sure that they are reset.
    """

    global read_counter
    global aggDB

    read_counter = 0
    aggDB = 0

def normalize(db):
    """
    Normalize the audio decibel data.
    """

    global read_counter
    global aggDB

    if read_counter >= READ_LIMIT:
        read_counter = 0

    aggDB += db - db_readings[read_counter]

    db_readings[read_counter] = db
    normalized_db = aggDB/READ_LIMIT

    read_counter += 1
    return normalized_db