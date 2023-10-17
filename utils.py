#!/user/bin/python3
import logging

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler()],
)


def toDf(item_path, dropCols=None):
    logging.info(
        "Starting function toDf with item_path: %s, dropCols: %s", item_path, dropCols
    )
    pf_list = []
    counter = 0
    for chunk in chunks(item_path):
        if dropCols is not None:
            chunk = chunk.drop(labels=dropCols, axis="columns")
        pf_list.append(chunk)
        counter += len(chunk.index)
        logging.info("read %d of %s" % (counter, item_path))
    df = pd.concat(pf_list)
    logging.info("Ending function toDf with DataFrame of size: %d", df.shape[0])
    return df


def chunks(item_path):
    logging.info("Starting function chunks with item_path: %s", item_path)
    chunkSize = 1000000
    chunks = pd.read_csv(item_path, header=0, chunksize=chunkSize)
    logging.info("Ending function chunks")
    return chunks
