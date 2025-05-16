import logging
from bpe_tokeniser import BPE
import pandas as pd


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Starting BPE tokenization process...")

    texts = (
        pd.read_csv("src/datasets/train_dataset.csv", sep=",", header=None)
        .values.flatten()
        .tolist()
    )

    BPE_tokeniser = BPE(texts)

    BPE_tokeniser.create_corpus()
    BPE_tokeniser.get_vocab()
    BPE_tokeniser.create_corpus()

    pairs = BPE_tokeniser.find_pairs()
    n_possible_merges = len(pairs)
    n_iter = min(n_possible_merges, 100)
    logger.info(f"Number of possible merges: {n_possible_merges}")
    logger.info(f"Number of iterations: {n_iter + 2}")

    BPE_tokeniser.train(n_iter)

if __name__ == "__main__":
    main()
