import logging
from bpe_tokeniser import BPE


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting BPE tokenization process...")

    # Example usage
    text = ("This is the Hugging Face Course. This chapter is about tokenization. This section shows several tokenizer algorithms. Hopefully, you will be able to understand how they are trained and generate tokens.")
    BPE_tokeniser = BPE(text)

    BPE_tokeniser.create_corpus()
    BPE_tokeniser.get_vocab()
    BPE_tokeniser.create_corpus()

    pairs = BPE_tokeniser.find_pairs()
    n_possible_merges = len(pairs)
    n_iter = min(n_possible_merges // 2, 10)
    logger.info(f"Number of possible merges: {n_possible_merges}")
    logger.info(f"Number of iterations: {n_iter + 2}")

    BPE_tokeniser.learn_bpe(n_iter)


if __name__ == "__main__":
    main()
