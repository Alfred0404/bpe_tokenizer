import logging
from bpe_tokeniser import BPE


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Starting BPE tokenization process...")

    # Example usage
    text = "Artificial intelligence is transforming industries across the world. Organizations are increasingly adopting intelligent systems to automate tasks, analyze data, and improve decision-making. With the rapid development of machine learning algorithms, companies can create powerful predictive models and deliver personalized experiences to their users. The integration of AI into everyday tools is no longer optional — it is becoming essential for innovation and efficiency."

    BPE_tokeniser = BPE(text)

    BPE_tokeniser.create_corpus()
    BPE_tokeniser.get_vocab()
    BPE_tokeniser.create_corpus()

    pairs = BPE_tokeniser.find_pairs()
    n_possible_merges = len(pairs)
    # n_iter = min(n_possible_merges // 2, 1000)
    n_iter = 120
    logger.info(f"Number of possible merges: {n_possible_merges}")
    logger.info(f"Number of iterations: {n_iter + 2}")

    BPE_tokeniser.train(n_iter)

    test_text = "Intelligent automation enhances productivity and accelerates innovation in modern organizations."

    logging.info(BPE_tokeniser.tokenize(test_text))


if __name__ == "__main__":
    main()
