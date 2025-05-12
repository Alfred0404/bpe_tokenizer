from collections import defaultdict
import logging


class BPE:
    """
    A class to represent the Byte Pair Encoding (BPE) algorithm.
    """

    def __init__(self, text: str):
        """
        Initializes the BPE object with a given corpus.
        Args:
            corpus (dict): The input corpus for BPE.
        """
        self.text = text
        self.corpus = {}  # format: {('w','o','r','d', '</w>): 2}
        self.vocab = []  # vocabulaire du corpus

    def create_corpus(self):
        """
        Tokenizes the input text using Byte Pair Encoding (BPE).
        Args:
            text (str): The input text to tokenize.
        """

        # delete all special characters and convert to lower case
        chars_to_delete = [
            "!",
            "@",
            "#",
            "$",
            "%",
            "^",
            "&",
            "*",
            "(",
            ")",
            "-",
            "_",
            "=",
            "+",
            "{",
            "}",
            "[",
            "]",
            "|",
            ":",
            ";",
            '"',
            "'",
            "<",
            ">",
            "?",
            ",",
            ".",
            "/",
            "\\",
            "`",
            "~",
        ]

        for char in chars_to_delete:
            self.text = self.text.replace(char, " ")

        self.text = self.text.lower()

        for word in self.text.split():
            token = tuple(list(word) + ["</w>"])
            if token not in self.corpus:
                self.corpus[token] = 1
            else:
                self.corpus[token] += 1

        logging.info(f"Corpus: {self.corpus}")

    def get_vocab(self):
        """
        Returns the vocabulary of the BPE object.
        """
        vocab = set()
        for word in self.corpus:
            vocab.update(word)
        self.vocab = list(vocab)
        logging.info(f"Vocabulary: {self.vocab}")

    def find_pairs(self):
        """
        Finds all the pairs of adjacent symbols in the vocabulary.
        Returns:
            dict: A dictionary with pairs as keys and their frequencies as values.
        """
        pairs = defaultdict(int)
        for word, freq in self.corpus.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
            logging.info(f"Pairs: {pairs}")
        return pairs

    def update_vocab(self):
        """
        Updates the vocabulary of the BPE object.
        """
        pairs_freq = self.find_pairs()

        if not pairs_freq:
            return

        most_freq_pair = max(pairs_freq, key=pairs_freq.get)
        logging.info(f"Most frequent pair: {most_freq_pair}")

        new_corpus = {}
        for word, freq in self.corpus.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == most_freq_pair:
                    new_word.append(most_freq_pair[0] + most_freq_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            if new_word not in new_corpus:
                new_corpus[new_word] = freq
            else:
                new_corpus[new_word] += freq

        self.corpus = new_corpus
        self.get_vocab()

        logging.info(f"Updated corpus: {self.corpus}")
        logging.info(f"Updated vocab: {self.vocab}")

    def learn_bpe(self, iterations: int):
        """
        Learns the BPE codes from the corpus.
        """
        # Implementation of BPE learning algorithm goes here

        self.create_corpus()
        self.get_vocab()

        for i in range(iterations):
            self.update_vocab()
            logging.info(f"Iteration {i + 1}/{iterations} completed.")
