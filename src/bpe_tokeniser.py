from collections import defaultdict
import logging


class BPE:
    """
    A class to represent the Byte Pair Encoding (BPE) algorithm.
    """

    def __init__(self, texts: str):
        """
        Initializes the BPE object with a given corpus.
        Args:
            text (str): The input corpus for BPE.
        """
        self.texts: list[str] = texts
        self.corpus: dict[tuple[str], int] = {}  # format: {('w','o','r','d', '</w>): 2}
        self.vocab: dict[int, str] = {}  # vocabulaire du corpus as {id: vocab}
        self.bpe_merges: list[str] = []  # Liste des fusions (paires) effectuées

    def create_corpus(self):
        """
        Tokenizes the input text using Byte Pair Encoding (BPE).
        """
        self.texts = [self.clean_text(text) for text in self.texts]
        for text in self.texts:  # Parcours de chaque texte
            for word in text.split():
                token = tuple(list(word) + ["</w>"])  # Ajout du marqueur de fin </w>
                if token not in self.corpus:
                    self.corpus[token] = 1
                else:
                    self.corpus[token] += 1

        logging.info(f"Corpus: {self.corpus}")

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text by removing special characters and converting to lowercase.
        """

        if text is None:
            raise ValueError("Text cannot be None")

        chars_to_delete: list[str] = [
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
            text = text.replace(char, " ")
        text = text.lower()
        return text

    def get_vocab(self):
        """
        Returns the vocabulary of the BPE object as a dict {id: vocab}.
        """
        vocab_set = set()
        for word in self.corpus:
            vocab_set.update(word)
        vocab_list = list(vocab_set)
        self.vocab = {i: v for i, v in enumerate(vocab_list)}
        logging.info(f"Vocabulary: {self.vocab}")

    def find_pairs(self) -> dict[tuple[str], int]:
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

        # Ajout de la paire fusionnée à la liste bpe_merges
        self.bpe_merges.append(most_freq_pair)

        new_corpus = {}
        for word, freq in self.corpus.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == most_freq_pair:
                    new_word.append(word[i] + word[i + 1])
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

    def train(self, iterations: int):
        """
        Learns the BPE codes from the corpus.
        """
        self.create_corpus()
        self.get_vocab()

        for i in range(iterations):
            self.update_vocab()
            logging.info(f"Iteration {i + 1}/{iterations} completed.")

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """
        Converts tokens to their corresponding IDs in the vocabulary.
        Args:
            tokens (list[str]): List of tokens to convert.
        Returns:
            list[int]: List of token IDs.
        """
        return [k for token in tokens for k, v in self.vocab.items() if v == token]

    def ids_to_tokens(self, ids: list[int]) -> list[str]:
        """
        Converts IDs to their corresponding tokens in the vocabulary.
        Args:
            ids (list[int]): List of IDs to convert.
        Returns:
            list[str]: List of tokens.
        """
        return [self.vocab[i] for i in ids if i in self.vocab]
