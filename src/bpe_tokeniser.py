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
            text (str): The input corpus for BPE.
        """
        self.text: str = text
        self.corpus: dict[tuple[str], int] = {}  # format: {('w','o','r','d', '</w>): 2}
        self.vocab: list[str] = []  # vocabulaire du corpus
        self.bpe_merges: list[str] = []  # Liste des fusions (paires) effectuées

    def create_corpus(self):
        """
        Tokenizes the input text using Byte Pair Encoding (BPE).
        """
        self.text = self.clean_text(self.text)
        for word in self.text.split():
            token = tuple(list(word) + ["</w>"])
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
        Returns the vocabulary of the BPE object.
        """
        vocab = set()
        for word in self.corpus:
            vocab.update(word)
        self.vocab = list(vocab)
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

    def tokenize(self, text: str):
        """
        Tokenizes the input text using the learned BPE codes.
        Args:
            text (str): The input text to tokenize.
        Returns:
            list[list[str]]: Tokenized text by word.
        """
        # clean the text
        text = self.clean_text(text)

        words = text.strip().split()
        tokenized_words = []

        for word in words:
            # Découpe le mot en symboles individuels
            symbols = list(word) + ["</w>"]

            # Applique les fusions dans l’ordre appris
            for merge in self.bpe_merges:
                i = 0
                while i < len(symbols) - 1:
                    if (symbols[i], symbols[i + 1]) == merge:
                        symbols[i : i + 2] = [symbols[i] + symbols[i + 1]]
                        i = max(i - 1, 0)  # recule pour vérifier les nouvelles fusions
                    else:
                        i += 1

            tokenized_words.append(symbols)

        return tokenized_words
