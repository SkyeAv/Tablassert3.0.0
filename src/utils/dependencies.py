import subprocess
import nltk
import sys


def main() -> None:
    """
    Main function to download necessary NLP resources.

    This function downloads the 'en_core_web_sm' model for spaCy and
    the 'stopwords' and 'punkt' resources for NLTK.
    """
    # Download spaCy's English core web model
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        check=True)

    # Download NLTK's stopwords and punkt tokenizer
    nltk.download("stopwords")
    nltk.download("punkt")


if __name__ == "__main__":
    main()
