import os

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    """
    Load a PDF file and return a list of LangChain Document objects (one per page).

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError:        If the file cannot be read, or yields no extractable text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"PDF file not found: '{file_path}'. "
            "Check that the path is correct and the file exists."
        )

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not read PDF '{file_path}'. "
            "The file may be corrupted, password-protected, or not a valid PDF.\n"
            f"Details: {exc}"
        ) from exc

    if not documents:
        raise ValueError(
            f"No text content could be extracted from '{file_path}'. "
            "The PDF may be empty or contain only scanned images (no selectable text)."
        )

    return documents