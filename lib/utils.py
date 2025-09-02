from pathlib import Path
from typing import List

import chromadb
import nltk
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from nltk.tokenize import sent_tokenize

from lib.config import (
    MAX_AGENT_RETRIES,
    VECTOR_STORE_PATH,
)
from lib.prompts import IT_AGENT_PROMPT, RC_AGENT_PROMPT

nltk.download("punkt")


def load_documents(source_directory: str) -> List[Document]:
    """Creates a list of langchain documents from pdf documents
    in the source directory using unstructured as a loader.

    Args:
        source_directory (str): directory where the raw pdfs are stored
    """
    documents = []
    pdf_files = list(Path(source_directory).glob("*.pdf"))
    for pdf in pdf_files:
        documents += PyPDFLoader(pdf).load()
    return documents


def split_documents_basic(
    documents: List[Document], chunk_size: int, include_linear_index: bool = False
) -> List[Document]:
    """Splits documents into chunks of specified size using
    recursive character splitter
    If specified, adds a linear index that encodes the sequential
    of the chunks within each document, made to handle multiple documents.

    Args:
        documents (List[Document]): list of langchain document objects
        chunk_size (int): size of chunks
        include_linear_index (bool): whether to include linear index
    """
    splitted_text = RecursiveCharacterTextSplitter(chunk_size=chunk_size).split_documents(documents)
    if include_linear_index:
        for i in range(len(splitted_text)):
            if (i == 0) or (splitted_text[i - 1].metadata["source"] != splitted_text[i].metadata["source"]):
                splitted_text[i].metadata["linear_index"] = 0
            else:
                splitted_text[i].metadata["linear_index"] = splitted_text[i - 1].metadata["linear_index"] + 1
    return splitted_text


def drop_document_duplicates(documents: List[Document]) -> List[Document]:
    """Simply removes duplicates from a list of Langchain Documents.

    Args:
        documents (List[Document]): list of docs
    """
    new_docs = []
    doc_contents = []
    for doc in documents:
        if doc.page_content not in doc_contents:
            doc_contents.append(doc.page_content)
            new_docs.append(doc)
    return new_docs


def load_vector_store(
    embedding: AzureOpenAIEmbeddings, collection_name: str, vs_location: str = VECTOR_STORE_PATH
) -> Chroma:
    if (Path(vs_location) / "chroma.sqlite3").exists():
        temp_client = chromadb.PersistentClient(vs_location)
        try:
            temp_client.get_collection(name=collection_name)
            chroma_db = Chroma(
                collection_name=collection_name,
                persist_directory=vs_location,
                embedding_function=embedding,
            )
            logger.info(f"Successfully loaded vector store from `{vs_location}` with collection `{collection_name}`")
            return chroma_db
        except ValueError:
            raise ValueError("No Collection with this name found")
    else:
        raise FileNotFoundError("No Chroma vector store found")


def build_vector_store(
    documents: List[Document],
    embedding: AzureOpenAIEmbeddings,
    collection_name: str,
    vs_location: str = VECTOR_STORE_PATH,
    distance_function: str = "cosine",
    erase_existing: bool = False,
) -> None:
    """Creates a persistent vector store from a list of documents.

    Args:
        documents (List[Document]): list of chunks
        embedding (ModelSetup): embedding model
        collection_name (str): Name of the collection to create.
        vs_location (str, optional): Location of created vector store.
        Defaults to VECTOR_STORE_PATH.
        distance_function (str, optional): Distance function to use.
        Defaults to "cosine".
        erase_existing (bool, optional): Whether to erase existing vector store.
    """
    try:
        load_vector_store(
            embedding=embedding,
            collection_name=collection_name,
            vs_location=vs_location,
        )
        logger.info("""Vector store and collection already exists.""")
        if erase_existing:
            logger.info(f"""parameters `erase_existing` is set to `{erase_existing}` > will perform deletion""")
            temp_client = chromadb.PersistentClient(vs_location)
            temp_client.delete_collection(collection_name)
            logger.info("Successfully deleted existing collection")
            raise ValueError("Creating new collection")

    except (FileNotFoundError, ValueError) as e:
        Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            collection_name=collection_name,
            persist_directory=vs_location,
            collection_metadata={"hnsw:space": distance_function},
        )
        if isinstance(e, FileNotFoundError):
            logger.info("No existing vector store, created new with collection")
        elif isinstance(e, ValueError):
            logger.info("Found vector store but not collection, created new collection")


class AgentChunker:
    """Class that instantiates a chunking method that uses an
    LLM agent to decide where to separate the text
    Can perform recursive or iterative chunking.
    """

    def __init__(self, agent: AzureChatOpenAI, recursive: bool = True, chunk_size: int = 256) -> None:
        """Class initializer.

        Args:
            agent (AzureChatOpenAI): LLM agent
            recursive (bool, optional): Whether to perform recursive (True)
            or iterative (False) chunking. Defaults to True.
            chunk_size (int, optional): Max chunk size. Defaults to 256.
        """
        self.agent = agent
        self.agent.max_tokens = 1
        self.recursive = recursive
        self.chunk_size = chunk_size

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Main method to be called on a list of
        documents in order to split them into chunks.

        Args:
            docs (List[Document]): list of Langchain Documents to chunk

        Returns:
            List[Document]: list of chunks
        """
        chunks = []
        for doc in docs:
            sent = sent_tokenize(doc.page_content)
            if self.recursive:
                doc_chunks = self.recursively_split(sent)
            else:
                doc_chunks = self.iteratively_split(sent)
            for i in range(len(doc_chunks)):
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk"] = i
                chunk_text = ""
                for j in range(len(doc_chunks[i])):
                    chunk_text += doc_chunks[i][j]
                chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
        return chunks

    def recursively_split(self, sentence_list: List[str]) -> List[List[str]]:
        """Submethod for the recursive split.

        Args:
            sentence_list (List[str]): list of strings corresponding
            to the text sentences

        Returns:
            List[List[str]]: list of lists of sentences, corresponding to chunks
        """
        text = self.format_text_for_llm(sentence_list, break_points=True)
        if len(text) > self.chunk_size:
            splits = []
            prompt = [
                ("system", RC_AGENT_PROMPT),
                ("human", text),
            ]
            split_id = self.get_response_and_check(full_prompt=prompt, sentence_count=len(sentence_list))
            splits.extend(self.recursively_split(sentence_list=sentence_list[:split_id]))
            splits.extend(self.recursively_split(sentence_list=sentence_list[split_id:]))
            return splits
        return [sentence_list]

    def iteratively_split(self, sentence_list: List[str]) -> List[List[str]]:
        """Submethod for the iterative split.

        Args:
            sentence_list (List[str]): list of strings
            corresponding to the text sentences

        Returns:
            List[List[str]]: list of lists of sentences, corresponding to chunks
        """
        splits = []
        for i in range(len(sentence_list) - 1):
            if i == 0:
                splits.append([sentence_list[0]])
            new_sentence = self.format_text_for_llm(sentence_list[i + 1])
            last_chunk = self.format_text_for_llm(splits[-1])
            prompt = [
                ("system", IT_AGENT_PROMPT),
                (
                    "human",
                    f"""PASSAGE: {last_chunk}

                 SENTENCE: {new_sentence}""",
                ),
            ]
            new_chunk = bool(self.get_response_and_check(full_prompt=prompt))
            if new_chunk:
                splits.append([new_sentence])
            else:
                splits[-1].extend([new_sentence])
        return splits

    def get_response_and_check(self, full_prompt: List[tuple[str, str]], sentence_count: int = 1) -> int:
        """Function to call agent and extract int response.
        Makes sure the response is in the correct format and if not, retries.

        Args:
            full_prompt (List[Tuple[str, str]]): fully assembled prompt
            sentence_count (int, optional): Number of sentences in
            text (variable only for recursive). Defaults to 1.

        Raises:
            ValueError: not a parsable int, or out of bounds
            RuntimeError: too mModelSetup errors exceeding max retries

        Returns:
            int: valid integer response
        """
        retries = 0
        upper_bound = (1 - self.recursive) + self.recursive * sentence_count
        while retries < MAX_AGENT_RETRIES:
            split_id = self.agent.invoke(full_prompt).content.replace(" ", "")
            try:
                split_id = int(split_id)
                if 0 <= split_id <= upper_bound:
                    print(f"Found after {retries} retries")
                    return split_id
                retries += 1
                raise ValueError("Out of range")
            except (ValueError, TypeError):
                retries += 1
        raise RuntimeError(f"Max retries exceeded, last response: {split_id}")

    def format_text_for_llm(self, sentences: List[str], break_points: bool = False) -> str:
        """Function to change list of sentences to one string
        to be read by an llm. Adds break points if need be.

        Args:
            sentences (List[str]): list of sentences
            break_points (bool, optional): Whether to add break
            points between the sentences. Defaults to False.

        Returns:
            str: full string
        """
        formatted_sentences = """"""
        for i in range(len(sentences)):
            formatted_sentences += f"""{sentences[i]}"""
            if (i < len(sentences)) and break_points:
                formatted_sentences += f"""

                    Break Point {i + 1}

                    """

        return formatted_sentences
