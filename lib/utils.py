from pathlib import Path
from typing import List

import nltk
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI
from nltk.tokenize import sent_tokenize

from lib.config import MAX_AGENT_RETRIES
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
        documents += PyPDFLoader(str(pdf)).load()
    return documents


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
