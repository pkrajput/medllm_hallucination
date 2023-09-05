import os
import yaml
import openai
from tqdm.auto import tqdm
from index_creator import PineconeIndex
from langchain.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def check_error_flag():
    return os.path.isfile("error_flag.txt")


def load_state():
    if os.path.isfile("last_processed_index.txt"):
        with open("last_processed_index.txt", "r") as file:
            return int(file.read())
    return None


# Function to load files
def load_file(filename):
    loader = PyPDFium2Loader(filename)
    pages = loader.load_and_split()
    return pages


# Function to get PDF content from files in a directory
def get_pdf_content(root_dir):
    filenames = [root_dir + name for name in os.listdir(root_dir)]
    split_pages_text = []
    for filename in filenames:
        pages = load_file(filename)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        )
        split_pages = text_splitter.split_documents(pages)
        for split_page in split_pages:
            split_pages_text.append(
                split_page.page_content.replace("\n", "")
                .replace("\r", " ")
                .replace("\t", " ")
                .strip()
            )
    maxlen = 0
    for split_page_text in split_pages_text:
        maxlen = max(maxlen, len(split_page_text))
    return split_pages_text


# Function to vectorize and upload PDF content
def vectorize_and_upload_pdf_content(
    index, embedding_model_name, textlist, openai_api_key
):
    batch_size = 100
    openai.api_key = openai_api_key
    for i in tqdm(range(0, len(textlist), batch_size)):
        lines_batch = textlist[i : i + batch_size]
        res = openai.Embedding.create(input=lines_batch, engine=embedding_model_name)
        embeddings = [record["embedding"] for record in res["data"]]
        meta = [{"text": line} for line in lines_batch]
        index_id = [str(j) for j in range(i, i + len(lines_batch))]
        to_upsert = zip(index_id, embeddings, meta)
        index.upsert(vectors=list(to_upsert))


# Function to process PDF files
def process_pdf_file(start_idx=0):
    with open("./vector_db/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    index = PineconeIndex().get_index()
    openai_api_key = config["openai"]["embedding"]["api_key"]
    embedding_model_name = config["openai"]["embedding"]["model_name"]
    textlist = get_pdf_content(config["pinecone"]["path"])
    vectorize_and_upload_pdf_content(
        index, embedding_model_name, textlist[start_idx:], openai_api_key
    )


if __name__ == "__main__":

    if not check_error_flag():
        last_processed_index = load_state()
        if last_processed_index is not None:
            process_pdf_file(start_idx=last_processed_index)
        else:
            process_pdf_file()

    if os.path.isfile("error_flag.txt"):
        os.remove("error_flag.txt")
