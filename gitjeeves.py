import os
import sys
import torch
import argparse
from git import Repo
from transformers import AutoTokenizer, AutoModel
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction
from chromadb.config import Settings
from tqdm import tqdm


# Constants for model and vector database
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PERSIST_DIRECTORY = "./chroma_db"  # Directory where Chroma's SQLite DB files will be stored


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to load the model and tokenizer
def load_model_and_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model
    except OSError as e:
        print(e)
        print("Try: huggingface-cli login")
        sys.exit(1)

# Function to ingest Git repos and store embeddings
def ingest_repo(repo_dir, tokenizer, model):
    client = PersistentClient()
    embedding_function = HuggingFaceEmbeddingFunction(MODEL_NAME)
    collection = client.get_or_create_collection("git_repos", embedding_function=embedding_function)

    repo = Repo(repo_dir)
    commits = list(repo.iter_commits())

    # Get the set of already processed commit hashes
    processed_commits = set()
    existing_metadata = collection.get(include=["metadatas"])
    if existing_metadata:
        processed_commits = {metadata["commit"] for metadata in existing_metadata["metadatas"]}

    for commit in tqdm(commits, desc=f"Commits for {os.path.basename(repo_dir)}"):
        if commit.hexsha in processed_commits:
            continue  # Skip already processed commits

        author = commit.author.name
        for file in commit.stats.files:
            file_path = os.path.join(repo_dir, file)
            if os.path.exists(file_path) and file_path.endswith(".py"):
                with open(file_path, "r") as f:
                    code = f.read()
                # Tokenize and embed file content
                tokens = tokenizer(code, return_tensors="pt")
                embeddings = model(**tokens).last_hidden_state.mean(dim=1).detach().numpy()
                # Create a unique ID
                unique_id = f"{commit.hexsha}_{file_path}"
                # Store in vector DB with metadata
                collection.add(ids=[unique_id],  # Unique ID required
                               embeddings=embeddings, 
                               metadatas={"author": author, "file": file_path, "commit": commit.hexsha, "code": code})

# Function to load the vector database
def load_vector_db():
    settings = Settings(
        chroma_db_impl="sqlite", 
        persist_directory=PERSIST_DIRECTORY
    )
    client = Client(settings)
    collection = client.get_collection("git_repos")
    return collection

# Enhanced Querying for Specific Use Cases
def query_repo(query, collection, tokenizer, model):
    query_tokens = tokenizer(query, return_tensors="pt")
    query_embedding = model(**query_tokens).last_hidden_state.mean(dim=1).detach().numpy()
    results = collection.query(embeddings=query_embedding, top_k=5)

    refined_results = []
    for result in results:
        metadata = result['metadata']
        code = metadata.get('code', '')
        if "def " in code and "OAuth" in code:  # Simple check for API functions and OAuth mentions
            refined_results.append(metadata)

    return refined_results

# Command-line argument parsing
def main():
    parser = argparse.ArgumentParser(description="Train or Serve a Vector Database from Git Repos using Llama3")
    parser.add_argument('--train', type=str, help="Path to the directory containing Git repos for training")
    parser.add_argument('--serve', action='store_true', help="Serve the model for querying via CLI")

    args = parser.parse_args()

    # Load the model and tokenizer
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME)

    if args.train:
        # Train mode
        repo_directory = args.train
        repos = [os.path.join(repo_directory, repo_name) for repo_name in os.listdir(repo_directory) if os.path.isdir(os.path.join(repo_directory, repo_name))]

        for repo_path in tqdm(repos, desc="Processing Repositories"):
            ingest_repo(repo_path, tokenizer, model)
        print("Training complete. Vector database saved to:", PERSIST_DIRECTORY)

    elif args.serve:
        # Serve mode
        collection = load_vector_db()
        print("Vector database loaded. Enter your queries below (multi-line input, Ctrl+D to submit):")

        while True:
            try:
                print("Query: ", end="", flush=True)
                query_lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    query_lines.append(line)
                query = "\n".join(query_lines)

                results = query_repo(query, collection, tokenizer, model)
                if results:
                    for result in results:
                        print(f"File: {result['file']}, Commit: {result['commit']}, Author: {result['author']}")
                else:
                    print("No results found.")
            except EOFError:
                print("\nExiting serve mode.")
                break


if __name__ == "__main__":
    main()
