import os
import argparse
import pickle
from git import Repo
from transformers import AutoTokenizer, AutoModelForCausalLM
from chromadb import Client
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction
from tqdm import tqdm

# Constants for model and vector database
MODEL_NAME = "meta-llama/Llama-3b"
DB_PATH = "vector_db.pkl"

# Function to load the model and tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Function to ingest Git repos and store embeddings
def ingest_repo(repo_dir, tokenizer, model, db_path=DB_PATH):
    client = Client()
    embedding_function = HuggingFaceEmbeddingFunction(MODEL_NAME)
    collection = client.create_collection("git_repos", embedding_function=embedding_function)
    
    repo = Repo(repo_dir)
    commits = list(repo.iter_commits())
    
    for commit in tqdm(commits, desc="Processing Commits"):
        author = commit.author.name
        for file in commit.stats.files:
            file_path = os.path.join(repo_dir, file)
            if os.path.exists(file_path) and file_path.endswith(".py"):
                with open(file_path, "r") as f:
                    code = f.read()
                # Tokenize and embed file content
                tokens = tokenizer(code, return_tensors="pt")
                embeddings = model(**tokens).last_hidden_state.mean(dim=1).detach().numpy()
                # Store in vector DB with metadata
                collection.add(embeddings=embeddings, 
                               metadatas={"author": author, "file": file_path, "commit": commit.hexsha, "code": code})
    
    # Save the collection to disk
    with open(db_path, 'wb') as f:
        pickle.dump(collection, f)

# Function to load the vector database from disk
def load_vector_db(db_path=DB_PATH):
    with open(db_path, 'rb') as f:
        return pickle.load(f)

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
        repos = [os.path.join(repo_directory, repo_name) for repo_name in os.listdir(repo_directory)]
        
        for repo_path in tqdm(repos, desc="Processing Repositories"):
            ingest_repo(repo_path, tokenizer, model)
        print("Training complete. Vector database saved to:", DB_PATH)
    
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