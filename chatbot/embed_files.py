from chatbot.embeddings import get_db

# Replace "API-Project" with the appropriate codebase name
codebase_name = "API-Project"
db = get_db(codebase_name, create_new=True)

print(f"Embedding for {codebase_name} completed.")
