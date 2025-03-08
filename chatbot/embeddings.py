import os
import fnmatch
from git import Repo
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from chatbot.pinecone_utils import connect_pinecone, insert_vectors

# Directory to store cloned repos
CLONE_DIR = "cloned_repos"

BLACKLIST_DIRS = [
    "**/dist/*",  # Build output directories
    "**/node_modules/*",  # Dependency folder for Node.js
    "**/build/*",  # Generic build output folder
    "**/public/*",  # Static assets for frontend projects
    "**/.venv/*",  # Virtual environments for Python
    "**/__pycache__/*",  # Python bytecode cache
    "**/.git/*",  # Git metadata
    "**/.idea/*",  # IDE (e.g., IntelliJ) project files
    "**/.vscode/*",  # Visual Studio Code project files
    "**/logs/*",  # Log files
    "**/coverage/*",  # Test coverage output
    "**/out/*",  # Build or output directories (e.g., TypeScript/JavaScript)
    "**/.next/*",  # Next.js build output
    "**/.expo/*",  # Expo cache for React Native projects
    "**/.cache/*",  # Cache directories
    "**/.eslintcache/*",  # Cache directories
    "**/target/*",  # Maven/Gradle build output (Java projects)
    "**/tmp/*",  # Temporary files
    "**/test-results/*",  # Test result outputs
    "**/cypress/*",  # Cypress testing outputs
    "**/e2e/*",  # End-to-end test directories
    "**/env/*",  # Environment-specific files
    "**/docs/*",  # Documentation directories
    "**/storybook-static/*",  # Storybook build outputs
    "**/functions/node_modules/*",  # Node.js modules in serverless functions
    "**/android/build/*",  # Build output for Android (React Native or native)
    "**/ios/build/*",  # Build output for iOS (React Native or native)
]


def clone_repo(repo_url, codebase_name):
    repo_dir = os.path.join(CLONE_DIR, codebase_name)
    if not os.path.exists(repo_dir):
        print(f"Cloning repo: {repo_url} into {repo_dir}")
        Repo.clone_from(repo_url, repo_dir)
    else:
        print(f"Repo already exists at {repo_dir}. Pulling latest changes.")
        repo = Repo(repo_dir)
        repo.remotes.origin.pull()
    return repo_dir


def get_docs(
    codebase,
    file_names=[
        ".tsx",  # TypeScript React
        ".ts",  # TypeScript
        ".js",  # JavaScript
        ".py",  # Python
        ".cpp",  # C++
        ".java",  # Java
        ".c",  # C
        ".go",  # Go
        ".rb",  # Ruby
        ".php",  # PHP
        ".swift",  # Swift
        ".cs",  # C#
        ".kt",  # Kotlin
        ".rs",  # Rust
        ".scala",  # Scala
        ".m",  # Objective-C
        ".sh",  # Shell script
        ".html",  # HTML
        ".css",  # CSS
        ".json",  # JSON files
        ".xml",  # XML files
        ".sql",  # SQL scripts
    ],
):
    """Load codebase files and return as documents."""
    docs = []
    project_dir = os.path.join(CLONE_DIR, codebase)
    if not os.path.exists(project_dir):
        print(f"No project directory found for codebase: {codebase}")
        return docs

    for dirpath, dirnames, filenames in os.walk(project_dir):
        dirnames[:] = [
            d
            for d in dirnames
            if not any(
                fnmatch.fnmatch(os.path.join(dirpath, d), pattern)
                for pattern in BLACKLIST_DIRS
            )
        ]
        for file in filenames:
            if file.endswith(tuple(file_names)):
                file_path = os.path.join(dirpath, file)
                loader = TextLoader(file_path, encoding="utf-8")
                docs.extend(loader.load_and_split())
    return docs


def save_to_pinecone(codebase):
    """Generate embeddings and save them to Pinecone."""
    connect_pinecone()  # Initialize Pinecone connection
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = get_docs(codebase)
    if not docs:
        print(f"No documents found for codebase: {codebase}")
        return

    vectors = []
    metadata = []
    for doc in docs:
        vector = embeddings.embed_query(doc.page_content)
        vectors.append(vector)
        metadata.append({"source": doc.metadata.get("source", "")})

    insert_vectors(vectors, metadata)  # Use Pinecone for storage
    print(f"Saved {len(vectors)} vectors to Pinecone for codebase: {codebase}")


def get_splitted_texts(codebase):
    docs = get_docs(codebase)
    if not docs:
        print(f"No documents found for codebase: {codebase}")
        return []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)
