import os
from openai import OpenAI
from chatbot.pinecone_utils import connect_pinecone, search_vectors, insert_vectors
from langchain_huggingface import HuggingFaceEmbeddings


class Chatbot:
    def __init__(self, model="gpt-4", codebase_name=None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.db = connect_pinecone()
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.model = model
        self.codebase_name = codebase_name
        self.conversation_history = []

    def add_to_history(self, question, answer):
        self.conversation_history.append({"question": question, "answer": answer})

    def get_persistent_context(self):
        if not self.conversation_history:
            return ""
        return "\n\n".join(
            f"Q: {entry['question']}\nA: {entry['answer']}"
            for entry in self.conversation_history
        )

    def get_response(self, question):
        print(f"Received question: {question}")  # Debugging input
        self.last_question = question

        # Perform similarity search on Pinecone
        try:
            query_vector = self.embeddings.embed_query(question)
            print(
                f"Query vector generated: {query_vector[:5]}... (truncated for display)"
            )  # Debugging vector

            search_results = self.db.query(
                vector=query_vector,
                top_k=10,
                include_metadata=True,
                namespace=self.codebase_name,
            )
            print(
                f"Raw search results: {search_results}"
            )  # Debugging raw Pinecone response

            # Validate and extract matches
            matches = search_results.get("matches", [])
            if not matches:
                print("No matches found in Pinecone query.")
                return {
                    "answer": "No relevant documents found for the query.",
                    "related_docs": "",
                }

            print(f"Extracted matches: {matches}")  # Debugging extracted matches
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return {
                "answer": f"Error during similarity search: {e}",
                "related_docs": "",
            }

        # Extract context from matches
        try:
            # Check if matches are in the expected format
            if isinstance(matches, list):
                context = "\n".join(
                    f"Source: {match['metadata'].get('source', 'Unknown')}, Content: {match['metadata'].get('content', 'No content available')}"
                    for match in matches
                )
            else:
                print(f"Unexpected 'matches' format: {type(matches)}")
                return {
                    "answer": "Unexpected data format in Pinecone matches.",
                    "related_docs": "",
                }

            print(f"Extracted context: {context}")  # Debugging extracted context
        except Exception as e:
            print(f"Error while extracting context: {e}")
            return {
                "answer": f"Error while extracting context: {e}",
                "related_docs": "",
            }

        # Build the messages for chat completion
        try:
            persistent_context = self.get_persistent_context()
            print(
                f"Persistent context: {persistent_context}"
            )  # Debugging conversation history

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant for codebase queries. Provide detailed and thorough answers by leveraging the following context. Cite specific parts of the context when answering. If the question is unrelated, respond with 'I don't know.'",
                },
                {"role": "assistant", "content": persistent_context},
                {"role": "user", "content": f"{context}\n\n{question}"},
            ]
            print(
                f"Generated messages for ChatCompletion: {messages}"
            )  # Debugging messages
        except Exception as e:
            print(f"Error while preparing messages: {e}")
            return {
                "answer": f"Error while preparing messages: {e}",
                "related_docs": "",
            }

        # Generate response using OpenAI
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )
            print(f"OpenAI response: {response}")  # Debugging OpenAI response

            # Check if response structure matches expected format
            if hasattr(response, "choices") and len(response.choices) > 0:
                self.last_answer = response.choices[0].message.content.strip()
            else:
                print(f"Unexpected OpenAI response format: {response}")
                self.last_answer = "Error: Unexpected OpenAI response format."
        except Exception as e:
            print(f"Error generating response: {e}")
            self.last_answer = f"Error generating response: {e}"

        # Add the current question and answer to the conversation history
        self.add_to_history(question, self.last_answer)
        print(
            f"Updated conversation history: {self.conversation_history}"
        )  # Debugging history

        return {"answer": self.last_answer, "related_docs": context}
