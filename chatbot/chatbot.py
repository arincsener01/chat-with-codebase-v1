import os
from openai import OpenAI
from chatbot.pinecone_utils import connect_pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings


class Chatbot:
    def __init__(self, model="gpt-4o", codebase_name=None):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
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
            )  # Print part of the vector

            search_results = self.db.query(
                vector=query_vector,
                top_k=10,
                include_metadata=True,
            )
            print(f"Search results: {search_results}")  # Debugging Pinecone results
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return {
                "answer": f"Error during similarity search: {e}",
                "related_docs": "",
            }

        # Extract context from search results
        try:
            context = "\n".join(
                f"Source: {match['metadata']['source']}, Content: {match.get('values', '')}"
                for match in search_results["matches"]
            )
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
                    "content": "You are an expert assistant helping with codebase queries. Provide detailed and thorough answers by leveraging the following context. Make sure to explain in simple terms when possible, and cite specific parts of the context when answering. If the question is unrelated to the codebase or seems like a random query (e.g., "
                    "How is the weather today?"
                    "), respond with "
                    "I don't know."
                    "",
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
            print("my name is arınç")
            response = self.client.chat.completions.create(
                # model=self.model,
                model="gpt-4o",
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )
            print(f"OpenAI response: {response}")  # Debugging OpenAI response
            # self.last_answer = response["choices"][0]["message"]["content"].strip()
            self.last_answer = response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            self.last_answer = f"Error generating response: {e}"

        # Add the current question and answer to the conversation history
        self.add_to_history(question, self.last_answer)
        print(
            f"Updated conversation history: {self.conversation_history}"
        )  # Debugging history

        return {"answer": self.last_answer, "related_docs": context}
