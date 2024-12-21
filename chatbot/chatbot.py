# import os
# from openai import OpenAI
# from chatbot.pinecone_utils import connect_pinecone, search_vectors, insert_vectors
# from langchain_huggingface import HuggingFaceEmbeddings
# from tiktoken import encoding_for_model
# from anthropic import Anthropic


# class Chatbot:
#     def __init__(self, model="gpt-4o-mini-2024-07-18", codebase_name=None):
#         self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
#         self.db = connect_pinecone()
#         self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         self.model = model
#         self.codebase_name = codebase_name
#         self.conversation_history = []

#     def add_to_history(self, question, answer):
#         self.conversation_history.append({"question": question, "answer": answer})

#     # def get_persistent_context(self):
#     #     if not self.conversation_history:
#     #         return ""
#     #     return "\n\n".join(
#     #         f"Q: {entry['question']}\nA: {entry['answer']}"
#     #         for entry in self.conversation_history
#     #     )

#     def get_persistent_context(self):
#         if not self.conversation_history:
#             return ""

#         # Limit the number of entries to reduce token size
#         token_limit = 128000  # Adjust based on model limits
#         context = []
#         token_count = 0

#         # Start from the most recent and work backward
#         for entry in reversed(self.conversation_history):
#             tokens = self.count_tokens(
#                 entry["question"] + entry["answer"], model=self.model
#             )
#             if token_count + tokens > token_limit:
#                 break
#             context.insert(0, f"Q: {entry['question']}\nA: {entry['answer']}")
#             token_count += tokens

#         return "\n\n".join(context)

#     def count_tokens(self, text, model="gpt-4o-mini-2024-07-18"):
#         if "claude" in model.lower():
#             encoding = encoding_for_model("gpt4")
#         else:
#             encoding = encoding_for_model(model)
#         return len(encoding.encode(text))

#     # def get_response(self, question):
#     #     print(f"Received question: {question}")  # Debugging input
#     #     self.last_question = question

#     #     # Perform similarity search on Pinecone
#     #     try:
#     #         query_vector = self.embeddings.embed_query(question)
#     #         print(
#     #             f"Query vector generated: {query_vector[:5]}... (truncated for display)"
#     #         )  # Debugging vector

#     #         search_results = self.db.query(
#     #             vector=query_vector,
#     #             top_k=10,
#     #             include_metadata=True,
#     #             namespace=self.codebase_name,
#     #         )
#     #         print(
#     #             f"Raw search results: {search_results}"
#     #         )  # Debugging raw Pinecone response

#     #         # Validate and extract matches
#     #         matches = search_results.get("matches", [])
#     #         if not matches:
#     #             print("No matches found in Pinecone query.")
#     #             return {
#     #                 "answer": "No relevant documents found for the query.",
#     #                 "related_docs": "",
#     #             }

#     #         print(f"Extracted matches: {matches}")  # Debugging extracted matches
#     #     except Exception as e:
#     #         print(f"Error during similarity search: {e}")
#     #         return {
#     #             "answer": f"Error during similarity search: {e}",
#     #             "related_docs": "",
#     #         }

#     #     # Extract context from matches
#     #     try:
#     #         # Check if matches are in the expected format
#     #         if isinstance(matches, list):
#     #             context = "\n".join(
#     #                 f"Source: {match['metadata'].get('source', 'Unknown')}, Content: {match['metadata'].get('content', 'No content available')}"
#     #                 for match in matches
#     #             )
#     #         else:
#     #             print(f"Unexpected 'matches' format: {type(matches)}")
#     #             return {
#     #                 "answer": "Unexpected data format in Pinecone matches.",
#     #                 "related_docs": "",
#     #             }

#     #         print(f"Extracted context: {context}")  # Debugging extracted context
#     #     except Exception as e:
#     #         print(f"Error while extracting context: {e}")
#     #         return {
#     #             "answer": f"Error while extracting context: {e}",
#     #             "related_docs": "",
#     #         }

#     #     # Build the messages for chat completion
#     #     try:
#     #         persistent_context = self.get_persistent_context()
#     #         print(
#     #             f"Persistent context: {persistent_context}"
#     #         )  # Debugging conversation history

#     #         messages = [
#     #             {
#     #                 "role": "system",
#     #                 "content": "You are a helpful assistant for codebase queries. Provide detailed and thorough answers by leveraging the following context. Cite specific parts of the context when answering. If the question is unrelated, respond with 'I don't know.'",
#     #             },
#     #             {"role": "assistant", "content": persistent_context},
#     #             {"role": "user", "content": f"{context}\n\n{question}"},
#     #         ]
#     #         print(
#     #             f"Generated messages for ChatCompletion: {messages}"
#     #         )  # Debugging messages
#     #     except Exception as e:
#     #         print(f"Error while preparing messages: {e}")
#     #         return {
#     #             "answer": f"Error while preparing messages: {e}",
#     #             "related_docs": "",
#     #         }

#     #     # Generate response using OpenAI
#     #     try:
#     #         # Estimate token usage and dynamically set max_tokens
#     #         input_tokens = sum(
#     #             self.count_tokens(msg["content"], self.model) for msg in messages
#     #         )
#     #         model_token_limit = (
#     #             128000 if self.model == "gpt-4o-mini-2024-07-18" else 8192
#     #         )  # Adjust per model
#     #         available_tokens = model_token_limit - input_tokens
#     #         max_tokens = max(
#     #             256, min(available_tokens, 1000)
#     #         )  # Ensure minimum output size

#     #         print(f"Input tokens: {input_tokens}, Max tokens: {max_tokens}")
#     #         response = self.client.chat.completions.create(
#     #             model=self.model,
#     #             messages=messages,
#     #             max_tokens=max_tokens,
#     #             temperature=0.7,
#     #         )
#     #         print(f"OpenAI response: {response}")  # Debugging OpenAI response

#     #         # Check if response structure matches expected format
#     #         if hasattr(response, "choices") and len(response.choices) > 0:
#     #             self.last_answer = response.choices[0].message.content.strip()
#     #         else:
#     #             print(f"Unexpected OpenAI response format: {response}")
#     #             self.last_answer = "Error: Unexpected OpenAI response format."
#     #     except Exception as e:
#     #         print(f"Error generating response: {e}")
#     #         self.last_answer = f"Error generating response: {e}"

#     #     # Add the current question and answer to the conversation history
#     #     self.add_to_history(question, self.last_answer)
#     #     print(
#     #         f"Updated conversation history: {self.conversation_history}"
#     #     )  # Debugging history

#     #     return {"answer": self.last_answer, "related_docs": context}
#     def get_response(self, question):
#         print(f"Received question: {question}")
#         self.last_question = question

#         # Perform similarity search on Pinecone
#         try:
#             query_vector = self.embeddings.embed_query(question)
#             search_results = self.db.query(
#                 vector=query_vector,
#                 top_k=10,
#                 include_metadata=True,
#                 namespace=self.codebase_name,
#             )

#             matches = search_results.get("matches", [])
#             if not matches:
#                 print("No matches found in Pinecone query.")
#                 return {
#                     "answer": "No relevant documents found for the query.",
#                     "related_docs": "",
#                 }
#         except Exception as e:
#             print(f"Error during similarity search: {e}")
#             return {
#                 "answer": f"Error during similarity search: {e}",
#                 "related_docs": "",
#             }

#         # Extract context from matches
#         try:
#             context = "\n".join(
#                 f"Source: {match['metadata'].get('source', 'Unknown')}, Content: {match['metadata'].get('content', 'No content available')}"
#                 for match in matches
#             )
#         except Exception as e:
#             print(f"Error while extracting context: {e}")
#             return {
#                 "answer": f"Error while extracting context: {e}",
#                 "related_docs": "",
#             }

#         # Get persistent context
#         persistent_context = self.get_persistent_context()

#         try:
#             if "claude" in self.model.lower():
#                 # Handle Claude API request
#                 system_prompt = "You are a helpful assistant for codebase queries. Provide detailed and thorough answers by leveraging the following context. Cite specific parts of the context when answering. If the question is unrelated, respond with 'I don't know.'"

#                 messages = [
#                     {"role": "assistant", "content": persistent_context},
#                     {"role": "user", "content": f"{context}\n\n{question}"},
#                 ]

#                 response = self.anthropic_client.messages.create(
#                     model="claude-3-sonnet-20240229",
#                     system=system_prompt,
#                     messages=messages,
#                     max_tokens=1000,
#                     temperature=0.7,
#                 )
#                 self.last_answer = response.content[0].text.strip()

#             else:
#                 # Handle OpenAI API request
#                 messages = [
#                     {
#                         "role": "system",
#                         "content": "You are a helpful assistant for codebase queries. Provide detailed and thorough answers by leveraging the following context. Cite specific parts of the context when answering. If the question is unrelated, respond with 'I don't know.'",
#                     },
#                     {"role": "assistant", "content": persistent_context},
#                     {"role": "user", "content": f"{context}\n\n{question}"},
#                 ]

#                 input_tokens = sum(
#                     self.count_tokens(msg["content"], self.model) for msg in messages
#                 )
#                 model_token_limit = (
#                     128000 if self.model == "gpt-4o-mini-2024-07-18" else 8192
#                 )
#                 available_tokens = model_token_limit - input_tokens
#                 max_tokens = max(256, min(available_tokens, 1000))

#                 response = self.openai_client.chat.completions.create(
#                     model=self.model,
#                     messages=messages,
#                     max_tokens=max_tokens,
#                     temperature=0.7,
#                 )
#                 self.last_answer = response.choices[0].message.content.strip()

#         except Exception as e:
#             print(f"Error generating response: {e}")
#             self.last_answer = f"Error generating response: {e}"

#         # Add the current question and answer to the conversation history
#         self.add_to_history(question, self.last_answer)

#         return {"answer": self.last_answer, "related_docs": context}
import os
from openai import OpenAI
from anthropic import Anthropic
from chatbot.pinecone_utils import connect_pinecone, search_vectors, insert_vectors
from langchain_huggingface import HuggingFaceEmbeddings
import tiktoken


class Chatbot:
    def __init__(self, model="gpt-4o-mini-2024-07-18", codebase_name=None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.db = connect_pinecone()
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.model = model
        self.codebase_name = codebase_name
        self.conversation_history = []
        self.tokenizer = self._get_tokenizer()

    def _get_tokenizer(self):
        """Get the appropriate tokenizer based on model"""
        if "claude" in self.model.lower():
            return tiktoken.get_encoding("cl100k_base")
        else:
            try:
                return tiktoken.get_encoding("cl100k_base")
            except KeyError:
                return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text, model=None):
        """Count tokens using the initialized tokenizer"""
        return len(self.tokenizer.encode(text))

    def add_to_history(self, question, answer):
        self.conversation_history.append({"question": question, "answer": answer})

    def get_persistent_context(self):
        if not self.conversation_history:
            return ""

        # Set token limits based on model
        if "claude" in self.model.lower():
            token_limit = 100000  # Claude's larger context window
        else:
            token_limit = 128000 if self.model == "gpt-4o-mini-2024-07-18" else 8192

        context_messages = []
        token_count = 0

        # Start from the most recent and work backward
        for entry in reversed(self.conversation_history):
            tokens = self.count_tokens(entry["question"] + entry["answer"])
            if token_count + tokens > token_limit:
                break
            context_messages.insert(0, {"role": "user", "content": entry["question"]})
            context_messages.insert(
                0, {"role": "assistant", "content": entry["answer"]}
            )
            token_count += tokens

        return context_messages

    def get_response(self, question):
        print(f"Received question: {question}")
        self.last_question = question

        # Perform similarity search on Pinecone
        try:
            query_vector = self.embeddings.embed_query(question)
            search_results = self.db.query(
                vector=query_vector,
                top_k=10,
                include_metadata=True,
                namespace=self.codebase_name,
            )

            matches = search_results.get("matches", [])
            if not matches:
                print("No matches found in Pinecone query.")
                return {
                    "answer": "No relevant documents found for the query.",
                    "related_docs": "",
                }
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return {
                "answer": f"Error during similarity search: {e}",
                "related_docs": "",
            }

        # Extract context from matches
        try:
            context = "\n".join(
                f"Source: {match['metadata'].get('source', 'Unknown')}, Content: {match['metadata'].get('content', 'No content available')}"
                for match in matches
            )
        except Exception as e:
            print(f"Error while extracting context: {e}")
            return {
                "answer": f"Error while extracting context: {e}",
                "related_docs": "",
            }

        # Get persistent context
        context_messages = self.get_persistent_context()

        try:
            if "claude" in self.model.lower():
                # Handle Claude API request
                system_prompt = "You are a helpful assistant for codebase queries. Provide detailed and thorough answers by leveraging the following context. Cite specific parts of the context when answering. If the question is unrelated, respond with 'I don't know.'"

                # Format messages for Claude
                messages = []
                if context_messages:
                    messages.extend(context_messages)

                # Add the current question with context
                messages.append({"role": "user", "content": f"{context}\n\n{question}"})

                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    system=system_prompt,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7,
                )
                self.last_answer = response.content[0].text.strip()

            else:
                # Handle OpenAI API request
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for codebase queries. Provide detailed and thorough answers by leveraging the following context. Cite specific parts of the context when answering. If the question is unrelated, respond with 'I don't know.'",
                    }
                ]

                if context_messages:
                    messages.extend(context_messages)

                messages.append({"role": "user", "content": f"{context}\n\n{question}"})

                # Calculate available tokens
                input_tokens = sum(
                    self.count_tokens(msg["content"]) for msg in messages
                )
                model_token_limit = (
                    128000 if self.model == "gpt-4o-mini-2024-07-18" else 8192
                )
                available_tokens = model_token_limit - input_tokens
                max_tokens = max(256, min(available_tokens, 1000))

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                self.last_answer = response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            self.last_answer = f"Error generating response: {e}"

        # Add to conversation history
        self.add_to_history(question, self.last_answer)

        return {"answer": self.last_answer, "related_docs": context}
