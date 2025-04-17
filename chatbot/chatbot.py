import os
from openai import OpenAI
from anthropic import Anthropic
from chatbot.pinecone_utils import connect_pinecone, search_vectors, insert_vectors
from langchain_huggingface import HuggingFaceEmbeddings
import tiktoken
from chatbot.reranker_config import CrossEncoderReranker
import google.generativeai as genai


class Chatbot:
    def __init__(self, model="gpt-4o-mini-2024-07-18", codebase_name=None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.gemini_client = genai
        self.db = connect_pinecone()
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.model = model
        self.codebase_name = codebase_name
        self.conversation_history = []
        self.tokenizer = self._get_tokenizer()
        self.reranker_config = CrossEncoderReranker()

    def _get_tokenizer(self):
        """Get the appropriate tokenizer based on model"""
        if "claude" in self.model.lower():
            return tiktoken.get_encoding("cl100k_base")
        elif "gemini" in self.model.lower():
            # Gemini uses a different tokenization mechanism
            # For estimation, we'll still use cl100k_base as an approximation
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
        elif "gemini" in self.model.lower():
            token_limit = 1000000  # Gemini 2.5 Pro has a very large context window
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

    def get_recent_user_messages(self, num_messages=3):
        """Retrieve the last N user and assistant messages as text for embedding."""
        if not self.conversation_history:
            return ""

        recent_entries = self.conversation_history[-num_messages:]
        combined = "\n".join(
            f"User: {entry['question']}\nAssistant: {entry['answer']}"
            for entry in recent_entries
        )
        return combined

    def format_context_for_llm(self, matches):
        """Format retrieved documents into a well-structured context for the LLM."""
        formatted_contexts = []

        for idx, match in enumerate(matches):
            source = match["metadata"].get("source", "Unknown")
            content = match["metadata"].get("content", "No content available")

            # Create a citation key
            citation_key = f"[{idx+1}]"

            # Format context with citation
            formatted_context = (
                f"{citation_key} Source: {source}\n" f"Content: {content}\n"
            )
            formatted_contexts.append(formatted_context)

        # Combine formatted contexts with citation instruction
        final_context = (
            f"Below are relevant documents to answer the user's question:\n\n"
            f"{''.join(formatted_contexts)}\n"
            f"Please cite the sources using their numbers [1], [2], etc. when answering."
        )

        return final_context

    def refine_query(self, original_query):
        """Use the LLM to refine the query before search."""
        prompt = f"""
        You're a search query optimizer. Your task is to rewrite the following query
        to be more effective for vector database retrieval. Make it more specific 
        and include important keywords without changing the original intent.
        
        Original query: {original_query}
        
        Rewritten query:
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a cheaper model for this task
            messages=[
                {
                    "role": "system",
                    "content": "You are a search query optimization assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # Lower temperature for more focused results
        )

        refined_query = response.choices[0].message.content.strip()

        # Extract just the query if the response contains explanation
        if "Rewritten query:" in refined_query:
            refined_query = refined_query.split("Rewritten query:")[-1].strip()

        print(f"[DEBUG] Original query: {original_query}")
        print(f"[DEBUG] Refined query: {refined_query}")

        return refined_query

    def get_response(self, question):
        print(f"Received question: {question}")
        self.last_question = question

        refined_question = self.refine_query(question)
        print(f"[DEBUG] Using refined question: {refined_question}")

        # Perform similarity search on Pinecone
        try:
            # query_vector = self.embeddings.embed_query(question)
            recent_context = self.get_recent_user_messages(num_messages=3)
            augmented_query = (
                f"{recent_context}\nUser Question: {refined_question}"
                if recent_context
                else refined_question
            )
            print("[DEBUG-QUERY] Augmented Query for Embedding:\n", augmented_query)

            query_vector = self.embeddings.embed_query(augmented_query)

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
        # change_made= reranker eklemesi
        try:
            # matches, genelde [ {id:..., score:..., metadata:{...}}, {...}, ... ] şeklinde
            print("[DEBUG] Reranker devreye giriyor...")
            reranked = self.reranker_config.re_rank(question, matches)
            # Artık (doc, skor) döndü. En ilgili en üstte olacak.
            for idx, (doc, score) in enumerate(reranked):
                print(
                    f"[DEBUG] doc {idx} => Score: {score:.2f}, Source: {doc['metadata'].get('source', 'N/A')}"
                )

            # Belirli sayıda doküman seçelim, örn. en iyi 5 doküman
            top_docs = reranked[:5]

            # 5) Bu dokümanları 'context' haline getirme
            reranked_matches = [doc for doc, _ in top_docs]
            context = self.format_context_for_llm(reranked_matches)

        except Exception as e:
            print(f"Error during re-ranking: {e}")
            return {
                "answer": f"Error during re-ranking: {e}",
                "related_docs": "",
            }
        # reranker bitiş

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
                    max_tokens=4096,
                    temperature=0.7,
                )
                self.last_answer = response.content[0].text.strip()

            elif "gemini" in self.model.lower():
                # Handle Gemini API request
                system_prompt = "You are a helpful assistant for codebase queries. Provide detailed and thorough answers by leveraging the following context. Cite specific parts of the context when answering. If the question is unrelated, respond with 'I don't know.'"

                # Configure the Gemini model
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_output_tokens": 4096,
                }

                gemini_model = self.gemini_client.GenerativeModel(
                    model_name="gemini-2.5-pro-exp-03-25",  # Use the experimental model
                    generation_config=generation_config,
                )

                # Format conversation history for Gemini
                gemini_history = []
                if context_messages:
                    for msg in context_messages:
                        if msg["role"] == "user":
                            gemini_history.append(
                                {"role": "user", "parts": [msg["content"]]}
                            )
                        elif msg["role"] == "assistant":
                            gemini_history.append(
                                {"role": "model", "parts": [msg["content"]]}
                            )

                # Create a chat session with history
                chat = gemini_model.start_chat(history=gemini_history)

                # Send the prompt with context
                prompt = (
                    f"{system_prompt}\n\nContext: {context}\n\nQuestion: {question}"
                )
                response = chat.send_message(prompt)

                self.last_answer = response.text

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
                max_tokens = max(256, min(available_tokens, 4096))

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

        token_count = self.count_tokens(self.last_answer)
        print(f"[DEBUG] Token count of the answer: {token_count}")

        return {"answer": self.last_answer, "related_docs": context}
