# change_made=yeni reranker class'ı oluşturduk.
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def re_rank(self, query, docs, max_length=512):
        """
        Parametreler:
        - query: Kullanıcı sorgusu (string)
        - docs: Pinecone'dan gelen belge listesi;
                her 'doc' sözlük veya benzeri bir yapı olabilir.
                İçindeki asıl metni doc["metadata"]["content"] vb. olarak alabilirsiniz.
        - max_length: Tokenizer'ın kesme yapacağı maksimum token sayısı

        Döndürür:
        - Sıralanmış (doc, skor) tuple listesi (skora göre büyükten küçüğe)
        """
        scored_docs = []

        for doc in docs:
            text = doc["metadata"].get("content", "")
            # content yoksa boş string
            if not text.strip():
                continue

            # Sorgu + doküman birleştirerek encode ediyoruz
            inputs = self.tokenizer.encode_plus(
                query, text, return_tensors="pt", truncation=True, max_length=max_length
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Cross-encoder MS MARCO modellerinde genelde tek bir skor (batch_size, 1) döner
                score = outputs.logits[0].item()

            scored_docs.append((doc, score))

        # Skora göre sırala (büyükten küçüğe)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs
