# -*- coding: utf-8 -*-
"""
Full RAG pipeline with three chunking strategies: word-based, sentence-based, semantic-based
Includes:
 - Persian normalizer and SQuAD mapper
 - Fine-tune embedding model (contrastive) if train_small provided or load existing fine-tuned model
 - Chunking (word, sentence, semantic)
 - Index building (BM25, TF-IDF, Embedding)
 - Retrieval methods (BM25, TF-IDF, Embedding, Hybrid)
 - QA pipeline (transformers) and evaluation (F1/EM/CosSim/MRR/Precision/Recall/Hit@K)

Requirements: sentence-transformers, transformers, rank_bm25, nltk, PyPDF2, evaluate, scikit-learn, torch

Make sure to provide train_small and val_small datasets in the expected format (list of dicts with keys: id, question, context, answers)
"""
import os, re, json, random, torch
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from evaluate import load
from tqdm import tqdm
import nltk
import statistics
# ==========================
# Configs (قابل تنظیم)
# ==========================


BASE = './qadata/'

PDF_PATH = globals().get('BASE')+'Drugs.pdf'
OUTPUT_DIR = globals().get('BASE')+'outputs'
EMBEDDING_MODEL_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
FINE_TUNED_EMBEDDING_DIR = globals().get('BASE')+'models/finetuned-emb'

TOP_K = globals().get('TOP_K', 5)
ALPHA = globals().get('ALPHA', 0.4)
# DEVICE = globals().get('DEVICE', 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
DEVICE = 'cuda'

# Chunking params
CHUNK_SIZE_WORD = globals().get('CHUNK_SIZE_WORD', 220)
WORD_OVERLAP = globals().get('WORD_OVERLAP', 20)
MAX_SENT_PER_CHUNK = globals().get('MAX_SENT_PER_CHUNK', 3)

# Semantic chunking params
SEM_MAX_SENT = globals().get('SEM_MAX_SENT', 8)
SEM_MIN_SENT = globals().get('SEM_MIN_SENT', 2)
SEM_SIM_THRESHOLD = globals().get('SEM_SIM_THRESHOLD', 0.45)
SEM_OVERLAP_SENT = globals().get('SEM_OVERLAP_SENT', 1)

# Training hyperparams for fine-tune
BATCH_SIZE = globals().get('BATCH_SIZE', 16)
EPOCHS = globals().get('EPOCHS', 1)


nltk.download('punkt_tab')
# -------------------------
# Data loader
# -------------------------
def read_qa(path):
    ds = []
    with open(Path(path), encoding="utf-8") as f:
        squad = json.load(f)
    for example in squad["data"]:
        title = example.get("title", "").strip()
        for paragraph in example["paragraphs"]:
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                answers = [a["text"].strip() for a in qa["answers"]]
                answer_starts = [a["answer_start"] for a in qa["answers"]]
                ds.append({
                    "title": title,
                    "context": context,
                    "question": qa["question"].strip(),
                    "id": qa["id"],
                    "answers": {"answer_start": answer_starts, "text": answers}
                })
    return ds

train_ds = read_qa("./qadata/pqa_train.json")
val_ds   = read_qa("./qadata/pqa_test.json")

train_dataset = Dataset.from_list(train_ds)
val_dataset   = Dataset.from_list(val_ds)
raw_ds = DatasetDict({"train": train_dataset, "validation": val_dataset})

# Load and map the new dataset
rag_eval_ds = read_qa("./qadata/drugs_aq_dataset.json")
rag_eval_dataset = Dataset.from_list(rag_eval_ds)


# ==========================
# Persian Normalizer + SQuAD Mapper
# ==========================

def normalize_persian(text: str) -> str:
    if not text:
        return ""
    # remove ZWNJ, normalize arabic letters to persian, collapse spaces
    text = text.replace("\u200c", " ").replace("ي", "ی").replace("ك", "ک")
    return re.sub(r"\s+", " ", text).strip()


def map_to_squad(example):
    answers = {"text": [normalize_persian(t) for t in example["answers"]["text"]],
               "answer_start": example["answers"].get("answer_start", [])}
    return {
        "id": str(example.get("id", "")),
        "context": normalize_persian(example.get("context", "")),
        "question": normalize_persian(example.get("question", "")),
        "answers": answers,
    }


mapped = raw_ds.map(map_to_squad)
train_small =mapped["train"]
val_small   = mapped["validation"]
rag_eval_dataset_mapped = rag_eval_dataset.map(map_to_squad)


# ==========================
# Fine-tune Embedding Model
# ==========================

# اگر مدل فاین‌تیون‌شده وجود داشته باشد، آن را لود کن؛ در غیر این صورت فاین‌تیون کن
if os.path.exists(FINE_TUNED_EMBEDDING_DIR) and os.listdir(FINE_TUNED_EMBEDDING_DIR):
    print(f"Found existing fine-tuned model at {FINE_TUNED_EMBEDDING_DIR}, loading...")
    embedding_model_finetuned = SentenceTransformer(FINE_TUNED_EMBEDDING_DIR)
else:
    print("No fine-tuned model found — starting fine-tuning (if train_small available)")

    if not train_small:
        print("train_small is empty. Skipping fine-tuning. Make sure to provide train_small or place a model in FINE_TUNED_EMBEDDING_DIR")
        # fallback to base embedding model
        embedding_model_finetuned = SentenceTransformer(EMBEDDING_MODEL_ID)
    else:
        # Prepare pairs
        train_examples = []
        for i in tqdm(range(len(train_small)), desc="Preparing fine-tune examples"):
            example = train_small[i]
            query = normalize_persian(example.get('question', ''))
            positive = normalize_persian(example.get('context', ''))
            # Random negative
            negative_idx = random.randint(0, len(train_small) - 1)
            while negative_idx == i:
                negative_idx = random.randint(0, len(train_small) - 1)
            negative = normalize_persian(train_small[negative_idx].get('context', ''))
            train_examples.append(InputExample(texts=[query, positive], label=1.0))
            train_examples.append(InputExample(texts=[query, negative], label=0.0))

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

        os.environ["WANDB_DISABLED"] = "true"
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)

        train_loss = losses.ContrastiveLoss(model=embedding_model)

        print("Starting fine-tuning embedding model...")
        embedding_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=EPOCHS, warmup_steps=100)

        os.makedirs(FINE_TUNED_EMBEDDING_DIR, exist_ok=True)
        embedding_model.save(FINE_TUNED_EMBEDDING_DIR)
        embedding_model_finetuned = SentenceTransformer(FINE_TUNED_EMBEDDING_DIR)
        print(f"Fine-tuned embedding model saved to {FINE_TUNED_EMBEDDING_DIR}")


# ==========================
# Extract and process PDF text
# ==========================

def extract_pdf_text(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pdf_text = ''
            for page in reader.pages:
                pdf_text += (page.extract_text() or '') + '\n'
        return normalize_persian(pdf_text)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

pdf_text = extract_pdf_text(PDF_PATH)
print(f"PDF text length: {len(pdf_text)} characters")

# ==========================
# Chunking (word / sentence / semantic)
# ==========================

def word_based_chunk(text, chunk_size=CHUNK_SIZE_WORD, overlap=WORD_OVERLAP):
    if not text:
        return []
    words = word_tokenize(text)
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunks.append(' '.join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks


def sentence_based_chunk(text, max_sentences=MAX_SENT_PER_CHUNK):
    if not text:
        return []
    sentences = [s.strip() for s in sent_tokenize(text) if s and s.strip()]
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk_sentences = sentences[i:i + max_sentences]
        if chunk_sentences:
            chunks.append(' '.join(chunk_sentences))
    return chunks


def semantic_based_chunk(text, model: SentenceTransformer = embedding_model_finetuned,
                         max_sent=SEM_MAX_SENT,
                         min_sent=SEM_MIN_SENT,
                         sim_threshold=SEM_SIM_THRESHOLD,
                         overlap_sent=SEM_OVERLAP_SENT):
    if not text:
        return []
    sentences = [s.strip() for s in sent_tokenize(text) if s and s.strip()]
    if not sentences:
        return []
    sent_embs = model.encode(sentences, show_progress_bar=False)
    chunks = []
    start = 0
    while start < len(sentences):
        end = min(len(sentences), start + min_sent)
        cur_embs = sent_embs[start:end]
        centroid = np.mean(cur_embs, axis=0, keepdims=True)
        while end < len(sentences) and (end - start) < max_sent:
            next_emb = sent_embs[end:end+1]
            sim = cosine_similarity(centroid, next_emb)[0][0]
            if sim >= sim_threshold:
                cur_embs = np.vstack([cur_embs, next_emb])
                centroid = np.mean(cur_embs, axis=0, keepdims=True)
                end += 1
            else:
                break
        chunk_text = ' '.join(sentences[start:end])
        if chunk_text.strip():
            chunks.append(chunk_text)
        if end >= len(sentences):
            break
        start = max(end - overlap_sent, start + 1)
    return chunks

# Create chunks
if pdf_text:
    word_chunks = word_based_chunk(pdf_text)
    sentence_chunks = sentence_based_chunk(pdf_text)
    semantic_chunks = semantic_based_chunk(pdf_text)
    print(f"Word-based chunks: {len(word_chunks)}")
    print(f"Sentence-based chunks: {len(sentence_chunks)}")
    print(f"Semantic-based chunks: {len(semantic_chunks)}")
else:
    word_chunks, sentence_chunks, semantic_chunks = [], [], []
    print("PDF text not found!")

# ==========================
# Build retrieval indexes
# ==========================

def build_bm25(chunks):
    if not chunks:
        return None
    tokenized_chunks = [word_tokenize(chunk) for chunk in chunks]
    return BM25Okapi(tokenized_chunks)

def build_tfidf(chunks):
    if not chunks:
        return None, None
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix

def build_embedding_index(chunks, embedding_model):
    if not chunks:
        return None
    print("Calculating embeddings for chunks...")
    chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    return chunk_embeddings

# Build indexes
print("Building retrieval indexes for word-based chunks...")
bm25_word = build_bm25(word_chunks)
tfidf_vectorizer_word, tfidf_matrix_word = build_tfidf(word_chunks)
embedding_index_word = build_embedding_index(word_chunks, embedding_model_finetuned)

print("Building retrieval indexes for sentence-based chunks...")
bm25_sentence = build_bm25(sentence_chunks)
tfidf_vectorizer_sentence, tfidf_matrix_sentence = build_tfidf(sentence_chunks)
embedding_index_sentence = build_embedding_index(sentence_chunks, embedding_model_finetuned)

print("Building retrieval indexes for semantic-based chunks...")
bm25_semantic = build_bm25(semantic_chunks)
tfidf_vectorizer_semantic, tfidf_matrix_semantic = build_tfidf(semantic_chunks)
embedding_index_semantic = build_embedding_index(semantic_chunks, embedding_model_finetuned)
# ==========================
# Retrieval functions
# ==========================

def retrieve_bm25(query, bm25_index, chunks, top_k=TOP_K):
    if not bm25_index or not chunks:
        return [], []
    tokenized_query = word_tokenize(normalize_persian(query))
    scores = bm25_index.get_scores(tokenized_query)
    scores = scores / max(scores) if max(scores) > 0 else scores
    top_indices = np.argsort(scores)[-top_k:][::-1]
    retrieved = [chunks[i] for i in top_indices]
    return retrieved, scores[top_indices]




def retrieve_tfidf(query, vectorizer, tfidf_matrix, chunks, top_k=TOP_K):
    if not vectorizer or tfidf_matrix is None or not chunks:
        return [], []
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, tfidf_matrix).flatten()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in top_idx], sims[top_idx]


def retrieve_embedding(query, chunk_embeddings, chunks, embedding_model, top_k=TOP_K):
    if chunk_embeddings is None or not chunks:
        return [], []
    q_emb = embedding_model.encode([query])
    sims = cosine_similarity(q_emb, chunk_embeddings).flatten()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in top_idx], sims[top_idx]


def hybrid_retrieve(query, chunks, bm25_index, chunk_embeddings, embedding_model, alpha=ALPHA, top_k=TOP_K):
    if not chunks:
        return [], []
    # BM25
    tokenized_query = word_tokenize(query)
    if bm25_index:
        bm25_scores = bm25_index.get_scores(tokenized_query)
        bm25_scores = bm25_scores / max(bm25_scores) if max(bm25_scores) > 0 else bm25_scores
    else:
        bm25_scores = np.zeros(len(chunks))
    # Embedding
    if chunk_embeddings is not None:
        q_emb = embedding_model.encode([query])
        emb_scores = cosine_similarity(q_emb, chunk_embeddings).flatten()
    else:
        emb_scores = np.zeros(len(chunks))
    # Combine
    final_scores = alpha * bm25_scores + (1 - alpha) * emb_scores
    top_idx = np.argsort(final_scores)[-top_k:][::-1]
    return [chunks[i] for i in top_idx], final_scores[top_idx]

# ==========================
# QA system
# ==========================

def initialize_qa_pipeline():
    try:
        # use a multilingual QA model (XQuAD-finetuned) which works better for non-English text
        qa = pipeline(
            "question-answering",
            model="mrm8488/bert-multi-cased-finetuned-xquadv1",
            device=0 if DEVICE == "cuda" else -1,
        )
        return qa
    except Exception as e:
        print(f"Error loading QA model: {e}")
        try:
            # fallback to original if multilingual model not available
            return pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=0 if DEVICE == "cuda" else -1,
            )
        except Exception as e2:
            print(f"Fallback QA load failed: {e2}")
            return None

qa_pipeline = initialize_qa_pipeline()


def answer_question(question, context, max_length=200):
    if not qa_pipeline or not context.strip():
        return "No answer found"
    try:
        if len(context) > 2000:
            context = context[:2000]
        result = qa_pipeline(question=question, context=context)
        return result.get('answer', 'No answer found')
    except Exception as e:
        print(f"Error in answering: {e}")
        return "Error in answering"

# ==========================
# RAG pipeline
# ==========================

def rag_pipeline(query, chunks, retrieval_method="hybrid", top_k=TOP_K, **kwargs):
    if retrieval_method == "bm25":
        retrieved_chunks, scores = retrieve_bm25(query, kwargs.get('bm25_index'), chunks, top_k)
    elif retrieval_method == "tfidf":
        retrieved_chunks, scores = retrieve_tfidf(query, kwargs.get('vectorizer'), kwargs.get('tfidf_matrix'), chunks, top_k)
    elif retrieval_method == "embedding":
        retrieved_chunks, scores = retrieve_embedding(query, kwargs.get('chunk_embeddings'), chunks, kwargs.get('embedding_model'), top_k)
    elif retrieval_method == "hybrid":
        retrieved_chunks, scores = hybrid_retrieve(query, chunks, kwargs.get('bm25_index'), kwargs.get('chunk_embeddings'), kwargs.get('embedding_model'), alpha=ALPHA, top_k=top_k)
    else:
        return "Invalid retrieval method", [], []

    if retrieved_chunks:
        context = ' '.join(retrieved_chunks)
        answer = answer_question(query, context)
        return answer, retrieved_chunks, scores
    else:
        return "No relevant information found", [], []

# ==========================
# Advanced evaluation metrics
# ==========================

def calculate_cosine_similarity(pred_answer, gt_answer, embedding_model):
    if not pred_answer.strip() or not gt_answer.strip():
        return 0.0
    try:
        pred_emb = embedding_model.encode([pred_answer])
        gt_emb = embedding_model.encode([gt_answer])
        return float(cosine_similarity(pred_emb, gt_emb)[0][0])
    except:
        return 0.0


def calculate_mrr(retrieved_chunks, gt_context, embedding_model, threshold=0.75):
    if not retrieved_chunks or not gt_context or not gt_context.strip():
        return 0.0
    try:
        # use the ground-truth context (or answer) embedding as the target
        gt_embedding = embedding_model.encode([gt_context])
        for rank, chunk in enumerate(retrieved_chunks, 1):
            if not chunk.strip():
                continue
            chunk_emb = embedding_model.encode([chunk])
            sim = cosine_similarity(gt_embedding, chunk_emb)[0][0]
            if sim >= threshold:
                return 1.0 / rank
        return 0.0
    except Exception:
        return 0.0


def evaluate_retrieval_advanced(retrieved_chunks, gt_context, embedding_model, threshold=0.75):
    """
    Returns (hit, precision, recall)
    - hit: 1 if any retrieved chunk similarity >= threshold else 0
    - precision: relevant_retrieved / retrieved_count
    - recall: proxy: 1.0 if any chunk contains the gt_context substring, else relevant_retrieved / retrieved_count
    """
    if not retrieved_chunks or not gt_context or not gt_context.strip():
        return 0, 0.0, 0.0
    try:
        gt_embedding = embedding_model.encode([gt_context])
        sims = []
        for chunk in retrieved_chunks:
            if not chunk.strip():
                sims.append(0.0)
                continue
            sim = cosine_similarity(gt_embedding, embedding_model.encode([chunk]))[0][0]
            sims.append(float(sim))
        relevant = sum(1 for s in sims if s >= threshold)
        hit = 1 if relevant > 0 else 0
        precision = relevant / len(retrieved_chunks) if retrieved_chunks else 0.0
        # conservative recall proxy: if exact substring of gt_context present in any chunk -> 1.0, otherwise fraction
        recall = 1.0 if any(gt_context.strip() in chunk for chunk in retrieved_chunks) else (relevant / len(retrieved_chunks) if retrieved_chunks else 0.0)
        return hit, precision, recall
    except Exception:
        return 0, 0.0, 0.0

# ==========================
# Comprehensive evaluation
# ==========================

def comprehensive_evaluation(dataset, chunks, chunk_type, retrieval_kwargs, max_examples=100):
    print(f"\n{'='*50}")
    print(f"Evaluating {chunk_type}")
    print(f"{'='*50}")

    # quick sanity checks
    print(f"Chunks count: {len(chunks) if chunks else 0}")
    if isinstance(dataset, list):
        data_iter = dataset[:max_examples]
    else:
        # huggingface Dataset or DatasetDict entries
        try:
            data_iter = [dict(example) for example in dataset.select(range(min(max_examples, len(dataset))))]
        except Exception:
            # fallback
            data_iter = list(dataset)[:max_examples] if dataset else []

    if not data_iter:
        print("Warning: dataset empty or failed to iterate")
        return {m: {k: 0.0 for k in ["f1","em","cosine_sim","mrr","hit","precision","recall"]} for m in ["bm25","tfidf","embedding","hybrid"]}

    # show one sample for debugging
    print("Example data sample (first):")
    print({k: data_iter[0].get(k) for k in ["id","question","answers","context"]})

    squad_metric = load("squad")
    methods = ["bm25", "tfidf", "embedding", "hybrid"]
    results = {}

    for method in methods:
        print(f"\nEvaluating {method} method...")
        method_results = {"f1": [], "em": [], "cosine_sim": [], "mrr": [], "hit": [], "precision": [], "recall": []}

        for example in tqdm(data_iter):
            if not example.get('answers', {}).get('text'):
                continue
            query = example['question']
            gt_answer = example['answers']['text'][0]
            gt_context = example['context']

            try:
                pred_answer, retrieved_chunks, scores = rag_pipeline(
                    query, chunks, retrieval_method=method, **retrieval_kwargs
                )
                qa_metric = squad_metric.compute(
                    predictions=[{"id": str(example['id']), "prediction_text": pred_answer}],
                    references=[{"id": str(example['id']), "answers": example['answers']}]
                )
                cos_sim = calculate_cosine_similarity(pred_answer, gt_answer, embedding_model_finetuned)
                mrr = calculate_mrr(retrieved_chunks, gt_context, embedding_model_finetuned)
                hit, precision, recall = evaluate_retrieval_advanced(retrieved_chunks, gt_context, embedding_model_finetuned)

                method_results["f1"].append(float(qa_metric.get('f1', 0.0)))
                method_results["em"].append(float(qa_metric.get('exact_match', 0.0)))
                method_results["cosine_sim"].append(float(cos_sim))
                method_results["mrr"].append(float(mrr))
                method_results["hit"].append(int(hit))
                method_results["precision"].append(float(precision))
                method_results["recall"].append(float(recall))
            except Exception as e:
                # don't fail the loop; print minimal debug info
                print(f"Error example {example.get('id','NA')}: {e}")
                continue

        # safe mean calculation
        aggregated = {}
        for metric, values in method_results.items():
            try:
                aggregated[metric] = statistics.mean(values) if values else 0.0
            except Exception:
                aggregated[metric] = float(np.mean(values)) if values else 0.0
        results[method] = aggregated

        print(f"\n{method} Results:")
        print(f"F1: {results[method]['f1']:.4f}")
        print(f"EM: {results[method]['em']:.4f}")
        print(f"Cosine Similarity: {results[method]['cosine_sim']:.4f}")
        print(f"MRR: {results[method]['mrr']:.4f}")
        print(f"Hit@{TOP_K}: {results[method]['hit']:.4f}")
        print(f"Precision: {results[method]['precision']:.4f}")
        print(f"Recall: {results[method]['recall']:.4f}")

    return results


# ==========================
# Run evaluation
# ==========================
print("Starting comprehensive system evaluation...")

# Word-based
word_retrieval_kwargs = {
    'bm25_index': bm25_word,
    'vectorizer': tfidf_vectorizer_word,
    'tfidf_matrix': tfidf_matrix_word,
    'chunk_embeddings': embedding_index_word,
    'embedding_model': embedding_model_finetuned,
}

def comprehensive_evaluation(dataset, chunks, chunk_type, retrieval_kwargs, max_examples=100):
   
    print(f"\n{'='*50}")
    print(f"Evaluating {chunk_type}")
    print(f"{'='*50}")

    # quick sanity checks
    print(f"Chunks count: {len(chunks) if chunks else 0}")
    if isinstance(dataset, list):
        data_iter = dataset[:max_examples]
    else:
        # huggingface Dataset or DatasetDict entries
        try:
            data_iter = [dict(example) for example in dataset.select(range(min(max_examples, len(dataset))))]
        except Exception:
            # fallback
            data_iter = list(dataset)[:max_examples] if dataset else []

    if not data_iter:
        print("Warning: dataset empty or failed to iterate")
        return {m: {k: 0.0 for k in ["f1","em","cosine_sim","mrr","hit","precision","recall"]} for m in ["bm25","tfidf","embedding","hybrid"]}

    # show one sample for debugging
    print("Example data sample (first):")
    print({k: data_iter[0].get(k) for k in ["id","question","answers","context"]})

    squad_metric = load("squad")
    methods = ["bm25", "tfidf", "embedding", "hybrid"]
    results = {}

    for method in methods:
        print(f"\nEvaluating {method} method...")
        method_results = {"f1": [], "em": [], "cosine_sim": [], "mrr": [], "hit": [], "precision": [], "recall": []}

        if not data_iter:
            # بدون دیتاست: صفرها را نگه می‌داریم تا کد fail نشود
            results[method] = {k: 0.0 for k in method_results}
            continue

        for example in tqdm(data_iter):

            

            if not example.get('answers', {}).get('text'):
                continue
            query = example['question']
            gt_answer = example['answers']['text'][0]
            gt_context = example['context']

            try:
                pred_answer, retrieved_chunks, scores = rag_pipeline(
                    query, chunks, retrieval_method=method, **retrieval_kwargs
                )
                qa_metric = squad_metric.compute(
                    predictions=[{"id": str(example['id']), "prediction_text": pred_answer}],
                    references=[{"id": str(example['id']), "answers": example['answers']}]
                )
                cos_sim = calculate_cosine_similarity(pred_answer, gt_answer, embedding_model_finetuned)
                mrr = calculate_mrr(retrieved_chunks, gt_context, embedding_model_finetuned)
                hit, precision, recall = evaluate_retrieval_advanced(retrieved_chunks, gt_context, embedding_model_finetuned)

                method_results["f1"].append(qa_metric['f1'])
                method_results["em"].append(qa_metric['exact_match'])
                method_results["cosine_sim"].append(cos_sim)
                method_results["mrr"].append(mrr)
                method_results["hit"].append(hit)
                method_results["precision"].append(precision)
                method_results["recall"].append(recall)


            except Exception as e:
                print(f"Error processing example {example.get('id', 'NA')}: {e}")
                continue
        results[method]={ k:statistics.mean(v) for (k,v) in  method_results.items()}
    return results

# word_results = comprehensive_evaluation(val_small, word_chunks, "Word-based Chunking", word_retrieval_kwargs)
word_results_new = comprehensive_evaluation(rag_eval_dataset_mapped, word_chunks, "Word-based Chunking", word_retrieval_kwargs)

# Sentence-based
sentence_retrieval_kwargs = {
    'bm25_index': bm25_sentence,
    'vectorizer': tfidf_vectorizer_sentence,
    'tfidf_matrix': tfidf_matrix_sentence,
    'chunk_embeddings': embedding_index_sentence,
    'embedding_model': embedding_model_finetuned,
}
# sentence_results = comprehensive_evaluation(val_small, sentence_chunks, "Sentence-based Chunking", sentence_retrieval_kwargs)
sentence_results_new = comprehensive_evaluation(
    rag_eval_dataset_mapped, sentence_chunks, "Sentence-based Chunking", sentence_retrieval_kwargs
)

# Semantic-based
semantic_retrieval_kwargs = {
    'bm25_index': bm25_semantic,
    'vectorizer': tfidf_vectorizer_semantic,
    'tfidf_matrix': tfidf_matrix_semantic,
    'chunk_embeddings': embedding_index_semantic,
    'embedding_model': embedding_model_finetuned,
}
# semantic_results = comprehensive_evaluation(val_small, semantic_chunks, "Semantic-based Chunking", semantic_retrieval_kwargs)
semantic_results_new = comprehensive_evaluation(
    rag_eval_dataset_mapped, semantic_chunks, "Semantic-based Chunking", semantic_retrieval_kwargs
)


# word_results = word_results or {}
# sentence_results = sentence_results or {}
# semantic_results = semantic_results or {}

word_results = word_results_new or {}
sentence_results = sentence_results_new or {}
semantic_results = semantic_results_new or {}
# ==========================
# Final comparison and summary
# ==========================
print(f"\n{'='*80}")
print("Results Summary and Comparison")
print(f"{'='*80}")


def print_comparison_table(results_map):
    methods = ["bm25", "tfidf", "embedding", "hybrid"]
    print(f"\n{'Method':<12} {'Chunk Type':<16} {'F1':<8} {'EM':<8} {'CosSim':<8} {'MRR':<8} {'Hit@K':<8} {'Prec':<8} {'Rec':<8}")
    print("-" * 100)
    for method in methods:
        for chunk_label, res in results_map.items():
            r = res[method]
            print(f"{method:<12} {chunk_label:<16} {r['f1']:<8.4f} {r['em']:<8.4f} {r['cosine_sim']:<8.4f} {r['mrr']:<8.4f} {r['hit']:<8.4f} {r['precision']:<8.4f} {r['recall']:<8.4f}")
        print("-" * 100)

results_map = {
    'Word-based': word_results,
    'Sentence-based': sentence_results,
    'Semantic-based': semantic_results,
}
print_comparison_table(results_map)

 
 

# ==========================
# Find best method (across all chunk types)
# ==========================

def find_best_method(all_results):
    weights = {'f1': 0.25, 'em': 0.20, 'cosine_sim': 0.20, 'mrr': 0.15, 'hit': 0.10, 'precision': 0.05, 'recall': 0.05}
    methods = ["bm25", "tfidf", "embedding", "hybrid"]
    best_scores = {}
    for chunk_label, res in all_results.items():
        for method in methods:
            score = sum(weights[m] * res[method][m] for m in weights.keys())
            best_scores[f"{method}__{chunk_label}"] = score
    best_method = max(best_scores, key=best_scores.get)
    return best_method, best_scores[best_method], best_scores

# best_method, best_score, all_scores = find_best_method(results_map)
best_method, best_score, all_scores = find_best_method(results_map)

print(f"\n{'='*60}")
print("Best Method Analysis")
print(f"{'='*60}")
print(f"Best method: {best_method}")
print(f"Overall score: {best_score:.4f}")
print(f"\nComplete ranking:")
for i, (k, v) in enumerate(sorted(all_scores.items(), key=lambda x: x[1], reverse=True), 1):
    print(f"{i:2d}. {k:<28}: {v:.4f}")

# ==========================
# Save results
# ==========================

def save_results(word_results, sentence_results, semantic_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results_summary = {
        "word_based_results": word_results,
        "sentence_based_results": sentence_results,
        "semantic_based_results": semantic_results,
        "best_method": best_method,
        "best_score": best_score,
        "evaluation_settings": {
            "top_k": TOP_K,
            "chunk_size_word": CHUNK_SIZE_WORD,
            "alpha_hybrid": ALPHA,
            "embedding_model": EMBEDDING_MODEL_ID,
            "fine_tuned": True,
            "sem_params": {
                "SEM_MAX_SENT": SEM_MAX_SENT,
                "SEM_MIN_SENT": SEM_MIN_SENT,
                "SEM_SIM_THRESHOLD": SEM_SIM_THRESHOLD,
                "SEM_OVERLAP_SENT": SEM_OVERLAP_SENT,
            }
        }
    }
    results_file = os.path.join(output_dir, "task_b_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {results_file}")

save_results(word_results, sentence_results, semantic_results, OUTPUT_DIR)
save_results(word_results_new, sentence_results_new, semantic_results_new, OUTPUT_DIR)
# ==========================
# Interactive QA demo (optional)
# ==========================

def interactive_qa_demo():
    print(f"\n{'='*60}")
    print("Interactive RAG System Demo")
    print(f"{'='*60}")
    print("Type 'quit' to exit")

    # parse best
    method, chunk_label = best_method.split('__')
    if 'Word' in chunk_label:
        chunks = word_chunks; kwargs = word_retrieval_kwargs
    elif 'Sentence' in chunk_label:
        chunks = sentence_chunks; kwargs = sentence_retrieval_kwargs
    else:
        chunks = semantic_chunks; kwargs = semantic_retrieval_kwargs

    print(f"Using method: {method} with {chunk_label} chunking")

    while True:
        try:
            question = input("\nYour question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!"); break
            if not question: continue
            answer, retrieved_chunks, scores = rag_pipeline(question, chunks, retrieval_method=method, **kwargs)
            print(f"\nAnswer: {answer}")
            print(f"\nNumber of retrieved chunks: {len(retrieved_chunks)}")
            print("\nRelevant texts:")
            for i, (chunk, score) in enumerate(zip(retrieved_chunks[:3], scores[:3])):
                print(f"{i+1}. (Score: {score:.3f}) {chunk[:200]}...")
        except KeyboardInterrupt:
            print("\nGoodbye!"); break
        except Exception as e:
            print(f"Error: {e}")

# interactive_qa_demo()

# ==========================
# Summary
# ==========================
print(f"\n{'='*80}")
print("Task B Execution Summary")
print(f"{'='*80}")
print("Completed tasks:")
print("   - استفاده از مدل فاین‌تیون‌شده برای امبدینگ")
print("   - پیاده‌سازی سه روش چانکینگ: کلمه‌ای، جمله‌ای، سمانتیک")
print("   - چهار روش بازیابی: BM25، TF-IDF، Embedding، Hybrid")
print("   - ارزیابی با F1، EM، Cosine Similarity، MRR، Precision، Recall، Hit@K")
print("   - مقایسه‌ی جامع بین روش‌های چانکینگ")
print("\nKey results:")
print(f"   - Best method: {best_method}")
print(f"   - Best method overall score: {best_score:.4f}")
print("\nImplemented improvements:")
print("   - سمانتیک‌-چانکینگ با threshold و centroid")
print("   - بهبود جدول مقایسه برای سه نوع چانکینگ")
print("   - ذخیره‌ی نتایج با پارامترهای سمانتیک")
print(f"\nSaved files:\n   - Embedding model: {FINE_TUNED_EMBEDDING_DIR}\n   - Evaluation results: {OUTPUT_DIR}/task_b_results.json")
print(f"\n{'='*80}")
print("Task B completed successfully!")
print(f"{'='*80}")