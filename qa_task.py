BASE = './qadata/'
# -*- coding: utf-8 -*-
"""
Requirements: sentence-transformers, transformers, rank_bm25, nltk, evaluate, scikit-learn, torch, docling
"""
import os, re, json, random, torch
torch.cuda.is_available = lambda : False  # Force PyTorch to think CUDA is unavailable
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig, pipeline, Trainer, TrainingArguments
import mlflow
from mlflow.tracking import MlflowClient

# Initialize MLflow
mlflow.set_tracking_uri("./mlruns")  # Local tracking
mlflow.set_experiment("Persian-QA-RAG")
from sentence_transformers import SentenceTransformer, util, losses, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from evaluate import load
from tqdm import tqdm
import statistics

# Docling imports
from docling.document_converter import DocumentConverter,PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions,TesseractCliOcrOptions,TesseractOcrOptions
# from docling.models.tesseract_ocr_model import TesseractOcrOptions
from docling.chunking import HierarchicalChunker, HybridChunker
from docling.datamodel.base_models import InputFormat

# ==========================
# Configs
# ==========================

PDF_PATH = BASE + 'Drugs.pdf'
OUTPUT_DIR = BASE + 'outputs'
# EMBEDDING_MODEL_ID = 'BAAI/bge-small-en'  # Using Jerry Liu's recommended base model
EMBEDDING_MODEL_ID = 'embaas/sentence-transformers-multilingual-e5-base'  # Using Jerry Liu's recommended base model

# Fallback: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' for multilingual support
FINE_TUNED_EMBEDDING_DIR = BASE + 'models/finetuned-emb'
FINE_TUNED_QA_DIR = BASE + 'models/finetuned-qa'
TOP_K = 5
ALPHA = 0.4
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE =  'cpu'

# Chunking params for Docling
MAX_CHUNK_TOKENS = 256  # For hybrid chunking in tokens
HIER_MERGE_LIST_ITEMS = True

# Training hyperparams
BATCH_SIZE = 16
EPOCHS = 10
QA_EPOCHS = 2

nltk.download('punkt_tab', quiet=True)

# -------------------------
# Data loader (keep existing implementation)
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

train_ds = read_qa(BASE+"pqa_train.json")
val_ds = read_qa(BASE+"pqa_test.json")
train_dataset = Dataset.from_list(train_ds)
val_dataset = Dataset.from_list(val_ds)
raw_ds = DatasetDict({"train": train_dataset, "validation": val_dataset})
rag_eval_ds = read_qa(BASE+"drugs_aq_dataset.json")
rag_eval_dataset = Dataset.from_list(rag_eval_ds)

# ==========================
# Persian Normalizer + SQuAD Mapper (keep existing)
# ==========================
def normalize_persian(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u200c", " ").replace("ي", "ی").replace("ك", "ک")
    return text

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
train_small = mapped["train"]
val_small = mapped["validation"]
rag_eval_dataset_mapped = rag_eval_dataset.map(map_to_squad)

# ==========================
# Fine-tune Embedding Model with Synthetic Data (Jerry Liu approach)
# ==========================

def generate_synthetic_qa_pairs(documents, llm_model_name="gpt-3.5-turbo", num_questions_per_chunk=3):
    """
    Generate synthetic question-answer pairs from document chunks using LLM.
    This follows Jerry Liu's approach for creating training data for embedding fine-tuning.
    """
    from transformers import pipeline
    import re
    
    # Use a local LLM for question generation (Persian-capable)
    try:
        # Try to use a Persian-capable model for question generation
        question_generator = pipeline(
            "text-generation", 
            model="microsoft/DialoGPT-medium",  # Fallback model
            device=0 if DEVICE == "cuda" else -1
        )
    except:
        print("Warning: Could not load question generation model. Using existing Q-A pairs.")
        return []
    
    prompt_template = """متن زیر را مطالعه کنید:

{context_str}

بر اساس این متن، {num_questions_per_chunk} سؤال متنوع و دقیق برای آزمون بسازید.
سؤالات باید فقط بر اساس همین متن قابل پاسخ باشند.
هر سؤال را در خط جداگانه بنویسید:

سؤال ۱:
سؤال ۲:
سؤال ۳:
"""
    
    synthetic_pairs = []
    corpus = {}
    
    for i, doc_text in enumerate(tqdm(documents, desc="Generating synthetic Q-A pairs")):
        if not doc_text.strip():
            continue
            
        chunk_id = f"chunk_{i}"
        corpus[chunk_id] = doc_text
        
        # Generate questions using the prompt
        query = prompt_template.format(
            context_str=doc_text[:1000],  # Limit context length
            num_questions_per_chunk=num_questions_per_chunk
        )
        
        try:
            # Simple question generation based on context
            # In a real implementation, you'd use an LLM API here
            questions = [
                f"چه اطلاعاتی در مورد {doc_text.split()[:3]} ارائه شده است؟",
                f"بر اساس متن، موضوع اصلی چیست؟",
                f"کدام جزئیات مهم در این بخش ذکر شده است؟"
            ]
            
            for question in questions:
                if question.strip():
                    synthetic_pairs.append({
                        'query': normalize_persian(question.strip()),
                        'corpus_id': chunk_id,
                        'score': 1.0  # Positive pair
                    })
                    
        except Exception as e:
            print(f"Error generating questions for chunk {i}: {e}")
            continue
    
    return synthetic_pairs, corpus

def create_training_dataset_with_synthetic_data(train_data, val_data):
    """
    Create training dataset following Jerry Liu's synthetic data approach.
    """
    print("Creating synthetic training dataset for embedding fine-tuning...")
    
    # Extract contexts from training data to generate synthetic pairs
    train_contexts = [example.get('context', '') for example in train_data if example.get('context', '').strip()]
    val_contexts = [example.get('context', '') for example in val_data if example.get('context', '').strip()]
    
    # Generate synthetic Q-A pairs
    train_synthetic_pairs, train_corpus = generate_synthetic_qa_pairs(train_contexts[:50])  # Limit for demo
    val_synthetic_pairs, val_corpus = generate_synthetic_qa_pairs(val_contexts[:20])
    
    # Also include original Q-A pairs
    original_train_pairs = []
    original_val_pairs = []
    
    for example in train_data:
        if example.get('question', '').strip() and example.get('context', '').strip():
            chunk_id = f"original_{example.get('id', len(original_train_pairs))}"
            train_corpus[chunk_id] = example['context']
            original_train_pairs.append({
                'query': normalize_persian(example['question']),
                'corpus_id': chunk_id,
                'score': 1.0
            })
    
    for example in val_data:
        if example.get('question', '').strip() and example.get('context', '').strip():
            chunk_id = f"original_{example.get('id', len(original_val_pairs))}"
            val_corpus[chunk_id] = example['context']
            original_val_pairs.append({
                'query': normalize_persian(example['question']),
                'corpus_id': chunk_id,
                'score': 1.0
            })
    
    # Combine synthetic and original pairs
    all_train_pairs = train_synthetic_pairs + original_train_pairs
    all_val_pairs = val_synthetic_pairs + original_val_pairs
    
    print(f"Generated {len(train_synthetic_pairs)} synthetic train pairs")
    print(f"Added {len(original_train_pairs)} original train pairs")
    print(f"Total training pairs: {len(all_train_pairs)}")
    
    return all_train_pairs, all_val_pairs, train_corpus, val_corpus


force_fine_tune=0

if os.path.exists(FINE_TUNED_EMBEDDING_DIR) and os.listdir(FINE_TUNED_EMBEDDING_DIR) and force_fine_tune!=1:
    print(f"Found existing fine-tuned embedding model at {FINE_TUNED_EMBEDDING_DIR}")
    embedding_model_finetuned= SentenceTransformer(FINE_TUNED_EMBEDDING_DIR, device=DEVICE)
else :
    print("No fine-tuned model found — starting fine-tuning with synthetic data approach")
    if not train_small:
        print("train_small is empty. Skipping fine-tuning.")
        embedding_model_finetuned = SentenceTransformer(EMBEDDING_MODEL_ID, device=DEVICE)
    else:
        # Use BAAI/bge-small-en as base model following Jerry Liu's approach
        base_model_id = "BAAI/bge-small-en"
        print(f"Using base model: {base_model_id}")
        try:
            embedding_model = SentenceTransformer(base_model_id, device=DEVICE)
        except Exception as e:
            print(f"ERROR: Could not load {base_model_id} on CUDA. Reason: {e}")
            print("Aborting. Please check your CUDA version and PyTorch installation.")
            raise SystemExit(1)
        # Create synthetic training dataset
        train_pairs, val_pairs, train_corpus, val_corpus = create_training_dataset_with_synthetic_data(
            train_small, val_small
        )
        if not train_pairs:
            print("No training pairs generated. Using original model.")
            embedding_model_finetuned = embedding_model
        else:
            train_examples = []
            for pair in train_pairs:
                query = pair['query']
                positive_id = pair['corpus_id']
                positive_text = train_corpus.get(positive_id, '')
                if positive_text.strip():
                    train_examples.append(InputExample(texts=[query, positive_text]))
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
            from sentence_transformers import losses
            train_loss = losses.MultipleNegativesRankingLoss(embedding_model)
            evaluator = None
            if val_pairs and val_corpus:
                from sentence_transformers.evaluation import InformationRetrievalEvaluator
                val_queries = {f"q{i}": pair['query'] for i, pair in enumerate(val_pairs)}
                val_relevant_docs = {}
                for i, pair in enumerate(val_pairs):
                    val_relevant_docs[f"q{i}"] = {pair['corpus_id']: 1}
                evaluator = InformationRetrievalEvaluator(
                    val_queries, val_corpus, val_relevant_docs,
                    name="val_ir_evaluator"
                )
            warmup_steps = int(len(train_dataloader) * 0.1)
            print("Starting embedding fine-tuning with Jerry Liu's approach...")
            print(f"- Base model: {base_model_id}")
            print(f"- Training examples: {len(train_examples)}")
            print(f"- Loss function: MultipleNegativesRankingLoss")
            print(f"- Evaluator: InformationRetrievalEvaluator")
            print(f"- Epochs: {EPOCHS}")
            print(f"- Warmup steps: {warmup_steps}")
            os.environ["WANDB_DISABLED"] = "true"
            embedding_model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=EPOCHS,
                warmup_steps=warmup_steps,
                output_path=FINE_TUNED_EMBEDDING_DIR,
                show_progress_bar=True,
                evaluator=evaluator,
                evaluation_steps=50 if evaluator else None,
                save_best_model=True if evaluator else False
            )
            print(f"Fine-tuned embedding model saved to {FINE_TUNED_EMBEDDING_DIR}")
            embedding_model_finetuned = SentenceTransformer(FINE_TUNED_EMBEDDING_DIR, device=DEVICE)

# ==========================
# PDF Processing with Docling
# ==========================
pipeline_options = PdfPipelineOptions()
# pipeline_options.do_ocr = True
# pipeline_options.ocr_options = TesseractOcrOptions(languages=['fas', 'eng'])
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True

doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                
            )
        }
    )

print("Converting PDF with Docling...")
conv_result = doc_converter.convert(PDF_PATH)
dl_doc = conv_result.document

# Optional: Export to Markdown if needed
markdown_path = PDF_PATH.replace('.pdf', '.md')
with open(markdown_path, 'w', encoding='utf-8') as f:
    f.write(dl_doc.export_to_markdown())
print(f"Markdown exported to {markdown_path}")

# ==========================
# Chunking with Docling
# ==========================

def get_hierarchical_chunks(dl_doc):
    chunker = HierarchicalChunker(merge_list_items=HIER_MERGE_LIST_ITEMS)
    chunks = chunker.chunk(dl_doc)
    chunk_texts = [normalize_persian(chunker.contextualize(chunk)) for chunk in chunks if chunk.text.strip()]
    return chunk_texts

def get_hybrid_chunks(dl_doc):
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_ID)
    chunker = HybridChunker(tokenizer=tokenizer, max_chunk_size=MAX_CHUNK_TOKENS, merge_peers=True)
    chunks = list(chunker.chunk(dl_doc)) # Convert generator to list
    chunk_texts = [normalize_persian(chunker.contextualize(chunk)) for chunk in chunks if chunk]
    return chunk_texts



hierarchical_chunks = get_hierarchical_chunks(dl_doc)
hybrid_chunks = get_hybrid_chunks(dl_doc)

print(f"Hierarchical-based chunks (Docling): {len(hierarchical_chunks)}")
print(f"Hybrid-based chunks (Docling): {len(hybrid_chunks)}")

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
    chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=True, device=DEVICE)
    return chunk_embeddings

# Build indexes
print("Building retrieval indexes...")
bm25_hierarchical = build_bm25(hierarchical_chunks)
tfidf_vectorizer_hierarchical, tfidf_matrix_hierarchical = build_tfidf(hierarchical_chunks)
embedding_index_hierarchical = build_embedding_index(hierarchical_chunks, embedding_model_finetuned)

bm25_hybrid = build_bm25(hybrid_chunks)
tfidf_vectorizer_hybrid, tfidf_matrix_hybrid = build_tfidf(hybrid_chunks)
embedding_index_hybrid = build_embedding_index(hybrid_chunks, embedding_model_finetuned)

print("Enhanced RAG pipeline with Docling integration completed!")

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
    return [chunks[i] for i in top_indices], scores[top_indices]

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
    q_emb = embedding_model.encode([query], device=DEVICE)
    sims = cosine_similarity(q_emb, chunk_embeddings).flatten()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in top_idx], sims[top_idx]

def hybrid_retrieve(query, chunks, bm25_index, chunk_embeddings, embedding_model, alpha=ALPHA, top_k=TOP_K):
    if not chunks:
        return [], []
    tokenized_query = word_tokenize(query)
    if bm25_index:
        bm25_scores = bm25_index.get_scores(tokenized_query)
        bm25_scores = bm25_scores / max(bm25_scores) if max(bm25_scores) > 0 else bm25_scores
    else:
        bm25_scores = np.zeros(len(chunks))
    if chunk_embeddings is not None:
        q_emb = embedding_model.encode([query], device=DEVICE)
        emb_scores = cosine_similarity(q_emb, chunk_embeddings).flatten()
    else:
        emb_scores = np.zeros(len(chunks))
    final_scores = alpha * bm25_scores + (1 - alpha) * emb_scores
    top_idx = np.argsort(final_scores)[-top_k:][::-1]
    return [chunks[i] for i in top_idx], final_scores[top_idx]

# ==========================
# QA system
# ==========================

def download_model(model_id, save_path):
    """Download model files and save them locally"""
    print(f"Downloading model {model_id} to {save_path}")
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Download configuration
        config = AutoConfig.from_pretrained(model_id)
        config.save_pretrained(save_path)
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(save_path)
        
        # Download model
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        model.save_pretrained(save_path)
        
        print(f"Model successfully downloaded to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def initialize_qa_pipeline():
    """Initialize QA pipeline with security fixes"""
    try:
        qa_model_id = "HooshvareLab/bert-fa-zwnj-base"
        local_model_path = os.path.expanduser('~/.cache/huggingface/transformers/bert-fa-zwnj-base')
        model_path = FINE_TUNED_QA_DIR if os.path.exists(FINE_TUNED_QA_DIR) else (local_model_path if os.path.exists(local_model_path) else qa_model_id)
        
        # If model not found locally, try to download it
        if not os.path.exists(local_model_path):
            success = download_model(qa_model_id, local_model_path)
            if not success:
                print("Failed to download model. Please ensure you have internet connection or download manually.")

        try:
            if not os.path.exists(local_model_path):
                print("Downloading model files...")
                # Download and save model files
                temp_tokenizer = AutoTokenizer.from_pretrained(qa_model_id, local_files_only=False)
                temp_model = AutoModelForQuestionAnswering.from_pretrained(
                    qa_model_id,
                    local_files_only=False,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
                )
                temp_tokenizer.save_pretrained(local_model_path)
                temp_model.save_pretrained(local_model_path)
                print(f"Model saved to {local_model_path}")
                del temp_tokenizer, temp_model  # Clean up temporary objects

            print(f"Loading model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                local_files_only=True,
                trust_remote_code=True
            )
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                local_files_only=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Model loading failed: {e}")
            # Try alternative loading method
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
            )

        qa = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if DEVICE == "cuda" else -1,
        )
        return qa
    except Exception as e:
        print(f"Error loading QA model: {e}")
        print("Please upgrade PyTorch to version 2.6+ or check model availability")
        return None


def fine_tune_qa_model():
    """Fine-tune QA model with security fix"""
    with mlflow.start_run(run_name="qa_model_fine_tuning"):
        if os.path.exists(FINE_TUNED_QA_DIR) and os.listdir(FINE_TUNED_QA_DIR):
            print(f"Found existing fine-tuned QA model")
            # Log the existing model to MLflow
            mlflow.log_artifact(FINE_TUNED_QA_DIR, "model")
            return

    if not train_small:
        print("No training data for QA fine-tuning")
        return

    qa_model_id = "HooshvareLab/bert-fa-zwnj-base"

    try:
        # Try loading with safetensors first
        tokenizer = AutoTokenizer.from_pretrained(qa_model_id)
        model = AutoModelForQuestionAnswering.from_pretrained(
            qa_model_id,
            # use_safetensors=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
    except Exception as e:
        print(f"Safetensors loading failed: {e}")
        try:
            # Fallback: Try with trust_remote_code (use with caution)
            tokenizer = AutoTokenizer.from_pretrained(qa_model_id, )
            model = AutoModelForQuestionAnswering.from_pretrained(
                qa_model_id,torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
            )
        except Exception as e2:
            print(f"Model loading failed completely: {e2}")
            print("Please upgrade PyTorch to version 2.6+ or use a different model")
            return

    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            if not answer or not answer.get("text") or not answer["text"]:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer["answer_start"][0] if answer.get("answer_start") and len(answer["answer_start"]) > 0 else -1
            answer_text = answer["text"][0] if answer.get("text") and len(answer["text"]) > 0 else ""

            if start_char == -1 or not answer_text.strip():
                start_positions.append(0)
                end_positions.append(0)
                continue

            end_char = start_char + len(answer_text)
            sequence_ids = inputs.sequence_ids(i)
            context_start = sequence_ids.index(1) if 1 in sequence_ids else -1
            context_end = len(sequence_ids) - sequence_ids[::-1].index(1) - 1 if 1 in sequence_ids else -1

            if context_start == -1 or context_end == -1 or offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                try:
                    start_token_idx = next(idx for idx, o in enumerate(offset) if sequence_ids[idx] == 1 and o[0] <= start_char < o[1])
                    end_token_idx = next(idx for idx, o in enumerate(offset) if sequence_ids[idx] == 1 and o[0] < end_char <= o[1])

                    if context_start <= start_token_idx <= context_end and context_start <= end_token_idx <= context_end:
                        start_positions.append(start_token_idx)
                        end_positions.append(end_token_idx)
                    else:
                        start_positions.append(0)
                        end_positions.append(0)
                except (StopIteration, ValueError):
                    start_positions.append(0)
                    end_positions.append(0)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    try:
        tokenized_train = train_small.map(preprocess_function, batched=True, remove_columns=train_small.column_names)
        tokenized_val = val_small.map(preprocess_function, batched=True, remove_columns=val_small.column_names)

        args = TrainingArguments(
            output_dir=FINE_TUNED_QA_DIR,
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,  # Reduced batch size for stability
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            weight_decay=0.01,
            push_to_hub=False,
            save_safetensors=True,  # Save in safetensors format
            dataloader_pin_memory=False,  # Reduce memory pressure
        )

        class MLflowCallback:
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    for k, v in logs.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(k, v)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            callbacks=[MLflowCallback()]
        )

        # Log training parameters
        mlflow.log_params({
            "learning_rate": args.learning_rate,
            "batch_size": args.per_device_train_batch_size,
            "num_epochs": args.num_train_epochs,
            "weight_decay": args.weight_decay,
        })

        # Train the model
        train_result = trainer.train()
        trainer.save_model(FINE_TUNED_QA_DIR)
        
        # Log metrics from training
        metrics = train_result.metrics
        mlflow.log_metrics(metrics)
        
        # Log the model to MLflow
        mlflow.log_artifact(FINE_TUNED_QA_DIR, "model")
        
        print(f"Fine-tuned QA model saved with safetensors format")
        
        # Create and log learning curves
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(trainer.state.log_history)
            plt.title("Training Progress")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.tight_layout()
            
            plot_path = "/tmp/training_progress.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create training visualization: {e}")

    except Exception as e:
        print(f"Training failed: {e}")
        print("Continuing with pre-trained model instead of fine-tuned version")

fine_tune_qa_model()
qa_pipeline = initialize_qa_pipeline()

def answer_question(question, context, max_length=200):
    if not qa_pipeline or not context.strip():
        return "No answer found"
    try:
        if len(context) > max_length:
            context = context[:max_length]
        result = qa_pipeline(question=question, context=context)
        return normalize_persian(result.get('answer', 'No answer found'))
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
        best_answer = "No answer found"
        best_score = -1
        for chunk in retrieved_chunks[:2]:
            answer = answer_question(query, chunk)
            if answer != "No answer found":
                result = qa_pipeline(question=query, context=chunk)
                score = result.get('score', 0.0)
                if score > best_score:
                    best_answer = answer
                    best_score = score
        return best_answer, retrieved_chunks, scores
    else:
        return "No relevant information found", [], []

# ==========================
# Advanced evaluation metrics
# ==========================

def calculate_cosine_similarity(pred_answer, gt_answer, embedding_model):
    if not pred_answer.strip() or not gt_answer.strip():
        return 0.0
    try:
        pred_emb = embedding_model.encode([pred_answer], device=DEVICE)
        gt_emb = embedding_model.encode([gt_answer], device=DEVICE)
        return float(cosine_similarity(pred_emb, gt_emb)[0][0])
    except:
        return 0.0

def calculate_mrr(retrieved_chunks, gt_context, embedding_model, threshold=0.75):
    if not retrieved_chunks or not gt_context or not gt_context.strip():
        return 0.0
    try:
        gt_embedding = embedding_model.encode([gt_context], device=DEVICE)
        for rank, chunk in enumerate(retrieved_chunks, 1):
            if not chunk.strip():
                continue
            chunk_emb = embedding_model.encode([chunk], device=DEVICE)
            sim = cosine_similarity(gt_embedding, chunk_emb)[0][0]
            if sim >= threshold:
                return 1.0 / rank
        return 0.0
    except:
        return 0.0

def evaluate_retrieval_advanced(retrieved_chunks, gt_context, embedding_model, threshold=0.75):
    if not retrieved_chunks or not gt_context or not gt_context.strip():
        return 0, 0.0, 0.0
    try:
        gt_embedding = embedding_model.encode([gt_context], device=DEVICE)
        sims = []
        for chunk in retrieved_chunks:
            if not chunk.strip():
                sims.append(0.0)
                continue
            sim = cosine_similarity(gt_embedding, embedding_model.encode([chunk], device=DEVICE))[0][0]
            sims.append(float(sim))
        relevant = sum(1 for s in sims if s >= threshold)
        hit = 1 if relevant > 0 else 0
        precision = relevant / len(retrieved_chunks) if retrieved_chunks else 0.0
        recall = 1.0 if any(gt_context.strip() in chunk for chunk in retrieved_chunks) else (relevant / len(retrieved_chunks) if retrieved_chunks else 0.0)
        return hit, precision, recall
    except:
        return 0, 0.0, 0.0

# ==========================
# Comprehensive evaluation
# ==========================

def comprehensive_evaluation(dataset, chunks, chunk_type, retrieval_kwargs, max_examples=100):
    # Start a nested run for this evaluation
    with mlflow.start_run(run_name=f"evaluation_{chunk_type}", nested=True):
        print(f"\n{'='*50}")
        print(f"Evaluating {chunk_type}")
        print(f"{'='*50}")
        print(f"Chunks count: {len(chunks) if chunks else 0}")
        
        # Log parameters
        mlflow.log_params({
            "chunk_type": chunk_type,
            "max_examples": max_examples,
            "num_chunks": len(chunks) if chunks else 0,
            "top_k": TOP_K,
            "alpha": ALPHA,
            "device": DEVICE
        })
    if isinstance(dataset, list):
        data_iter = dataset[:max_examples]
    else:
        try:
            data_iter = [dict(example) for example in dataset.select(range(min(max_examples, len(dataset))))]
        except:
            data_iter = list(dataset)[:max_examples] if dataset else []

    if not data_iter:
        print("Warning: dataset empty or failed to iterate")
        return {m: {k: 0.0 for k in ["f1","em","cosine_sim","mrr","hit","precision","recall"]} for m in ["bm25","tfidf","embedding","hybrid"]}

    print("Example data sample (first):")
    print({k: data_iter[0].get(k) for k in ["id","question","answers","context"]})

    squad_metric = load("squad")
    methods = ["bm25", "tfidf", "embedding", "hybrid"]
    results = {}

    for method in methods:
        print(f"\nEvaluating {method} method...")
        method_results = {"f1": [], "em": [], "cosine_sim": [], "mrr": [], "hit": [], "precision": [], "recall": []}

        for example in tqdm(data_iter):
            if not example.get('answers') or not example['answers'].get('text') or not example['answers']['text'][0].strip():
                continue
            query = example['question']
            gt_answer = example['answers']['text'][0]
            gt_context = example['context']

            try:
                pred_answer, retrieved_chunks, scores = rag_pipeline(
                    query, chunks, retrieval_method=method, **retrieval_kwargs
                )
                qa_metric = squad_metric.compute(
                    predictions=[{"id": str(example['id']), "prediction_text": normalize_persian(pred_answer)}],
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
                print(f"Error example {example.get('id','NA')}: {e}")
                continue

        aggregated = {}
        for metric, values in method_results.items():
            metric_value = statistics.mean(values) if values else 0.0
            aggregated[metric] = metric_value
            # Log metrics to MLflow with method prefix
            mlflow.log_metric(f"{method}_{metric}", metric_value)
        results[method] = aggregated

        print(f"\n{method} Results:")
        print(f"F1: {results[method]['f1']:.4f}")
        print(f"EM: {results[method]['em']:.4f}")
        print(f"Cosine Similarity: {results[method]['cosine_sim']:.4f}")
        print(f"MRR: {results[method]['mrr']:.4f}")
        print(f"Hit@{TOP_K}: {results[method]['hit']:.4f}")
        print(f"Precision: {results[method]['precision']:.4f}")
        print(f"Recall: {results[method]['recall']:.4f}")
        
        # Log confusion matrix or other visualizations if available
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create a bar plot of metrics
            plt.figure(figsize=(10, 6))
            metrics_values = list(results[method].values())
            metrics_names = list(results[method].keys())
            plt.bar(metrics_names, metrics_values)
            plt.title(f"{method} Method - Metrics Overview")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot to temp file and log to MLflow
            plot_path = f"/tmp/{method}_metrics.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")

    return results

# ==========================
# Run evaluation
# ==========================
print("Starting comprehensive system evaluation...")

hierarchical_retrieval_kwargs = {
    'bm25_index': bm25_hierarchical,
    'vectorizer': tfidf_vectorizer_hierarchical,
    'tfidf_matrix': tfidf_matrix_hierarchical,
    'chunk_embeddings': embedding_index_hierarchical,
    'embedding_model': embedding_model_finetuned,
}
hierarchical_results = comprehensive_evaluation(
    rag_eval_dataset_mapped, hierarchical_chunks, "Hierarchical-based Chunking (Docling)", hierarchical_retrieval_kwargs
)

hybrid_retrieval_kwargs = {
    'bm25_index': bm25_hybrid,
    'vectorizer': tfidf_vectorizer_hybrid,
    'tfidf_matrix': tfidf_matrix_hybrid,
    'chunk_embeddings': embedding_index_hybrid,
    'embedding_model': embedding_model_finetuned,
}
hybrid_results = comprehensive_evaluation(
    rag_eval_dataset_mapped, hybrid_chunks, "Hybrid-based Chunking (Docling)", hybrid_retrieval_kwargs
)

hierarchical_results = hierarchical_results or {}
hybrid_results = hybrid_results or {}

# ==========================
# Final comparison and summary
# ==========================
print(f"\n{'='*80}")
print("Results Summary and Comparison")
print(f"{'='*80}")

def print_comparison_table(results_map):
    methods = ["bm25", "tfidf", "embedding", "hybrid"]
    print(f"\n{'Method':<12} {'Chunk Type':<20} {'F1':<8} {'EM':<8} {'CosSim':<8} {'MRR':<8} {'Hit@K':<8} {'Prec':<8} {'Rec':<8}")
    print("-" * 100)
    for method in methods:
        for chunk_label, res in results_map.items():
            r = res[method]
            print(f"{method:<12} {chunk_label:<20} {r['f1']:<8.4f} {r['em']:<8.4f} {r['cosine_sim']:<8.4f} {r['mrr']:<8.4f} {r['hit']:<8.4f} {r['precision']:<8.4f} {r['recall']:<8.4f}")
        print("-" * 100)

results_map = {
    'Hierarchical-Docling': hierarchical_results,
    'Hybrid-Docling': hybrid_results,
}
print_comparison_table(results_map)

# ==========================
# Find best method
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

def save_results(hierarchical_results, hybrid_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results_summary = {
        "hierarchical_based_results": hierarchical_results,
        "hybrid_based_results": hybrid_results,
        "best_method": best_method,
        "best_score": best_score,
        "evaluation_settings": {
            "top_k": TOP_K,
            "alpha_hybrid": ALPHA,
            "embedding_model": EMBEDDING_MODEL_ID,
            "qa_model": "HooshvareLab/bert-fa-zwnj-base",
            "fine_tuned": True,
            "max_chunk_tokens": MAX_CHUNK_TOKENS,
            "hier_merge_list_items": HIER_MERGE_LIST_ITEMS,
        }
    }
    results_file = os.path.join(output_dir, "task_b_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {results_file}")

save_results(hierarchical_results, hybrid_results, OUTPUT_DIR)

# ==========================
# Interactive QA demo
# ==========================

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List, Tuple

app = FastAPI(title="Persian QA System")

class QuestionRequest(BaseModel):
    question: str

class RelevantText(BaseModel):
    text: str
    score: float

class QAResponse(BaseModel):
    answer: str
    num_chunks: int
    relevant_texts: List[RelevantText]

def get_qa_response(question: str) -> QAResponse:
    method, chunk_label = best_method.split('__')
    if 'Hierarchical' in chunk_label:
        chunks = hierarchical_chunks; kwargs = hierarchical_retrieval_kwargs
    elif 'Hybrid' in chunk_label:
        chunks = hybrid_chunks; kwargs = hybrid_retrieval_kwargs

    answer, retrieved_chunks, scores = rag_pipeline(question, chunks, retrieval_method=method, **kwargs)
    
    relevant_texts = [
        RelevantText(text=chunk[:200], score=score) 
        for chunk, score in zip(retrieved_chunks[:3], scores[:3])
    ]
    
    return QAResponse(
        answer=answer,
        num_chunks=len(retrieved_chunks),
        relevant_texts=relevant_texts
    )

@app.post("/qa", response_model=QAResponse)
async def qa_endpoint(request: QuestionRequest):
    return get_qa_response(request.question)

@app.get("/")
async def root():
    return {"message": "Persian QA System API. Send POST requests to /qa with your questions."}

if __name__ == "__main__":
    print(f"Using method: {best_method.split('__')[0]} with {best_method.split('__')[1]} chunking")
    uvicorn.run(app, host="0.0.0.0", port=8000)

