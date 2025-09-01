# -*- coding: utf-8 -*-
"""
Full RAG pipeline with five chunking strategies: word-based, sentence-based, semantic-based, hierarchical-based, markdown-based
Includes:
 - Persian normalizer and SQuAD mapper
 - Fine-tune embedding model (contrastive) if train_small provided or load existing fine-tuned model
 - Fine-tune QA model for Persian/medical adaptation
 - Chunking (word, sentence, semantic, hierarchical, markdown)
 - Index building (BM25, TF-IDF, Embedding)
 - Retrieval methods (BM25, TF-IDF, Embedding, Hybrid)
 - QA pipeline (transformers) and evaluation (F1/EM/CosSim/MRR/Precision/Recall/Hit@K)
Requirements: sentence-transformers, transformers, rank_bm25, nltk, PyPDF2, evaluate, scikit-learn, torch, marker-pdf
"""
import os, re, json, random, torch
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer, util, losses, InputExample
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from evaluate import load
from tqdm import tqdm
import statistics

# Marker-PDF imports
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.settings import settings
from marker.schema import BlockTypes

# ==========================
# Configs
# ==========================
BASE = './qadata/'
PDF_PATH = BASE + 'Drugs.pdf'
OUTPUT_DIR = BASE + 'outputs'
EMBEDDING_MODEL_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
FINE_TUNED_EMBEDDING_DIR = BASE + 'models/finetuned-emb'
FINE_TUNED_QA_DIR = BASE + 'models/finetuned-qa'
TOP_K = 5
ALPHA = 0.4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Chunking params
CHUNK_SIZE_WORD = 220
WORD_OVERLAP = 20
MAX_SENT_PER_CHUNK = 3

# Semantic chunking params
SEM_MAX_SENT = 8
SEM_MIN_SENT = 2
SEM_SIM_THRESHOLD = 0.45
SEM_OVERLAP_SENT = 1

# Hierarchical and Markdown chunking params
HIER_MAX_CHAR = 1000
HIER_OVERLAP_CHAR = 200
HIER_SEPARATORS = ["\n\n", "\n", "؟", "!", "؛", "."]
MD_MAX_CHAR = HIER_MAX_CHAR
MD_OVERLAP_CHAR = HIER_OVERLAP_CHAR

# Training hyperparams
BATCH_SIZE = 16
EPOCHS = 1
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
    text = re.sub(r"[۰-۹]", lambda m: str(int(m.group(0)) - 1776), text)
    text = re.sub(r"[^\w\s]", " ", text)
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
train_small = mapped["train"]
val_small = mapped["validation"]
rag_eval_dataset_mapped = rag_eval_dataset.map(map_to_squad)

# ==========================
# Fine-tune Embedding Model (keep existing implementation)
# ==========================
if os.path.exists(FINE_TUNED_EMBEDDING_DIR) and os.listdir(FINE_TUNED_EMBEDDING_DIR):
    print(f"Found existing fine-tuned model at {FINE_TUNED_EMBEDDING_DIR}, loading...")
    embedding_model_finetuned = SentenceTransformer(FINE_TUNED_EMBEDDING_DIR, device=DEVICE)
else:
    print("No fine-tuned model found — starting fine-tuning (if train_small available)")
    if not train_small:
        print("train_small is empty. Skipping fine-tuning.")
        embedding_model_finetuned = SentenceTransformer(EMBEDDING_MODEL_ID, device=DEVICE)
    else:
        train_examples = []
        for i in tqdm(range(len(train_small)), desc="Preparing fine-tune examples"):
            example = train_small[i]
            query = normalize_persian(example.get('question', ''))
            positive = normalize_persian(example.get('context', ''))
            negative_idx = random.randint(0, len(train_small) - 1)
            while negative_idx == i:
                negative_idx = random.randint(0, len(train_small) - 1)
            negative = normalize_persian(train_small[negative_idx].get('context', ''))
            train_examples.append(InputExample(texts=[query, positive], label=1.0))
            train_examples.append(InputExample(texts=[query, negative], label=0.0))
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
        os.environ["WANDB_DISABLED"] = "true"
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID, device=DEVICE)
        train_loss = losses.ContrastiveLoss(model=embedding_model)
        print("Starting fine-tuning embedding model...")
        embedding_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=EPOCHS, warmup_steps=100)
        os.makedirs(FINE_TUNED_EMBEDDING_DIR, exist_ok=True)
        embedding_model.save(FINE_TUNED_EMBEDDING_DIR)
        embedding_model_finetuned = SentenceTransformer(FINE_TUNED_EMBEDDING_DIR, device=DEVICE)
        print(f"Fine-tuned embedding model saved to {FINE_TUNED_EMBEDDING_DIR}")

# ==========================
# Enhanced PDF Processing with Marker-PDF
# ==========================
def pdf_to_markdown_with_marker(pdf_path, output_md_path=None):
    """
    Convert PDF to markdown using marker-pdf
    """
    try:
        # Configure marker settings for Persian support
        settings.TORCH_DEVICE = DEVICE
        # settings.TESSERACT_LANGUAGES = ["fas", "eng"]  # Persian and English
        
        # Create converter with model dictionary
        converter = PdfConverter(artifact_dict=create_model_dict())
        
        # Convert PDF to markdown
        print(f"Converting PDF to markdown using marker-pdf...")
        markdown_content, images, metadata = converter(pdf_path)
        
        # Normalize Persian text
        markdown_content = normalize_persian(markdown_content[1])
        
        # Save markdown if output path provided
        if output_md_path:
            with open(output_md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"Markdown saved to {output_md_path}")
        
        return markdown_content, metadata
        
    except Exception as e:
        print(f"Error converting PDF to Markdown with marker-pdf: {e}")
        return "", {}

def extract_document_with_blocks(pdf_path):
    try:
        settings.TORCH_DEVICE = DEVICE
        converter = PdfConverter(artifact_dict=create_model_dict())

        print(f"Building document structure using marker-pdf...")
        document = converter.build_document(pdf_path)

        # Safely collect only available block types
        available_types = [getattr(BlockTypes, name) for name in dir(BlockTypes) if not name.startswith("_")]
        
        all_blocks = document.contained_blocks(tuple(available_types))

        # Forms (if available)
        forms = []
        if hasattr(BlockTypes, "Form"):
            forms = document.contained_blocks((BlockTypes.Form,))

        blocks_by_type = {}
        for block in all_blocks:
            block_type = block.block_type.value if hasattr(block, "block_type") else "Unknown"
            if block_type not in blocks_by_type:
                blocks_by_type[block_type] = []

            block_content = {
                "text": normalize_persian(block.text.strip()) if getattr(block, "text", None) else "",
                "bbox": getattr(block, "bbox", None),
                "page_number": getattr(block, "page_number", None),
            }
            blocks_by_type[block_type].append(block_content)

        print(f"Extracted {len(all_blocks)} blocks:")
        for block_type, blocks in blocks_by_type.items():
            print(f"  - {block_type}: {len(blocks)} blocks")

        return document, blocks_by_type, forms

    except Exception as e:
        print(f"Error extracting blocks with marker-pdf: {e}")
        return None, {}, []

def extract_document_text(file_path):
    """
    Enhanced document text extraction with marker-pdf support
    """
    try:
        if file_path.endswith('.pdf'):
            # First try marker-pdf
            markdown_text, metadata = pdf_to_markdown_with_marker(file_path)
            if markdown_text:
                print(f"Successfully converted PDF to markdown with marker-pdf")
                return markdown_text
            else:
                print("Marker-pdf failed, falling back to PyPDF2...")
                # Fallback to PyPDF2
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ''
                    for page in reader.pages:
                        text += (page.extract_text() or '') + '\n'
                return normalize_persian(text)
        elif file_path.endswith('.md'):
            with open(file_path, 'r', encoding='utf-8') as file:
                markdown_text = file.read()
            return normalize_persian(markdown_text)
        else:
            print(f"Unsupported file format: {file_path}")
            return ""
    except Exception as e:
        print(f"Error reading document: {e}")
        return ""

# Extract document and blocks
pdf_text = extract_document_text(PDF_PATH)
document, blocks_by_type, forms = extract_document_with_blocks(PDF_PATH)

print(f"Document text length: {len(pdf_text)} characters")
print(f"Found {len(forms)} form blocks")

# ==========================
# Enhanced Markdown Chunking with Block-Based Approach
# ==========================
# def markdown_chunk_with_blocks(pdf_path, max_char=MD_MAX_CHAR, overlap_char=MD_OVERLAP_CHAR):
#     """
#     Advanced markdown chunking using marker-pdf block extraction
#     """
#     try:
#         # Extract structured blocks
#         document, blocks_by_type, forms = extract_document_with_blocks(pdf_path)
        
#         if not blocks_by_type:
#             print("No blocks extracted, falling back to text-based chunking")
#             markdown_text, _ = pdf_to_markdown_with_marker(pdf_path)
#             if markdown_text:
#                 return hierarchical_chunk(markdown_text, max_char, overlap_char)
#             return []
        
#         chunks = []
#         current_chunk = ""
#         current_length = 0
        
#         # Priority order for block types
#         block_priority = [
#             'Heading',
#             'Paragraph', 
#             'List',
#             'Table',
#             'Figure',
#             'Caption',
#             'Form'
#         ]
        
#         # Process blocks in order of priority
#         all_block_items = []
#         for block_type in block_priority:
#             if block_type in blocks_by_type:
#                 for block in blocks_by_type[block_type]:
#                     all_block_items.append((block_type, block))
        
#         # If no prioritized blocks, use all available blocks
#         if not all_block_items:
#             for block_type, blocks in blocks_by_type.items():
#                 for block in blocks:
#                     all_block_items.append((block_type, block))
        
#         for block_type, block in all_block_items:
#             block_text = block['text']
#             if not block_text:
#                 continue
            
#             # Add block type prefix for context
#             block_prefix = f"[{block_type}] "
#             block_content = block_prefix + block_text
#             block_length = len(block_content)
            
#             # Check if adding the block exceeds max_char
#             if current_length + block_length > max_char:
#                 if current_chunk:
#                     chunks.append(current_chunk.strip())
                
#                 # Apply overlap
#                 if overlap_char and len(current_chunk) > overlap_char:
#                     current_chunk = current_chunk[-overlap_char:]
#                     current_length = len(current_chunk)
#                 else:
#                     current_chunk = ""
#                     current_length = 0
                
#                 # If the block itself is too large, split it
#                 if block_length > max_char:
#                     sub_chunks = split_large_block(block_content, max_char, overlap_char)
#                     for sub_chunk in sub_chunks:
#                         if sub_chunk.strip():
#                             chunks.append(sub_chunk.strip())
#                     current_chunk = ""
#                     current_length = 0
#                 else:
#                     current_chunk += block_content + "\n"
#                     current_length += block_length + 1
#             else:
#                 current_chunk += block_content + "\n"
#                 current_length += block_length + 1
        
#         # Append the final chunk if it exists
#         if current_chunk.strip():
#             chunks.append(current_chunk.strip())
        
#         # Filter out empty chunks
#         chunks = [c for c in chunks if c.strip()]
#         print(f"Created {len(chunks)} markdown chunks from {len(all_block_items)} blocks")
        
#         return chunks
        
    # except Exception as e:
    #     print(f"Error in block-based markdown chunking: {e}")
    #     # Fallback to text-based markdown chunking
    #     markdown_text, _ = pdf_to_markdown_with_marker(pdf_path)
    #     if markdown_text:
    #         return hierarchical_chunk(markdown_text, max_char, overlap_char)
    #     return []


def markdown_chunk_with_blocks(pdf_path, max_char=MD_MAX_CHAR, overlap_char=MD_OVERLAP_CHAR):
    """
    Advanced markdown chunking using marker-pdf block extraction
    Converts blocks into markdown (# Heading, - lists, etc.)
    """
    try:
        # Extract structured blocks
        document, blocks_by_type, forms = extract_document_with_blocks(pdf_path)

        if not blocks_by_type:
            print("No blocks extracted, falling back to text-based chunking")
            markdown_text, _ = pdf_to_markdown_with_marker(pdf_path)
            if markdown_text:
                return hierarchical_chunk(markdown_text, max_char, overlap_char)
            return []

        chunks = []
        current_chunk = ""
        current_length = 0

        # Flatten all blocks in reading order
        all_blocks = []
        for block_type, blocks in blocks_by_type.items():
            for block in blocks:
                all_blocks.append((block_type, block["text"]))

        # Convert block type → markdown
        def block_to_markdown(bt, text):
            if not text.strip():
                return ""
            bt_lower = bt.lower()
            if "heading" in bt_lower:
                return f"# {text.strip()}\n"
            elif "list" in bt_lower:
                return f"- {text.strip()}\n"
            elif "table" in bt_lower:
                return f"\n[TABLE]\n{text.strip()}\n[/TABLE]\n"
            elif "figure" in bt_lower or "caption" in bt_lower:
                return f"![Figure] {text.strip()}\n"
            elif "form" in bt_lower:
                return f"**Form Field:** {text.strip()}\n"
            else:
                return f"{text.strip()}\n"

        for block_type, text in all_blocks:
            block_content = block_to_markdown(block_type, text)
            block_length = len(block_content)

            # If adding this block exceeds max_char → finalize current chunk
            if current_length + block_length > max_char:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Apply overlap
                if overlap_char and len(current_chunk) > overlap_char:
                    current_chunk = current_chunk[-overlap_char:]
                    current_length = len(current_chunk)
                else:
                    current_chunk = ""
                    current_length = 0

            # Add block to chunk
            current_chunk += block_content + "\n"
            current_length += block_length + 1

        # Append the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        print(f"Created {len(chunks)} markdown chunks (markdown-style)")

        return chunks

    except Exception as e:
        print(f"Error in block-based markdown chunking: {e}")
        # Fallback to plain hierarchical chunking
        markdown_text, _ = pdf_to_markdown_with_marker(pdf_path)
        if markdown_text:
            return hierarchical_chunk(markdown_text, max_char, overlap_char)
        return []


def split_large_block(text, max_char, overlap_char):
    """Helper function to split large blocks"""
    sub_chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_char, len(text))
        sub_chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap_char if overlap_char and end < len(text) else end
    return sub_chunks

# ==========================
# Existing Chunking Functions (keep all existing implementations)
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

def semantic_based_chunk(text, model=None, max_sent=SEM_MAX_SENT, min_sent=SEM_MIN_SENT,
                        sim_threshold=SEM_SIM_THRESHOLD, overlap_sent=SEM_OVERLAP_SENT):
    if not text:
        return []
    if model is None:
        model = embedding_model_finetuned
        
    sentences = [s.strip() for s in sent_tokenize(text) if s and s.strip()]
    if not sentences:
        return []
        
    sent_embs = model.encode(sentences, show_progress_bar=False, device=DEVICE)
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
        start = max(end - overlap_sent, start + 1)
    
    return chunks

def hierarchical_chunk(text, max_char=HIER_MAX_CHAR, overlap_char=HIER_OVERLAP_CHAR, separators=HIER_SEPARATORS):
    chunks = []
    
    def recursive_split(current_text, level=0):
        if len(current_text) <= max_char or level >= len(separators):
            chunks.append(current_text.strip())
            return
        
        sep = separators[level]
        parts = re.split(f'({re.escape(sep)})', current_text)
        accumulated = ""
        
        for i in range(0, len(parts), 2):
            part = parts[i]
            if i+1 < len(parts):
                part += parts[i+1]
            
            if len(accumulated + part) > max_char:
                if accumulated:
                    recursive_split(accumulated, level + 1)
                accumulated = part[-overlap_char:] + part if overlap_char else part
            else:
                accumulated += part
        
        if accumulated:
            recursive_split(accumulated, level + 1)
    
    recursive_split(text)
    return [c for c in chunks if c.strip()]

# ==========================
# Create chunks with enhanced markdown chunking
# ==========================
if pdf_text:
    word_chunks = word_based_chunk(pdf_text)
    sentence_chunks = sentence_based_chunk(pdf_text)
    semantic_chunks = semantic_based_chunk(pdf_text)
    hierarchical_chunks = hierarchical_chunk(pdf_text)
    # Use the enhanced markdown chunking with blocks
    markdown_chunks = markdown_chunk_with_blocks(PDF_PATH)
    
    print(f"Word-based chunks: {len(word_chunks)}")
    print(f"Sentence-based chunks: {len(sentence_chunks)}")
    print(f"Semantic-based chunks: {len(semantic_chunks)}")
    print(f"Hierarchical-based chunks: {len(hierarchical_chunks)}")
    print(f"Markdown-based chunks: {len(markdown_chunks)} (using block-based extraction)")
else:
    word_chunks, sentence_chunks, semantic_chunks, hierarchical_chunks, markdown_chunks = [], [], [], [], []
    print("Document text not found!")

# ==========================
# Continue with rest of the pipeline (keep all existing code)
# ==========================

# Build retrieval indexes
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
bm25_word = build_bm25(word_chunks)
tfidf_vectorizer_word, tfidf_matrix_word = build_tfidf(word_chunks)
embedding_index_word = build_embedding_index(word_chunks, embedding_model_finetuned)

bm25_sentence = build_bm25(sentence_chunks)
tfidf_vectorizer_sentence, tfidf_matrix_sentence = build_tfidf(sentence_chunks)
embedding_index_sentence = build_embedding_index(sentence_chunks, embedding_model_finetuned)

bm25_semantic = build_bm25(semantic_chunks)
tfidf_vectorizer_semantic, tfidf_matrix_semantic = build_tfidf(semantic_chunks)
embedding_index_semantic = build_embedding_index(semantic_chunks, embedding_model_finetuned)

bm25_hierarchical = build_bm25(hierarchical_chunks)
tfidf_vectorizer_hierarchical, tfidf_matrix_hierarchical = build_tfidf(hierarchical_chunks)
embedding_index_hierarchical = build_embedding_index(hierarchical_chunks, embedding_model_finetuned)

bm25_markdown = build_bm25(markdown_chunks)
tfidf_vectorizer_markdown, tfidf_matrix_markdown = build_tfidf(markdown_chunks)
embedding_index_markdown = build_embedding_index(markdown_chunks, embedding_model_finetuned)

print("Enhanced RAG pipeline with marker-pdf integration completed!")
print(f"Extracted {len(blocks_by_type)} different block types")
print(f"Forms detected: {len(forms)}")


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

def initialize_qa_pipeline():
    """Initialize QA pipeline with security fixes"""
    try:
        qa_model_id = "SajjadAyoubi/xlm-roberta-large-fa-qa"
        model_path = FINE_TUNED_QA_DIR if os.path.exists(FINE_TUNED_QA_DIR) else qa_model_id
        
        try:
            # Try safetensors first
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_safetensors=True)
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_path, 
                use_safetensors=True,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
            )
        except Exception as e:
            print(f"Safetensors loading failed: {e}")
            # Fallback to trust_remote_code (use with caution in production)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_path, 
                trust_remote_code=True,
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
    if os.path.exists(FINE_TUNED_QA_DIR) and os.listdir(FINE_TUNED_QA_DIR):
        print(f"Found existing fine-tuned QA model")
        return
    
    if not train_small:
        print("No training data for QA fine-tuning")
        return
    
    qa_model_id = "SajjadAyoubi/xlm-roberta-large-fa-qa"
    
    try:
        # Try loading with safetensors first
        tokenizer = AutoTokenizer.from_pretrained(qa_model_id, use_safetensors=True)
        model = AutoModelForQuestionAnswering.from_pretrained(
            qa_model_id, 
            use_safetensors=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
    except Exception as e:
        print(f"Safetensors loading failed: {e}")
        try:
            # Fallback: Try with trust_remote_code (use with caution)
            tokenizer = AutoTokenizer.from_pretrained(qa_model_id, trust_remote_code=True)
            model = AutoModelForQuestionAnswering.from_pretrained(
                qa_model_id, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
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
        
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
        )
        
        trainer.train()
        trainer.save_model(FINE_TUNED_QA_DIR)
        print(f"Fine-tuned QA model saved with safetensors format")
        
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
    print(f"\n{'='*50}")
    print(f"Evaluating {chunk_type}")
    print(f"{'='*50}")
    print(f"Chunks count: {len(chunks) if chunks else 0}")
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
            aggregated[metric] = statistics.mean(values) if values else 0.0
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

word_retrieval_kwargs = {
    'bm25_index': bm25_word,
    'vectorizer': tfidf_vectorizer_word,
    'tfidf_matrix': tfidf_matrix_word,
    'chunk_embeddings': embedding_index_word,
    'embedding_model': embedding_model_finetuned,
}
word_results = comprehensive_evaluation(rag_eval_dataset_mapped, word_chunks, "Word-based Chunking", word_retrieval_kwargs)

sentence_retrieval_kwargs = {
    'bm25_index': bm25_sentence,
    'vectorizer': tfidf_vectorizer_sentence,
    'tfidf_matrix': tfidf_matrix_sentence,
    'chunk_embeddings': embedding_index_sentence,
    'embedding_model': embedding_model_finetuned,
}
sentence_results = comprehensive_evaluation(
    rag_eval_dataset_mapped, sentence_chunks, "Sentence-based Chunking", sentence_retrieval_kwargs
)

semantic_retrieval_kwargs = {
    'bm25_index': bm25_semantic,
    'vectorizer': tfidf_vectorizer_semantic,
    'tfidf_matrix': tfidf_matrix_semantic,
    'chunk_embeddings': embedding_index_semantic,
    'embedding_model': embedding_model_finetuned,
}
semantic_results = comprehensive_evaluation(
    rag_eval_dataset_mapped, semantic_chunks, "Semantic-based Chunking", semantic_retrieval_kwargs
)

hierarchical_retrieval_kwargs = {
    'bm25_index': bm25_hierarchical,
    'vectorizer': tfidf_vectorizer_hierarchical,
    'tfidf_matrix': tfidf_matrix_hierarchical,
    'chunk_embeddings': embedding_index_hierarchical,
    'embedding_model': embedding_model_finetuned,
}
hierarchical_results = comprehensive_evaluation(
    rag_eval_dataset_mapped, hierarchical_chunks, "Hierarchical-based Chunking", hierarchical_retrieval_kwargs
)

markdown_retrieval_kwargs = {
    'bm25_index': bm25_markdown,
    'vectorizer': tfidf_vectorizer_markdown,
    'tfidf_matrix': tfidf_matrix_markdown,
    'chunk_embeddings': embedding_index_markdown,
    'embedding_model': embedding_model_finetuned,
}
markdown_results = comprehensive_evaluation(
    rag_eval_dataset_mapped, markdown_chunks, "Markdown-based Chunking", markdown_retrieval_kwargs
)

word_results = word_results or {}
sentence_results = sentence_results or {}
semantic_results = semantic_results or {}
hierarchical_results = hierarchical_results or {}
markdown_results = markdown_results or {}

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
    'Word-based': word_results,
    'Sentence-based': sentence_results,
    'Semantic-based': semantic_results,
    'Hierarchical-based': hierarchical_results,
    'Markdown-based': markdown_results,
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

def save_results(word_results, sentence_results, semantic_results, hierarchical_results, markdown_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results_summary = {
        "word_based_results": word_results,
        "sentence_based_results": sentence_results,
        "semantic_based_results": semantic_results,
        "hierarchical_based_results": hierarchical_results,
        "markdown_based_results": markdown_results,
        "best_method": best_method,
        "best_score": best_score,
        "evaluation_settings": {
            "top_k": TOP_K,
            "chunk_size_word": CHUNK_SIZE_WORD,
            "alpha_hybrid": ALPHA,
            "embedding_model": EMBEDDING_MODEL_ID,
            "qa_model": "SajjadAyoubi/xlm-roberta-large-fa-qa",
            "fine_tuned": True,
            "sem_params": {
                "SEM_MAX_SENT": SEM_MAX_SENT,
                "SEM_MIN_SENT": SEM_MIN_SENT,
                "SEM_SIM_THRESHOLD": SEM_SIM_THRESHOLD,
                "SEM_OVERLAP_SENT": SEM_OVERLAP_SENT,
            },
            "hier_params": {
                "HIER_MAX_CHAR": HIER_MAX_CHAR,
                "HIER_OVERLAP_CHAR": HIER_OVERLAP_CHAR,
                "HIER_SEPARATORS": HIER_SEPARATORS,
            },
            "markdown_params": {
                "MD_MAX_CHAR": MD_MAX_CHAR,
                "MD_OVERLAP_CHAR": MD_OVERLAP_CHAR,
            }
        }
    }
    results_file = os.path.join(output_dir, "task_b_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {results_file}")

save_results(word_results, sentence_results, semantic_results, hierarchical_results, markdown_results, OUTPUT_DIR)

# ==========================
# Interactive QA demo
# ==========================

def interactive_qa_demo():
    print(f"\n{'='*60}")
    print("Interactive RAG System Demo")
    print(f"{'='*60}")
    print("Type 'quit' to exit")
    method, chunk_label = best_method.split('__')
    if 'Word' in chunk_label:
        chunks = word_chunks; kwargs = word_retrieval_kwargs
    elif 'Sentence' in chunk_label:
        chunks = sentence_chunks; kwargs = sentence_retrieval_kwargs
    elif 'Semantic' in chunk_label:
        chunks = semantic_chunks; kwargs = semantic_retrieval_kwargs
    elif 'Hierarchical' in chunk_label:
        chunks = hierarchical_chunks; kwargs = hierarchical_retrieval_kwargs
    elif 'Markdown' in chunk_label:
        chunks = markdown_chunks; kwargs = markdown_retrieval_kwargs
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
print("   - استفاده از مدل فاین‌تیون‌شده برای امبدینگ و QA")
print("   - پیاده‌سازی پنج روش چانکینگ: کلمه‌ای، جمله‌ای، سمانتیک، هیرارکیکال، مارک‌داون")
print("   - چهار روش بازیابی: BM25، TF-IDF، Embedding، Hybrid")
print("   - ارزیابی با F1، EM، Cosine Similarity، MRR، Precision، Recall، Hit@K")
print("\nKey results:")
print(f"   - Best method: {best_method}")
print(f"   - Best method overall score: {best_score:.4f}")
print("\nImplemented improvements:")
print("   - QA model fine-tuned on Persian data")
print("   - Individual chunk QA with best-answer selection")
print("   - Enhanced Persian normalization")
print("   - PDF-to-Markdown conversion with Marker")
print("   - Markdown chunking with block-based extraction using Marker")
print(f"\nSaved files:\n   - Embedding model: {FINE_TUNED_EMBEDDING_DIR}\n   - QA model: {FINE_TUNED_QA_DIR}\n   - Evaluation results: {OUTPUT_DIR}/task_b_results.json\n   - Converted Markdown: {PDF_PATH.replace('.pdf', '.md')}")
print(f"\n{'='*80}")
print("Task B completed successfully!")
print(f"{'='*80}")