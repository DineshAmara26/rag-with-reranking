# RAG with Reranking

A Retrieval-Augmented Generation (RAG) application that answers questions from PDF documents using semantic search, vector retrieval, Cross-Encoder reranking, and Google Gemini.

## Project Overview

This project implements a two-stage retrieval pipeline.

First, relevant document chunks are retrieved using vector similarity with FAISS. Then, the retrieved chunks are reranked using a Cross-Encoder to improve the relevance of the final context. Google Gemini is used to generate the final answer from the retrieved context.

## Architecture

```text
PDF Document
     ↓
OCR Text Extraction
     ↓
Text Chunking
     ↓
Sentence Transformer Embeddings
     ↓
FAISS Vector Database
     ↓
Initial Retrieval
     ↓
Cross-Encoder Reranking
     ↓
Gemini
     ↓
Final Answer
