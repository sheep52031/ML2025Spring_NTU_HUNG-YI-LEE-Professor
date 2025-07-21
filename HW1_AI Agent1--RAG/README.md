# HW1 AI Agent with RAG - Retrieval Augmented Generation System

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system using multiple AI agents to answer questions by combining web search capabilities with Large Language Model (LLM) inference. The system uses LLaMA 3.1 8B model and processes 90 questions from public and private datasets.

## Technical Stack

- **Model**: LLaMA 3.1 8B Instruct (Quantized Q8_0)
- **Runtime Environment**: Linux RTX 3080 GPU
- **Framework**: llama-cpp-python, LangChain, ChromaDB
- **Embedding Model**: paraphrase-multilingual-MiniLM-L12-v2
- **Search**: Google Search API with web scraping

## System Architecture

### Multi-Agent Design

The system employs three specialized agents working in coordination:

1. **Question Extraction Agent**
   - **Role**: Professional question analyst
   - **Function**: Extracts core questions from complex descriptions
   - **Output**: Simplified, clear question statements

2. **Keyword Extraction Agent**
   - **Role**: Professional keyword extraction expert
   - **Function**: Identifies 2-5 optimal search keywords from questions
   - **Output**: Comma-separated keyword list

3. **QA Agent**
   - **Role**: Knowledge integrator and answer generator
   - **Function**: Generates answers based on retrieved context
   - **Output**: Traditional Chinese responses based on provided data

### RAG Pipeline Implementation

#### Stage 1: Question Understanding & Preprocessing
```
User Question → Question Extraction Agent → Core Question → Keyword Extraction Agent → Search Keywords
```

#### Stage 2: Information Retrieval
```
Keywords → Google Search → Web Content → Text Cleaning → Document Chunking
```

#### Stage 3: Vector Processing & Similarity Search
```
Document Chunks → Embedding Vectorization → ChromaDB Storage → Similarity Search → Top-K Relevant Docs
```

#### Stage 4: Content Summarization & Answer Generation
```
Relevant Docs → Summarization → Context Assembly → QA Agent → Final Answer
```

## Key Features

### Advanced RAG Techniques
- **Multi-stage processing**: Question understanding, retrieval, summarization, and generation
- **Semantic search**: Uses multilingual sentence transformers for Chinese text understanding
- **Dynamic information retrieval**: Real-time web search for up-to-date information
- **Context management**: Document chunking and summarization to handle token limits

### Robust Implementation
- **Asynchronous processing**: Parallel web scraping for improved efficiency
- **Checkpoint mechanism**: Resume processing from interruption points
- **Error handling**: Graceful handling of HTTP 429 errors and network issues
- **Multi-format output**: Individual text files, combined text file, and CSV format

## Performance Analysis

### Current Issues Identified

1. **Search Quality Problems**
   - **Issue**: Keywords too broad, resulting in irrelevant search results
   - **Example**: Question about "虎山雄風飛揚" school song returns irrelevant content
   - **Impact**: Low answer accuracy

2. **Document Processing Limitations**
   - **Issue**: Fixed 500-character chunking may break semantic integrity
   - **Impact**: Loss of contextual information

3. **Answer Generation Challenges**
   - **Issue**: QA Agent often provides incorrect or off-topic answers
   - **Example**: Rugby scoring question (answer should be "5") returns running technique information
   - **Root Cause**: Poor retrieval quality leads to irrelevant context

### Speed Performance Issues

The system exhibits slow performance due to:
- **Multiple LLM calls**: Each question requires 6+ model inferences (extraction, keywords, summarization, QA)
- **Sequential processing**: Synchronous execution of RAG pipeline stages
- **Web scraping overhead**: Network latency for multiple web page retrievals
- **Vector operations**: Embedding computation and similarity search operations

## Results & Accuracy Analysis

### Output Files
- `20250707.csv`: Complete Q&A results in CSV format
- `20250707_*.txt`: Individual answer files for debugging
- `20250707.txt`: Combined answers file

### Accuracy Assessment
Comparison between system outputs and ground truth reveals significant accuracy issues:

**Sample Accuracy Problems**:
- Question 1: "虎山雄風飛揚" school song → System couldn't find answer (should be "光華國小")
- Question 5: Rugby Union try scoring → System answered about running technique (should be "5 points")
- Question 6: 卑南族 ancestor origin → System answered "Panama" (should be "臺東縣太麻里鄉")

**Root Causes**:
1. **Keyword extraction ineffectiveness**: Generated keywords don't match specific query requirements
2. **Search result irrelevance**: Google search returns unrelated content
3. **Context filtering failure**: Similarity search doesn't identify truly relevant information
4. **Answer generation drift**: QA agent provides information not contained in context

## Improvement Recommendations

### 1. Enhanced Keyword Strategy
- Implement Named Entity Recognition (NER)
- Adjust search strategies based on question types
- Add synonym expansion capabilities

### 2. Document Processing Optimization
- Use semantic-based chunking instead of fixed-length splitting
- Preserve structural information (headers, paragraphs)
- Implement overlapping sliding windows

### 3. Quality Control Enhancement
- Add retrieval relevance scoring
- Implement answer confidence assessment
- Design multi-round retrieval strategies for failed searches

### 4. Performance Optimization
- Implement caching mechanisms for repeated searches
- Parallel processing of multiple chunks
- Use lightweight embedding models

## Installation & Usage

### Prerequisites
```bash
# Install required packages
pip install llama-cpp-python==0.3.4 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
pip install googlesearch-python bs4 charset-normalizer requests-html lxml_html_clean
pip install sentence-transformers chromadb langchain langchain-community
```

### Model Download
```bash
# Download LLaMA 3.1 8B quantized model (~8GB)
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
```

### Dataset Download
```bash
# Download question datasets
wget https://www.csie.ntu.edu.tw/~ulin/public.txt
wget https://www.csie.ntu.edu.tw/~ulin/private.txt
```

### Execution
1. Ensure GPU runtime is enabled
2. Run all cells in the Jupyter notebook sequentially
3. Monitor progress through individual answer file generation
4. Download final results in ZIP format

## Limitations & Considerations

### Technical Limitations
- **Google Search Rate Limits**: HTTP 429 errors may occur with frequent searches
- **Model Context Window**: 16K token limit affects processing of long documents
- **GPU Memory**: Requires 16GB+ VRAM for optimal performance

### Design Trade-offs
- **Accuracy vs Speed**: Multiple agent calls improve quality but reduce speed
- **Context vs Efficiency**: Detailed summarization improves context but increases processing time
- **Robustness vs Complexity**: Error handling adds complexity but improves reliability

## Future Enhancements

1. **Advanced RAG Techniques**
   - Implement HyDE (Hypothetical Document Embeddings)
   - Add re-ranking mechanisms
   - Use graph-based knowledge retrieval

2. **Model Improvements**
   - Fine-tune embedding models for domain-specific content
   - Implement model ensembling
   - Add hallucination detection

3. **System Optimization**
   - Implement distributed processing
   - Add real-time monitoring
   - Create adaptive retry mechanisms

## Contributors

- **Student ID**: 20250707
- **Course**: ML2025 Spring NTU (Prof. Hung-Yi Lee)
- **Assignment**: Homework 1 - AI Agent with RAG

## License

This project is for educational purposes as part of the ML2025 course curriculum.