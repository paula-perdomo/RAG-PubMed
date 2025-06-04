# PubMed RAG-MCQA: Retrieval-Augmented Generation for Multiple-Choice Question Answering

This project implements a Retrieval-Augmented Generation (RAG) system designed to answer multiple-choice questions based on a corpus of PubMed abstracts. The system retrieves relevant documents using contextual embeddings and then uses a fine-tuned Large Language Model (LLM) to select the correct answer ID from a given set of options.

## Project Objective

The primary goal is to build an end-to-end system that, given a question and a set of four textual options, can accurately predict the ID of the correct option. The system leverages information retrieval techniques combined with the generative capabilities of a fine-tuned LLM, all grounded in a provided corpus of 500,000 PubMed abstracts.

## System Architecture

The system follows a RAG (Retrieval-Augmented Generation) approach, specifically tailored for a multiple-choice question answering (MCQA) format. It consists of two main components:

1.  **Retriever (Encoder-based):**
    * **Encoder Model:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` is used via the `sentence-transformers` library to generate dense vector embeddings for the document corpus and incoming questions.
    * **Document Corpus:** `pubmed_500K.json`, containing 500,000 PubMed abstracts. The `'contents'` field (concatenation of title and abstract) is used for embedding.
    * **Vector Store:** FAISS (Facebook AI Similarity Search) with an `IndexFlatL2` is used for efficient similarity search. The index and document embeddings are stored locally.
    * **Retrieval Logic:** For a given question, the retriever encodes it and searches the FAISS index to find the top 5 most relevant document excerpts from `pubmed_500K.json`. From these 5, the single document with the smallest distance (highest relevance) is selected as the context for the decoder.

2.  **Generator (Decoder-based):**
    * **LLM Model:** `meta-llama/Llama-3.2-1B` is used as the base model for the decoder.
    * **Fine-tuning:** The Llama 1B model is fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation).
        * **Task:** The fine-tuning task is a multiple-choice question answering format. Given a context, a question, and four text options (labeled 0-3), the model is trained to output the ID of the correct option.
        * **Training Data:** Derived from `pubmed_QA_train.json`. Each entry in this file (containing `id`, `excerpt`, `question`, `statement` (correct answer), and `distractors`) is processed. The `excerpt` serves as context. The `statement` and three `distractors` form the four options, with the `statement`'s position randomized. The model is trained on prompts that include this context, question, and options, to predict the correct option's ID.
    * **Inference:** The fine-tuned Llama model receives the question, the four options provided in the test data, and the single best context retrieved by the encoder. It then predicts the ID of the correct option by evaluating the logits for the valid choice tokens ("0", "1", "2", "3").

## Datasets Used

* **`pubmed_500K.json`**: A corpus of 500,000 PubMed abstracts. Used for building the FAISS index for the retriever component.
* **`pubmed_QA_train.json`**: Contains questions, correct answer statements, distractors, and context excerpts. Used to create the training dataset for fine-tuning the Llama decoder.
* **`pubmed_QA_eval.json`**: Similar structure to `pubmed_QA_train.json`. Used to create the evaluation dataset for assessing the fine-tuned Llama decoder during development.
* **`pubmed_QA_test_questions.json`**: Contains questions and four multiple-choice options per question. This is the final test set used to evaluate the end-to-end RAG system.

## Key Technologies & Libraries

* **Python 3**
* **Hugging Face Libraries:**
    * `transformers`: For loading and using the PubMedBERT encoder and Llama 1B decoder.
    * `datasets`: For handling and processing datasets.
    * `peft`: For Parameter-Efficient Fine-Tuning (LoRA) of the Llama model.
    * `trl`: For Supervised Fine-tuning (SFTTrainer) of the Llama model.
    * `accelerate`: For efficient model training and inference.
    * `bitsandbytes`: For model quantization (e.g., QLoRA with 4-bit precision), not explicitly shown in the final RAG system but used in decoder fine-tuning notebook.
* **`sentence-transformers`**: For easily using the PubMedBERT model as a sentence encoder.
* **`faiss-gpu` (or `faiss-cpu`)**: For creating the vector index and performing similarity searches.
* **`torch`**: The deep learning framework backend.
* **`numpy`**: For numerical operations, especially with embeddings.
* **`pandas`**: For data manipulation and saving results.
* **`json`, `random`, `re`**: Standard Python libraries for data handling and text processing.

## Notebook Descriptions

The project is structured across three main Jupyter notebooks:

1.  **`encoder_g12.ipynb`**:
    * Loads the `pubmed_500K.json` document corpus.
    * Initializes the `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` model using `sentence-transformers`.
    * Generates embeddings for the `'contents'` field of each document.
    * Builds a FAISS `IndexFlatL2` using these embeddings.
    * Saves the FAISS index, document embeddings (`.npy`), and corresponding document IDs (`.json`) for later use by the retriever. Paths for saved files include timestamps (e.g., `pubmed_faiss_pubmedbert_20250516_020433.index`).

2.  **`decoder_g12.ipynb`**:
    * Loads the `pubmed_QA_train.json` and `pubmed_QA_eval.json` datasets.
    * Processes these datasets using a custom function (`procesar_pubmed_qa`) to create a multiple-choice question answering format: `(id, context, question, options, id_answer)`. The `context` is derived from the `excerpt` field, and the correct answer (`statement`) is randomly placed among the `distractors` to form the 4 `options`.
    * Loads the base `meta-llama/Llama-3.2-1B` model and its tokenizer.
    * Applies PEFT (LoRA) configuration for efficient fine-tuning.
    * Uses `SFTTrainer` from the `trl` library to fine-tune the Llama model on the processed training data. The training prompt includes context, question, options, and the target completion is the ID of the correct answer.
    * Saves the trained LoRA adapter and tokenizer to `/content/drive/MyDrive/UniAndes/MAIA-202411/3. modelos-avanzados-para-el-procesamiento-de-lenguaje-natural/W7/maia-pln-2025/decoder/final_adapter`.
    * Includes an evaluation function that assesses accuracy by comparing predicted option IDs (derived from model logits for choice tokens) against reference IDs.

3.  **`information_retrieval_system_g12.ipynb` (Main RAG Pipeline)**:
    * **Loads Encoder Components**:
        * The pre-trained `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` sentence encoder.
        * The pre-built FAISS index (`pubmed_faiss_pubmedbert_20250516_020433.index`).
        * The original document texts from `pubmed_500K.json` to map FAISS results back to actual content.
    * **Loads Fine-tuned Decoder Components**:
        * The base `meta-llama/Llama-3.2-1B` model.
        * The fine-tuned LoRA adapter from `/content/drive/MyDrive/UniAndes/MAIA-202411/3. modelos-avanzados-para-el-procesamiento-de-lenguaje-natural/W7/maia-pln-2025/decoder/final_adapter`.
        * The corresponding tokenizer.
    * **Implements the RAG Workflow**:
        * Reads questions and options from `pubmed_QA_test_questions.json`.
        * For each question:
            * The `retrieve_relevant_documents` function (modified to search top 5 and return the best one) retrieves the single most relevant context from `pubmed_500K.json`.
            * The `get_predicted_option_id` function formats a prompt containing this retrieved context, the original question, and the four provided options.
            * The fine-tuned Llama model predicts the ID of the correct option using a constrained approach (selecting the most probable token among "0", "1", "2", "3").
    * **Outputs Results**: Saves the predictions (Question ID, Predicted Answer ID) to `rag_test_predictions.json` and `rag_test_predictions.csv`.

## Setup and Execution

1.  **Prerequisites:**
    * Python 3.x
    * `pip` for installing packages
    * Access to a machine with a GPU (CUDA enabled) is highly recommended, especially for fine-tuning and efficient inference with Llama models. The project was developed using NVIDIA L4 and A100 GPUs on Google Colab.

2.  **Installation:**
    Install the required Python libraries. You can typically do this by running the first cell in each notebook, which contains `pip install` commands for:
    ```bash
    pip install transformers torch accelerate bitsandbytes peft trl datasets sentence_transformers faiss-gpu # or faiss-cpu
    pip install pandas numpy
    ```
    (Note: `faiss-gpu-cu11==1.10.0` was specified in `encoder_g12.ipynb` and `information_retrieval_system_g12.ipynb`).

3.  **Hugging Face Login:**
    For downloading and using gated models like Llama 3, you need to be logged into your Hugging Face account.
    ```python
    from huggingface_hub import login
    # login(token="YOUR_HF_ACCESS_TOKEN")
    ```
    The notebooks use `userdata.get('HF_TOKEN')` to retrieve the token, assuming it's set as a secret in the Colab environment.

4.  **Data Paths:**
    The notebooks expect data files and saved model/index components to be in specific Google Drive paths (e.g., `/content/drive/MyDrive/UniAndes/MAIA-202411/...`). You will need to:
    * Upload the provided JSON datasets (`pubmed_500K.json`, `pubmed_QA_train.json`, `pubmed_QA_eval.json`, `pubmed_QA_test_questions.json`) to your Google Drive or local environment.
    * Adjust the file paths in the notebooks accordingly.

5.  **Order of Execution:**
    1.  **`encoder_g12.ipynb`**: Run this first to generate document embeddings and the FAISS index from `pubmed_500K.json`. The outputs (index file, embeddings, doc IDs) are essential for the RAG system.
    2.  **`decoder_g12.ipynb`**: Run this notebook to fine-tune the `meta-llama/Llama-3.2-1B` model using the processed data from `pubmed_QA_train.json` and `pubmed_QA_eval.json`. This will save the LoRA adapter.
    3.  **`information_retrieval_system_g12.ipynb`**: Run this last. It loads the components created by the previous two notebooks and processes the `pubmed_QA_test_questions.json` to generate final predictions.

## Output

The main output of the system, generated by `information_retrieval_system_g12.ipynb`, is a JSON file (`rag_test_predictions.json`) and a CSV file (`rag_test_predictions.csv`). These files contain the predictions for each question in the test set, with each entry typically including:
* `ID` (or `question_id`): The ID of the test question.
* `answer` (or `predicted_answer_id`): The predicted ID (0-3) of the correct option.

Additional details like the question text, options, and a sample of retrieved context are also included in the JSON output for easier inspection.