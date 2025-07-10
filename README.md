CrediTrust Complaint Analysis Chatbot
Project Overview
This project develops an intelligent complaint-answering chatbot for CrediTrust Financial, leveraging Retrieval-Augmented Generation (RAG) to extract actionable insights from unstructured customer complaint narratives. The goal is to empower Product Managers, Customer Support, and Compliance teams to quickly understand complaint trends and obtain evidence-backed answers, shifting from a reactive to a proactive problem-solving approach.

Business Objective
CrediTrust Financial aims to transform its raw customer complaint data into a strategic asset. This AI tool is designed to:

Reduce the time required to identify major complaint trends from days to minutes.

Enable non-technical teams to get direct, evidence-backed answers about complaints.

Foster a proactive culture for identifying and resolving customer pain points.

Features
Data Preprocessing: Cleans and filters raw customer complaint narratives, focusing on relevant financial products.

Text Chunking & Embedding: Breaks down cleaned narratives into manageable chunks and converts them into numerical vector embeddings using a Sentence Transformer model.

Vector Store Indexing: Stores embeddings in a FAISS vector database for efficient similarity search.

Retrieval-Augmented Generation (RAG): Combines semantic search (retrieval) with a Large Language Model (generation) to provide grounded answers.

Interactive Chat Interface: A user-friendly web interface built with Gradio for easy interaction with the chatbot, including displaying sources for transparency.


Project Structure
credi_trust_chatbot/
├── data/
│   ├── complaints.csv                  # Raw dataset (downloaded)
│   ├── filtered_complaints.csv         # Cleaned and filtered dataset (Output of Task 1)
│   └── (eda_plots).png                 # EDA visualization outputs (from Task 1)
│   └── (bot_avatar.png)                # Optional: Chatbot avatar image
├── src/
│   ├── __init__.py                     # Makes src a Python package
│   ├── eda_preprocessing.py            # Script for Task 1 (EDA & Preprocessing)
│   ├── vector_store_builder.py         # Script for Task 2 (Chunking, Embedding, Indexing)
│   └── rag_pipeline.py                 # Script for Task 3 (RAG Core Logic & Evaluation)
├── vector_store/
│   └── faiss_complaints_index/         # Directory for persisted FAISS index (Output of Task 2)
│       ├── index.faiss
│       └── index.pkl
├── app.py                              # Gradio web application (Task 4)
├── requirements.txt                    # List of Python dependencies
├── README.md                           # Project overview, setup, and usage instructions (This file)
└── report.md                           # Detailed project report

Setup Instructions
Follow these steps to set up and run the project on your local machine.

1. Prerequisites
Python 3.8+

pip (Python package installer)

2. Clone the Repository (Hypothetical)
If this were a Git repository, you would clone it:

git clone <repository_url>
cd credi_trust_chatbot

Since you're building this step-by-step, ensure your local directory structure matches the "Project Structure" section above.

3. Install Dependencies
Navigate to the project's root directory (credi_trust_chatbot/) and install the required Python packages:

pip install -r requirements.txt

requirements.txt content:

pandas
matplotlib
seaborn
scikit-learn
nltk
langchain
langchain-community
faiss-cpu # Or faiss-gpu if you have a compatible GPU
sentence-transformers
transformers
torch # Or tensorflow (pytorch is generally easier for sentence-transformers/HuggingFace)
accelerate # For faster inference with HuggingFace models
bitsandbytes # For 8-bit quantization with HuggingFace models (optional, for smaller models)
gradio # For the interactive UI

4. Download the Dataset
Download the Consumer Complaint Database from the CFPB website and place the complaints.csv file into the data/ directory.

Go to: https://www.consumerfinance.gov/data-research/consumer-complaints/

Look for a "Download the Data" link (usually a large CSV file).

Place the downloaded file as credi_trust_chatbot/data/complaints.csv.

How to Run the Project (Step-by-Step)
Execute each task sequentially from the project's root directory (credi_trust_chatbot/).

Task 1: Data Preprocessing and EDA
This step cleans and filters the raw complaint data.

python src/eda_preprocessing.py

Expected Output:

Console logs detailing data loading, EDA, filtering, and cleaning.

New files in data/: filtered_complaints.csv, product_distribution_initial.png, narrative_length_distribution.png, product_distribution_filtered.png.

Task 2: Text Chunking, Embedding, and Vector Store Indexing
This step transforms cleaned narratives into searchable vector embeddings and builds the FAISS index.

python src/vector_store_builder.py

Expected Output:

Console logs detailing chunking, embedding model loading, and FAISS index creation/saving.

A new directory vector_store/faiss_complaints_index/ containing index.faiss and index.pkl.

Note: The first run will download the embedding model, which may take some time.

Task 3: RAG Core Logic and Qualitative Evaluation
This step tests the core retrieval and generation capabilities of your RAG pipeline.

python src/rag_pipeline.py

Expected Output:

Console logs showing the loading of the embedding model, FAISS index, and the Large Language Model (LLM).

For each pre-defined evaluation question, the console will display the AI's generated answer and the retrieved source complaint chunks.

Note: The first run will download the LLM (google/gemma-2b-it), which is a large download and will take significant time and disk space.

Task 4: Launch the Interactive Chatbot Interface
This step starts the Gradio web application, allowing you to interact with your RAG chatbot via a web browser.

python app.py

Expected Output:

Console logs indicating the loading of all RAG components.

A local URL (e.g., http://127.0.0.1:7860) will be printed to the console.

Open this URL in your web browser to access the chatbot interface.

Usage
Once the Gradio application is running (Task 4):

Open your web browser to the provided local URL.

Type your question about customer complaints (e.g., "What are the common problems with personal loan applications?").

Click "Send" or press Enter.

The chatbot will display its answer, followed by the specific complaint excerpts (sources) it used to formulate the response, including product, issue, company, and complaint ID.

Use the "Clear Chat" button to reset the conversation.

Future Enhancements
Quantitative Evaluation: Implement metrics for retrieval accuracy, faithfulness, and answer relevance.

Advanced UI Features: Add filtering options (by product, company, date range) directly in the UI.

Multi-turn Conversation: Enhance the chatbot to maintain context across multiple user interactions.

Deployment: Deploy the application to a cloud platform for broader accessibility.

LLM Optimization: Explore fine-tuning smaller LLMs or using more advanced retrieval techniques for improved performance and cost efficiency.
