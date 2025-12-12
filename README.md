# 🛍️ ShopGenie: AI Shopping Assistant

**ShopGenie** is an intelligent E-commerce recommendation system powered by **RAG (Retrieval-Augmented Generation)**. It uses vector search to understand user queries and retrieves the best-matching products from an inventory dataset, then uses a Large Language Model (DeepSeek) to provide personalized explanations.

## 🚀 Features

* **Natural Language Search:** Ask for products as if you were talking to a human (e.g., *"I need running shoes that are good for trails"*).
* **RAG Architecture:** Combines vector search (FAISS) with a powerful LLM (DeepSeek) for accurate, context-aware answers.
* **Smart Inventory Search:** Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) to find semantically similar products.
* **Source Transparency:** "View Source Products" expander shows exactly which database items were used to generate the answer.
* **Interactive UI:** Built with **Streamlit** for a clean, chat-based experience.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM Framework:** LangChain
* **Model Provider:** HuggingFace Hub (DeepSeek-V3.2)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** `all-MiniLM-L6-v2`
* **Data Handling:** Pandas

---

## ⚙️ Setup & Installation (Local)

Follow these steps to run the project locally on your machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/SyedFarzanHussain/RAG-Product-Recommender-Using-LangChain-FAISS-.git](https://github.com/SyedFarzanHussain/RAG-Product-Recommender-Using-LangChain-FAISS-.git)
cd RAG-Product-Recommender-Using-LangChain-FAISS-
````

### 2\. Install Dependencies

You can install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

*(Or if you use `uv`, run `uv sync`)*

### 3\. Set Up Environment Variables

This project requires a HuggingFace API token to run the LLM.

1.  Create a file named `.env` in the root directory.
2.  Add your HuggingFace token inside it:
    ```env
    HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here
    ```

### 4\. Prepare the Data

Ensure you have your product inventory file named `products_dataset.csv` in the main folder.

  * *Note: The CSV must contain columns for `product_id`, `title`, and `description`.*

### 5\. Run the Application

Start the Streamlit server:

```bash
streamlit run main.py
```

-----

## ☁️ Deployment Guide (Streamlit Cloud)

You can deploy this app to the web for free using Streamlit Community Cloud.

### 1\. Push to GitHub

Ensure your latest code is pushed to GitHub (do not upload your `.env` file).

### 2\. Connect to Streamlit

1.  Go to [share.streamlit.io](https://share.streamlit.io/) and sign up with GitHub.
2.  Click **"New app"** and select **"Use existing repo"**.
3.  Choose this repository (`SyedFarzanHussain/RAG-Product-Recommender...`).
4.  Set **Main file path** to `main.py`.

### 3\. Add Secrets (Crucial Step)

Before clicking "Deploy":

1.  Click **"Advanced Settings"**.
2.  Paste your API key into the "Secrets" field in TOML format:
    ```toml
    HUGGINGFACEHUB_API_TOKEN = "your_actual_token_here"
    ```
3.  Click **Save** and then **Deploy**.

-----

## 📂 Project Structure

```text
├── main.py                # Main application logic (Streamlit + LangChain)
├── products_dataset.csv   # The inventory data
├── requirements.txt       # Python dependencies
├── .env                   # API Keys (Not uploaded to GitHub)
└── README.md              # Project documentation
```

## 🧠 How It Works

1.  **Ingestion:** The app loads `products_dataset.csv` and converts product descriptions into vector embeddings.
2.  **Storage:** These embeddings are stored in a local FAISS vector store.
3.  **Retrieval:** When you ask a question, the system searches FAISS for the 4 most relevant products.
4.  **Generation:** These products are sent to the **DeepSeek-V3.2** model, which crafts a helpful response explaining *why* these products fit your needs.

-----

*Created by Syed Farzan Hussain*

```
```
