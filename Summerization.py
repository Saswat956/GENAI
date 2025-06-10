# Core Libraries
import json
import re
import unicodedata
import numpy as np
from typing import List
from tqdm import tqdm
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# AWS & SageMaker
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.jumpstart.model import JumpStartModel

# Vector Store
import faiss

# Step 0: Setup roles and endpoints
role = get_execution_role()
client = boto3.client("sagemaker-runtime")

# Step 1: Clean text function
def clean_text(text: str) -> str:
    text = ' '.join(text.split())
    text = re.sub(r'[^\w\s.,;:!?()\'\"-]', '', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return text

# Step 2: Load and clean PDF
PATH=""
loader = PyPDFLoader(PATH)
documents = loader.load()
cleaned_texts = [clean_text(doc.page_content) for doc in documents]
full_text = "\n".join(cleaned_texts)

# Step 3: Chunk text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ".", ","]
)
chunks = text_splitter.split_text(full_text)
chunked_documents = [Document(page_content=chunk) for chunk in chunks]

# Step 4: Deploy MiniLM embedding model
embedding_model = HuggingFaceModel(
    env={"HF_MODEL_ID": "sentence-transformers/all-MiniLM-L6-v2", "HF_TASK": "feature-extraction"},
    role=role,
    transformers_version="4.6",
    pytorch_version="1.7",
    py_version="py36"
)
encoder = embedding_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name="minilm-summary-encoder"
)

# Step 5: Batch embeddings
def batch_predict(encoder, texts: List[str], batch_size: int = 8):
    all_outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            out = encoder.predict({"inputs": batch})
            if not out or not isinstance(out, list):
                print(f"[Warning] Empty or unexpected output at batch {i}")
                continue
            if isinstance(out[0][0], list):  # Token-level
                batch_embeddings = [np.mean(t, axis=0).tolist() for t in out]
            else:
                batch_embeddings = out
            all_outputs.extend(batch_embeddings)
        except Exception as e:
            print(f"[Error] batch {i}: {e}")
    return all_outputs

texts = [doc.page_content for doc in chunked_documents]
embeddings = batch_predict(encoder, texts, batch_size=4)
array = np.array(embeddings, dtype="float32")

# Step 6: Optional clustering with FAISS (like previous summarization flow)
dimension = array.shape[1]
num_clusters = 30
kmeans = faiss.Kmeans(d=dimension, k=num_clusters, niter=20, verbose=True)
kmeans.train(array)
centroids = kmeans.centroids

index = faiss.IndexFlatL2(dimension)
index.add(array)
_, closest_indices = index.search(centroids, 1)
sorted_indices = np.sort(closest_indices.flatten())
extracted_docs = [chunked_documents[i] for i in sorted_indices]

# Step 7: Deploy LLaMA 3.1 model (JumpStart)
llm_model = JumpStartModel(model_id="")
generator = llm_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.4xlarge",
    endpoint_name="llama-summary-8b",
    accept_eula=True
)

# Step 8: Summary prompt template
SUMMARY_PROMPT = """You will be given different passages from a book one by one. Provide a summary of the following text. Your result must be detailed and at least 2 paragraphs. When summarizing, directly dive into the narrative or descriptions from the text without using introductory phrases like 'In this passage'. Focus on the main events, characters, and themes to present a unified view of the content.

Passage:

{text}

SUMMARY:"""

# Step 9: Generate summaries
final_summary = ""
for doc in tqdm(extracted_docs, desc="Summarizing"):
    prompt = SUMMARY_PROMPT.replace("{text}", doc.page_content)
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 700,
            "top_p": 0.9,
            "temperature": 0.6
        }
    }
    response = client.invoke_endpoint(
        EndpointName="llama-summary-8b",
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    result = json.loads(response["Body"].read())
    summary = result[0]["generated_text"] if isinstance(result, list) else result
    final_summary += summary + "\n\n"

# Step 10: Save to PDF
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 15)
        self.cell(80)
        self.cell(30, 10, "Summary", 1, 0, "C")
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

pdf = PDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
final_summary_utf8 = final_summary.encode("latin-1", "replace").decode("latin-1")
pdf.multi_cell(0, 10, final_summary_utf8)
pdf.output("final_summary.pdf")
----------------------------------------------------------------------------------------------------------------------------------------------


SUMMARY_PROMPT = """
You will be provided with a section of a document. This section may contain information under specific headings or thematic blocks.

Your task is to:
1. Identify any headings or implicit topic shifts in the passage.
2. For each heading or block, summarize the key points in a structured and detailed manner.
3. Use bullet points or short paragraphs for each topic to maintain clarity.
4. If the section involves legal, financial, or contractual language, explain it in simple terms without changing the meaning.
5. Avoid introductory filler like "This passage discusses". Just begin with the extracted insights.

Document Section:

{text}

--- Start your chain-of-thought summary below ---
"""
