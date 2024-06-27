import fitz  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_pdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text


def split_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def retrieve_documents(query, documents, top_k=3):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    
    return [documents[idx] for idx in top_k_indices]

def generate_response(retrieved_docs, query):
    response = f"Based on your query '{query}', here is some information:\n"
    for i, doc in enumerate(retrieved_docs):
        response += f"{i+1}. {doc}\n"
    response += "\nThis information is provided based on the most relevant documents found."
    return response

def rag_system(query, documents):
    retrieved_docs = retrieve_documents(query, documents)
    response = generate_response(retrieved_docs, query)
    return response

if __name__ == "__main__":
    file_path = "C:\\Users\\soham\\interview\\rag\\data\\meidtations.pdf"
    text = read_pdf(file_path)
    documents = split_text(text)
    
    query = "Tell me about Marcus Aurelius Antoninus Augustus."
    
    response = rag_system(query, documents)
    print(response)
