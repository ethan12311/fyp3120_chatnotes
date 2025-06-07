# ===== bot.py =====
import PyPDF2, numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load AI model (auto-downloads first time)
print("Loading AI brain...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. Process Lecture 2 PDF
def load_lecture2():
    print("Digesting lecture notes...")
    reader = PyPDF2.PdfReader("lecture2.pdf")
    notes = []
    for page_num in range(len(reader.pages)):
        text = reader.pages[page_num].extract_text()
        notes.append({
            "text": text,
            "page": page_num+1,
            "vector": model.encode(text)  # AI converts text to math vector
        })
    print(f"Lecture 2 ready! {len(notes)} pages indexed")
    return notes

# 3. Q&A Function
def ask_bot(question, lecture_data):
    q_vector = model.encode(question)
    best_page = None
    highest_score = -1
    
    for page in lecture_data:
        # Calculate similarity score (cosine similarity)
        score = np.dot(q_vector, page["vector"]) 
        if score > highest_score:
            highest_score = score
            best_page = page
            
    if best_page:
        return f" Answer from page {best_page['page']}:\n{best_page['text'][:500]}..."
    return "Sorry bro, not found in Lecture 1 "

# 4. Main Loop
lecture1 = load_lecture2()
print("\n" + "="*50)
print("CN BOT ACTIVE! Ask about Lecture 1 (type 'quit' to exit)")
print("="*50)

while True:
    question = input("\nYou: ")
    if question.lower() == "quit":
        break
    answer = ask_bot(question, lecture1)
    print("\nBot:", answer)