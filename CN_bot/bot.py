# ===== bot.py =====
# ===== 升級版 bot.py =====
import re
import numpy as np
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 配置 HKUST OpenAI API
openai.api_key = "YOUR_HKUST_OPENAI_KEY"
openai.api_base = "https://hkust.azure-api.net/openai"

# 2. 增強 PDF 處理 (支援公式/代碼)
def extract_text_with_context(pdf_path):
    reader = PdfReader(pdf_path)
    clean_pages = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        
        # 強化清理邏輯
        text = re.sub(r'\n{3,}', '\n\n', text)  # 合併多餘換行
        text = re.sub(r' {2,}', ' ', text)       # 刪除多餘空格
        text = re.sub(r'-\n', '', text)           # 處理斷字
        
        # 保留結構標記
        text = f"# PAGE {i+1} #\n{text}"
        clean_pages.append(text)
    
    return "\n\n".join(clean_pages)

# 3. 智能文本分塊
def chunk_text(text, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    return splitter.split_text(text)

# 4. 載入模型 + 生成語義向量
print("🦾 Loading HKUST-enhanced AI model...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

def encode_with_context(texts):
    return model.encode(texts, show_progress_bar=True)

# 5. 核心問答引擎
class LectureBot:
    def __init__(self, pdf_path):
        self.raw_text = extract_text_with_context(pdf_path)
        self.chunks = chunk_text(self.raw_text)
        print(f"📚 Indexed {len(self.chunks)} knowledge chunks")
        self.chunk_vectors = encode_with_context(self.chunks)
        
    def _find_relevant_chunks(self, question, top_k=3):
        q_vector = model.encode([question])
        sim_scores = cosine_similarity(q_vector, self.chunk_vectors)[0]
        top_indices = np.argsort(sim_scores)[-top_k:][::-1]
        return [(self.chunks[i], sim_scores[i]) for i in top_indices]
    
    def _generate_answer(self, question, context_chunks):
        context_str = "\n\n---\n\n".join([f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        # 調用 HKUST OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": f"""
                你是我校課程助教，請嚴格根據提供的講義上下文回答問題。
                回答需符合教授教學風格，包含:
                - 精確概念解釋
                - 相關公式/代碼示例 (如適用)
                - 易混淆點提醒
                若問題超出講義範圍，請說明並建議參考資料
                """},
                {"role": "user", "content": f"""
                [講義上下文]:
                {context_str}
                
                [學生問題]:
                {question}
                """}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message['content'].strip()
    
    def ask(self, question, confidence_threshold=0.65):
        relevant_chunks = self._find_relevant_chunks(question)
        
        # 置信度過濾
        if relevant_chunks[0][1] < confidence_threshold:
            return "please ask Prof Meng or the TA about the query"
            
        pure_chunks = [chunk for chunk, score in relevant_chunks]
        return self._generate_answer(question, pure_chunks)

# 6. 主程序
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎓 HKUST LectureGPT Activated - COMP2011 Edition")
    print("="*50)
    
    bot = LectureBot("lecture2.pdf")
    
    while True:
        try:
            question = input("\n🎤 Student: ")
            if question.lower() in ['exit', 'quit']: break
            answer = bot.ask(question)
            print(f"\n🤖 Bot: {answer}")
        except KeyboardInterrupt:
            print("\nSession ended")
            break
