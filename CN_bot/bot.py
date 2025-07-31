# @title 📚 Simple Lecture Assistant (Click ▶️ to run)
import os
from getpass import getpass
from IPython.display import display, Markdown
import openai

# 1. Upload your PDFs
print("1️⃣ Upload your lecture PDFs:")
print("- Click the 📁 folder icon on left")
print("- Click [Upload] and select your PDF files")
print("- Wait until files appear in the file list")
input("Press Enter AFTER uploading files...")

# 2. Enter your Azure OpenAI key
print("\n2️⃣ Enter your Azure OpenAI credentials:")
print("Note: Your key should be a long string (no 'sk-' prefix)")
azure_api_key = getpass("Paste your Azure API key: ")
azure_endpoint = "https://hkust.azure-api.net"  # Your school's endpoint

# 3. Select PDF file
pdf_files = [f for f in os.listdir() if f.lower().endswith(".pdf")]
if not pdf_files:
    print("❌ No PDF files found! Please upload files first")
else:
    print("\n📚 Available lecture notes:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"{i}. {pdf}")
    
    try:
        selection = int(input("\n👉 Enter number of PDF to use: ")) - 1
        selected_pdf = pdf_files[selection]
        
        # 4. Ask question
        print("\n3️⃣ Ask your question about the lecture:")
        question = input("What would you like to know? ")
        
        # 5. Get answer
        print("\n🧠 Thinking...")
        client = openai.AzureOpenAI(
            api_key=azure_api_key,
            api_version="2023-05-15",
            azure_endpoint=azure_endpoint
        )
        
        response = client.chat.completions.create(
            model="gpt-35-turbo",  # Try "gpt-4" if this doesn't work
            messages=[
                {"role": "system", "content": f"You are a helpful teaching assistant. Answer questions based solely on the content of the lecture PDF: {selected_pdf}"},
                {"role": "user", "content": question}
            ],
            temperature=0.3
        )
        
        # 6. Show answer
        display(Markdown(f"### 📝 Answer:\n{response.choices[0].message.content}"))
        print("\n🔁 Run this cell again to ask another question")
        
    except Exception as e:
        print(f"⚠️ Error: {str(e)}")
        print("Possible fixes:")
        print("- Try a different PDF number")
        print("- Ask a simpler question")
        print("- Contact IT about model deployment names")
