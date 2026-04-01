import google.generativeai as genai
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content([prompt, image_part])  # Multimodal!
from openai import OpenAI
client = OpenAI(base_url="https://api.perplexity.ai")  # Custom endpoint
response = client.chat.completions.create(model="llama-3.1-sonar-small-128k-online", ...)