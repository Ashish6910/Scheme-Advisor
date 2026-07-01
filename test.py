import google.generativeai as genai

# ✅ Use your Gemini API key
genai.configure(api_key="AQ.Ab8RN6LBM-BMDEgS3nmwBUIk-y82G6qRADYrJWegWR3azEnXZw")

# ✅ List all available models
models = genai.list_models()

for m in models:
    print(m.name)