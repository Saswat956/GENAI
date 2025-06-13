import openai

# Paste your actual OpenAI API key here
openai.api_key = "your-api-key-here"

try:
    response = openai.ChatCompletion.create(
        model="gpt-4-0125-preview",  # GPT-4.1 model
        messages=[{"role": "user", "content": "Is this GPT-4.1 responding?"}]
    )
    print("✅ API Key is working and GPT-4.1 responded:")
    print(response['choices'][0]['message']['content'])

except openai.error.AuthenticationError:
    print("❌ Invalid API Key.")
except openai.error.PermissionError:
    print("❌ API key doesn't have access to GPT-4.1 (model: gpt-4-0125-preview).")
except Exception as e:
    print("⚠️ Error:", str(e))