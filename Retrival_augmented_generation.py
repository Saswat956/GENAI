# Define the list of structured questions (headings reformulated as questions)
structured_questions = [
    
]

# Output collector
full_output = ""

# Loop through each heading/question
for idx, question in enumerate(structured_questions, 1):
    print(f"[Processing] Section {idx}: {question}")
    
    # Step 1: Embed the question
    query_embedding = encoder.predict({"inputs": [question]})[0]

    # Step 2: Query the vector DB for relevant chunks
    result = collection.query(query_embeddings=[query_embedding], n_results=2)
    context = "\n".join(result["documents"][0])

    # Step 3: Format the prompt with chain-of-thought style
    prompt = f"""You are a legal assistant extracting structured information from a reinsurance agreement.
Think step-by-step and answer the QUESTION using the CONTEXT.
If the answer is not in the CONTEXT, respond with "Not specified in the document."

CONTEXT:
{context}

QUESTION: {question}

Let's think step-by-step:

ANSWER:"""

    # Step 4: Generate the answer
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 700,
            "top_p": 0.9,
            "temperature": 0.6
        }
    }

    response = client.invoke_endpoint(
        EndpointName="llama-3-generator-n3",
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    generated_output = json.loads(response["Body"].read())
    answer = generated_output[0]["generated_text"] if isinstance(generated_output, list) else generated_output

    # Step 5: Append the formatted answer
    full_output += f"\n\n---\nSection {idx}: {question}\n\n{answer.strip()}\n"

# Step 6: Save to file
with open("Treaty_Summary.txt", "w") as f:
    f.write(full_output)

print("✅ Treaty summary document created as 'Treaty_Summary.txt'")
