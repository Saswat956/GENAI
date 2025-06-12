import boto3

# Set region (must support Bedrock)
region = "us-east-1"

# Create Bedrock Runtime client
bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)

import json

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

payload = {
    "messages": [
        {"role": "user", "content": "Summarize the key terms of a ."}
    ],
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9
}

response = bedrock_runtime.invoke_model(
    modelId=model_id,
    contentType="application/json",
    accept="application/json",
    body=json.dumps(payload)
)

result = json.loads(response["body"].read())
print(result["content"][0]["text"] if "content" in result else result)

bedrock = boto3.client("bedrock", region_name=region)
models = bedrock.list_foundation_models()
for m in models["modelSummaries"]:
    print(m["modelId"], "-", m["providerName"])


output = json.loads(response["body"].read())
