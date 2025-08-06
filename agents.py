import streamlit as st
from PIL import Image
import pytesseract
import openai
import io

# Configure your OpenAI key
openai.api_key = "your-openai-api-key"

# ------------------- AGENTS ------------------- #

def preprocessing_agent(image: Image.Image):
    """Convert to grayscale for better OCR"""
    return image.convert("L")

def ocr_agent(image: Image.Image):
    """Extract text using pytesseract"""
    return pytesseract.image_to_string(image)

def chart_classifier(text: str):
    """Guess chart type from OCR text"""
    if "%" in text and "distribution" in text.lower():
        return "Pie Chart"
    elif "x-axis" in text.lower() and "y-axis" in text.lower():
        return "Bar Chart or Line Chart"
    else:
        return "Unknown Chart"

def insight_generator_agent(text: str, chart_type: str):
    """Call GPT-4 to generate insights"""
    prompt = f"""You are a data analyst. Based on this extracted text from a {chart_type}, summarize key insights.

Extracted Text:
{text}

Only focus on high-level takeaways: what’s increasing, decreasing, biggest/smallest, etc."""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a smart data analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ------------------- STREAMLIT UI ------------------- #

st.title("📊 Universal Chart Insight Extractor")
st.write("Upload any statistical plot — bar chart, pie chart, line graph, etc. — and get automated insights using OCR + GPT.")

uploaded_file = st.file_uploader("Upload Chart Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chart", use_column_width=True)

    # Run agent flow
    gray_img = preprocessing_agent(image)
    extracted_text = ocr_agent(gray_img)
    chart_type = chart_classifier(extracted_text)
    insights = insight_generator_agent(extracted_text, chart_type)

    # Output
    st.subheader("🔍 Extracted Text")
    st.text(extracted_text)

    st.subheader("📌 Detected Chart Type")
    st.markdown(f"**{chart_type}**")

    st.subheader("📈 Insights (by GPT-4)")
    st.markdown(insights)