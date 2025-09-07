import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="Koyl AI: Nutrition Advisor",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Custom CSS for Better Styling -----------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .input-container {
        background: #f8f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e1e8ed;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #ff7b7b 0%, #667eea 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Main Header -----------------
st.markdown("""
<div class="main-header">
    <h1>🥗 Koyl AI: Nutrition Advisor</h1>
    <p style="font-size: 18px; margin-top: 10px;">
        Get personalized dietary recommendations based on peer-reviewed research
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------- Sidebar: API Key -----------------
st.sidebar.markdown("""
<div class="sidebar-header">
    <h3>🔑 GROQ API Configuration</h3>
</div>
""", unsafe_allow_html=True)

groq_api_key = st.sidebar.text_input(
    "Enter your GROQ API key:", 
    type="password",
    help="Your API key is secure and not stored"
)

st.sidebar.markdown("**[Get your API key here →](https://console.groq.com)**")

# Add some helpful information in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📋 How to Use:
1. Enter your GROQ API key above
2. Specify patient conditions
3. List any allergies
4. Get AI-powered recommendations

### ✨ Features:
- **Research-backed** advice
- **Personalized** recommendations  
- **Allergy-aware** suggestions
- **Medical literature** context
""")

if not groq_api_key:
    st.error("🚨 **Please enter your GROQ API key in the sidebar to continue.**")
    st.info("💡 **Tip:** Get your free API key from the Groq Console using the link in the sidebar.")
    st.stop()

# ----------------- Load Model & Embeddings -----------------
try:
    model = ChatGroq(
        model="Meta-Llama/Llama-4-Scout-17b-16e-Instruct",
        groq_api_key=groq_api_key
    )
except Exception as e:
    st.error(f"❌ **Failed to load ChatGroq model:** {e}")
    st.stop()

embedding = HuggingFaceEmbeddings(model_name='avsolatorio/GIST-small-Embedding-v0')

# ----------------- Load FAISS Vector Store -----------------
@st.cache_resource
def load_vectorstore():
    try:
        vs = FAISS.load_local(
            "faiss_index",
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )
        return vs
    except Exception as e:
        st.error(f"❌ **Failed to load FAISS index:** {e}")
        return None

vectorstore = load_vectorstore()
if vectorstore is None:
    st.stop()

retriever = vectorstore.as_retriever()

# ----------------- Prompt Template -----------------
prompt = ChatPromptTemplate.from_template("""
You are a nutrition AI assistant that gives dietary recommendations based on peer-reviewed research.
Given a patient's health condition(s) and allergy profile, provide specific and research-backed nutrition advice.
Use only medically accurate, peer-reviewed information. Cite nutritional reasoning if available.

Patient Condition(s): {condition}
Allergies: {allergies}

Context from medical literature:
{context}

What are the best dietary recommendations for this patient?
""")

document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)

# ----------------- User Input Section -----------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="input-container">
        <h4>🏥 Patient Conditions</h4>
    </div>
    """, unsafe_allow_html=True)
    
    condition = st.text_input(
        "", 
        placeholder="e.g., high blood pressure, diabetes, heart disease",
        key="condition",
        help="Enter one or more health conditions separated by commas"
    )

with col2:
    st.markdown("""
    <div class="input-container">
        <h4>🚫 Allergy Profile</h4>
    </div>
    """, unsafe_allow_html=True)
    
    allergies = st.text_input(
        "", 
        placeholder="e.g., dairy, gluten, nuts, shellfish",
        key="allergies",
        help="List any food allergies or intolerances"
    )

# ----------------- Generate Button -----------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 **Get Personalized Dietary Recommendations**"):
    if not condition or not allergies:
        st.warning("⚠️ **Please fill in both condition and allergy fields to proceed.**")
        st.stop()
    
    query = f"{condition} {allergies}"
    
    # Show progress with better messaging
    with st.spinner("🔍 **Searching medical literature for relevant information...**"):
        retrieved_docs = retriever.invoke(query)
    
    with st.spinner("🧠 **AI is analyzing and generating personalized recommendations...**"):
        response = document_chain.invoke({
            "condition": condition,
            "allergies": allergies,
            "context": retrieved_docs
        })
    
    # Display results in an attractive format
    st.markdown("---")
    st.markdown("""
    <div class="recommendation-box">
        <h2 style="color: #667eea; margin-bottom: 1rem;">🎯 Your Personalized Dietary Recommendations</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**👤 Patient Profile:** {condition}")
    st.markdown(f"**🚫 Allergies:** {allergies}")
    st.markdown("---")
    
    # Display the response with better formatting
    st.markdown("### 📋 Recommendations:")
    st.write(response)
    
    # Add disclaimer
    st.markdown("---")
    st.info("⚠️ **Medical Disclaimer:** These recommendations are AI-generated based on medical literature. Always consult with a healthcare professional before making significant dietary changes.")

# ----------------- Footer -----------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px; margin-top: 2rem;">
    <p>💡 <strong>Koyl AI Nutrition Advisor</strong> - Powered by AI and Medical Research</p>
    <p>🔬 Evidence-based • 🎯 Personalized • 🛡️ Allergy-aware</p>
</div>
""", unsafe_allow_html=True)
