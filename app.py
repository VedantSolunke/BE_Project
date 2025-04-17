# app.py

import streamlit as st
from utils import IPCEmbeddingManager, preprocess_user_input, format_search_results, analyze_with_llm, tracer
import os
import sys
from dotenv import load_dotenv
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Ensure LangSmith project is set
langsmith_project = os.getenv("LANGSMITH_PROJECT", "pr-gripping-skunk-72")
os.environ["LANGSMITH_PROJECT"] = langsmith_project
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

# Set page configuration
st.set_page_config(
    page_title="Legal Vault : Legal Assistance Platform",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .content-block {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        color: #ffffff;
    }
    .content-block h3 {
        font-size: 18px;
        font-weight: bold;
        padding: 12px 15px;
        margin: 0 0 15px 0;
        border-radius: 4px;
        color: #ffffff;
    }
    /* Overview section styling */
    .overview-section {
        border-left: 4px solid #64B5F6;
        background-color: #0D47A1;
    }
    .overview-section h3 {
        background-color: #1565C0;
    }
    /* Offense section styling */
    .offense-section {
        border-left: 4px solid #BA68C8;
        background-color: #4A148C;
    }
    .offense-section h3 {
        background-color: #6A1B9A;
    }
    /* Punishment section styling */
    .punishment-section {
        border-left: 4px solid #EF5350;
        background-color: #B71C1C;
    }
    .punishment-section h3 {
        background-color: #C62828;
    }
    /* Additional Information section styling */
    .additional-info-section {
        border-left: 4px solid #81C784;
        background-color: #1B5E20;
    }
    .additional-info-section h3 {
        background-color: #2E7D32;
    }
    /* LLM Analysis section styling */
    .llm-analysis-section {
        border-left: 4px solid #FFD54F;
        background-color: #F57F17;
    }
    .llm-analysis-section h3 {
        background-color: #FF8F00;
    }
    .section-text {
        line-height: 1.6;
        font-size: 16px;
        padding: 10px;
        color: #ffffff;
    }
    .metadata-items {
        display: grid;
        gap: 10px;
        padding: 10px;
        color: #ffffff;
    }
    .metadata-items strong {
        color: #90CAF9;
        margin-right: 8px;
    }
    .relevance-score {
        margin-top: 15px;
        padding: 10px 15px;
        border-radius: 4px;
        text-align: right;
        font-size: 16px;
        font-weight: bold;
    }
    .relevance-high {
        background-color: #1B5E20;
        color: #81C784;
        border-left: 4px solid #81C784;
    }
    .relevance-medium {
        background-color: #E65100;
        color: #FFB74D;
        border-left: 4px solid #FFB74D;
    }
    .relevance-low {
        background-color: #B71C1C;
        color: #EF5350;
        border-left: 4px solid #EF5350;
    }
    .stExpander {
        border: 1px solid #333333;
        border-radius: 8px;
        margin-bottom: 15px;
        background-color: #1a1a1a;
    }
    /* Improve expander header visibility */
    .streamlit-expanderHeader {
        background-color: #2d2d2d;
        border-radius: 8px 8px 0 0;
        padding: 10px 15px;
        color: #ffffff;
    }
    /* Add hover effect to sections */
    .content-block:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: box-shadow 0.3s ease;
    }
    /* Additional styling for better visibility */
    .section-text a {
        color: #90CAF9;
    }
    .section-text strong {
        color: #90CAF9;
    }
    /* LLM Analysis Container */
    .llm-container {
        background-color: #2D2D2D;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 30px;
        border: 1px solid #444;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .llm-container h2 {
        color: #FFD54F;
        margin-bottom: 20px;
        border-bottom: 2px solid #FFD54F;
        padding-bottom: 10px;
    }
    .llm-summary {
        background-color: #3A3A3A;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 4px solid #FFD54F;
    }
    .bullet-point {
        margin-left: 20px;
        position: relative;
        padding-left: 15px;
    }
    .bullet-point:before {
        content: "•";
        position: absolute;
        left: 0;
        color: #FFD54F;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the embedding manager
@st.cache_resource
def get_embedding_manager():
    return IPCEmbeddingManager()

def main():
    st.title("⚖️ Legal Vault : Legal Assistance Platform")
    st.markdown("""
    ### Welcome to the Legal Assistance Platform! 
    This tool helps you find relevant IPC sections based on your case details.
    
    **Please describe your case in detail, including:**
    - What happened
    - Who was involved
    - When and where it occurred
    - Any injuries or damages
    """)

    # Initialize embedding manager
    embedding_manager = get_embedding_manager()

    col1, col2 = st.columns([3, 1])
    
    with col1:
        # User input
        user_input = st.text_area(
            "Describe your case:",
            placeholder="Example: I was attacked by a person with a knife on March 5th, 2025, and suffered injuries on my arm. The incident occurred at a park near my home.",
            height=150
        )

    with col2:
        # Number of results to show
        num_results = st.slider("Number of results:", 1, 10, 5)
        st.markdown("""
        #### Relevance Guide:
        - 🟢 90-100%: Highly Relevant
        - 🟡 70-89%: Moderately Relevant
        - 🔴 Below 70%: Less Relevant
        """)

    if st.button("🔍 Analyze Case", type="primary"):
        if user_input:
            with st.spinner("🔄 Analyzing your case..."):
                try:
                    # Create a run name for LangSmith tracing
                    run_name = f"Legal Analysis - {user_input[:30]}..." if len(user_input) > 30 else user_input
                    tracer.run_name = run_name
                    
                    # Setup Streamlit callback handler for live updates
                    st_callback = StreamlitCallbackHandler(st.container())
                    
                    # Preprocess input
                    processed_input = preprocess_user_input(user_input)
                    
                    # Search for similar sections with tracing
                    results = embedding_manager.search_similar_sections(processed_input, k=num_results)
                    formatted_results = format_search_results(results)
                    
                    # NEW: Analyze results with LLM with visual tracing
                    with st.spinner("🧠 Generating legal analysis..."):
                        llm_analysis = analyze_with_llm(user_input, formatted_results)
                    
                    # Display LangSmith project info
                    st.sidebar.markdown("### 📊 LangSmith Tracking")
                    st.sidebar.info(f"This session is being tracked in LangSmith project: **{langsmith_project}**")
                    
                    # NEW: Display LLM Analysis first
                    st.markdown("## 🧠 Legal Analysis of Your Case")
                    
                    # Fix: Using raw string literals to handle backslashes correctly
                    summary_html = llm_analysis['summary'].replace('\n', '<br>').replace('•', '<div class="bullet-point">').replace('\n-', '</div><div class="bullet-point">')
                    st.markdown(
                        f"""<div class="llm-container">
                            <h2>LLM-Based Analysis & Recommendations</h2>
                            <div class="llm-summary">
                                {summary_html}
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                    # Display detailed results
                    st.markdown("## 📋 Relevant IPC Sections")
                    
                    for result in formatted_results:
                        # Determine relevance indicator
                        if result['relevance'] >= 90:
                            relevance_class = "relevance-high"
                            relevance_emoji = "🟢"
                        elif result['relevance'] >= 70:
                            relevance_class = "relevance-medium"
                            relevance_emoji = "🟡"
                        else:
                            relevance_class = "relevance-low"
                            relevance_emoji = "🔴"
                        
                        # Create expandable section
                        with st.expander(f"{relevance_emoji} {result['title']} - Relevance: {result['relevance']}%"):
                            sections = result['description'].split('###')
                            
                            for section in sections:
                                if section.strip():
                                    # Get section title and content
                                    lines = section.strip().split('\n', 1)
                                    if len(lines) == 2:
                                        title, content = lines
                                        title = title.strip()
                                        content = content.strip()
                                        
                                        # Apply appropriate styling based on section
                                        if "Overview" in title:
                                            st.markdown(f"""
                                                <div class='content-block overview-section'>
                                                    <h3>{title}</h3>
                                                    <div class='section-text'>{content}</div>
                                                </div>
                                            """, unsafe_allow_html=True)
                                        elif "Offense Details" in title:
                                            st.markdown(f"""
                                                <div class='content-block offense-section'>
                                                    <h3>{title}</h3>
                                                    <div class='section-text'>{content}</div>
                                                </div>
                                            """, unsafe_allow_html=True)
                                        elif "Punishment" in title:
                                            st.markdown(f"""
                                                <div class='content-block punishment-section'>
                                                    <h3>{title}</h3>
                                                    <div class='section-text'>{content}</div>
                                                </div>
                                            """, unsafe_allow_html=True)
                                        elif "Additional Information" in title:
                                            st.markdown(f"""
                                                <div class='content-block additional-info-section'>
                                                    <h3>{title}</h3>
                                                    <div class='metadata-items'>
                                                        {content.replace('**', '<strong>').replace('**', '</strong>')}
                                                    </div>
                                                </div>
                                            """, unsafe_allow_html=True)
                            
                            # Display relevance score
                            st.markdown(
                                f"""<div class='relevance-score {relevance_class}'>
                                    Match Score: {result['relevance']}%
                                </div>""",
                                unsafe_allow_html=True
                            )
                            
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
        else:
            st.warning("⚠️ Please enter your case details to proceed.")

    # Add footer
    st.markdown("---")
    st.markdown("""
    ### ℹ️ About
    This platform uses advanced AI technology to help you find relevant IPC sections based on your case details.
    The information provided is for general guidance only and should not be considered legal advice.
    
    **Note:** Always consult with a qualified legal professional for specific legal advice.
    """)

if __name__ == "__main__":
    main()