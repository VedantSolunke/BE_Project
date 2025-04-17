# utils.py

import os
import pickle
import faiss
import pandas as pd
import numpy as np
import requests
import json
import uuid
from langchain.schema import Document
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Ensure LangSmith project is set
langsmith_project = os.getenv("LANGSMITH_PROJECT", "pr-gripping-skunk-72")
os.environ["LANGSMITH_PROJECT"] = langsmith_project

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI API directly
api_key = os.getenv("OPENAI_API_KEY")

# Create LangChain tracer only if API key is available
langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "")
if langsmith_api_key:
    try:
        tracer = LangChainTracer(
            project_name=langsmith_project
        )
        logger.info("LangSmith tracing enabled")
    except Exception as e:
        logger.warning(f"Failed to initialize LangSmith tracer: {str(e)}")
        tracer = None
else:
    logger.warning("LangSmith API key not provided, tracing disabled")
    tracer = None

def get_embedding(text, api_key):
    """
    Get embedding from OpenAI using LangChain with tracing enabled.
    
    Args:
        text: Text to get embedding for
        api_key: OpenAI API key
        
    Returns:
        Embedding vector
    """
    try:
        logger.info("Getting embedding for query using LangChain")
        
        # Try direct OpenAI API call first to avoid LangChain's extra parameters
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = response.data[0].embedding
            logger.info("Successfully generated embedding using direct OpenAI API")
            return embedding
        except Exception as direct_error:
            logger.warning(f"Direct OpenAI API call failed: {str(direct_error)}")
            
            # Fallback to a very minimal LangChain implementation
            from langchain_openai import OpenAIEmbeddings
            import inspect
            
            # Get only the parameters that OpenAIEmbeddings accepts in this environment
            init_params = inspect.signature(OpenAIEmbeddings.__init__).parameters
            
            # Build a dictionary with only the parameters that are actually accepted
            embedding_kwargs = {"openai_api_key": api_key}
            if "model" in init_params:
                embedding_kwargs["model"] = "text-embedding-ada-002"
                
            embeddings = OpenAIEmbeddings(**embedding_kwargs)
            embedding = embeddings.embed_query(text)
            logger.info("Successfully generated embedding using minimal LangChain parameters")
            return embedding
            
    except Exception as e:
        logger.error(f"All embedding attempts failed: {str(e)}")
        raise

class LegalAnalysis(BaseModel):
    """Model for LLM output of legal analysis"""
    summary: str = Field(description="Overall summary of how the case relates to all relevant IPC sections")
    section_analyses: List[dict] = Field(description="Analysis of each section's relevance to the case")

class IPCEmbeddingManager:
    def __init__(self, csv_path="FIR_DATASET.csv"):
        self.csv_path = csv_path
        self.vector_store_path = "ipc_vector_store"
        self.index = None
        self.documents = None
        self.df = None

    def load_and_process_data(self):
        """Load and process the IPC dataset."""
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Successfully loaded dataset with {len(self.df)} entries")
            
            # Create text chunks for each IPC section
            documents = []
            for _, row in self.df.iterrows():
                text = f"""
                IPC Section: {row['URL']}
                Description: {row['Description']}
                Offense: {row['Offense']}
                Punishment: {row['Punishment']}
                Cognizable: {row['Cognizable']}
                Bailable: {row['Bailable']}
                Court: {row['Court']}
                """
                documents.append(Document(page_content=text, metadata={"section": row['URL']}))
            
            return documents
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def create_embeddings(self):
        """Inform user to use manual_create_embeddings.py instead."""
        logger.warning("Please use manual_create_embeddings.py script to create embeddings instead.")
        raise NotImplementedError("This method is not implemented. Use manual_create_embeddings.py instead.")

    def save_vector_store(self, path="ipc_vector_store"):
        """Inform user to use manual_create_embeddings.py instead."""
        logger.warning("Please use manual_create_embeddings.py script to create and save embeddings instead.")
        raise NotImplementedError("This method is not implemented. Use manual_create_embeddings.py instead.")

    def load_vector_store(self, path="ipc_vector_store"):
        """Load the vector store from disk."""
        try:
            # Load the FAISS index
            self.index = faiss.read_index(os.path.join(path, "index.faiss"))
            
            # Load the documents
            with open(os.path.join(path, "documents.pkl"), "rb") as f:
                self.documents = pickle.load(f)
            
            logger.info(f"Successfully loaded vector store from {path}")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    def search_similar_sections(self, query, k=5):
        """Search for similar IPC sections based on the query."""
        try:
            if self.index is None or self.documents is None:
                self.load_vector_store()
            
            # Get embedding for the query
            query_embedding = get_embedding(query, api_key)
            
            # Convert to numpy array
            query_embedding_array = np.array([query_embedding]).astype('float32')
            
            # Search the index
            D, I = self.index.search(query_embedding_array, k)
            
            # Get the documents for the indices
            results = []
            for i, (distance, idx) in enumerate(zip(D[0], I[0])):
                if idx < len(self.documents):
                    doc = Document(
                        page_content=self.documents[idx]["page_content"], 
                        metadata=self.documents[idx]["metadata"]
                    )
                    results.append((doc, float(distance)))
            
            return results
        except Exception as e:
            logger.error(f"Error searching similar sections: {str(e)}")
            raise

def preprocess_user_input(text):
    """Preprocess user input for better matching."""
    # Basic preprocessing - can be enhanced with more NLP techniques
    return text.strip().lower()

def format_search_results(results):
    """Format search results for display with enhanced styling."""
    formatted_results = []
    
    # Find min and max scores for normalization
    scores = [score for _, score in results]
    min_score = min(scores)
    score_range = max(scores) - min_score
    
    for idx, (doc, score) in enumerate(results, 1):
        content = doc.page_content
        section = doc.metadata.get('section', 'Unknown Section')
        
        # Parse the content into a structured format
        sections = {}
        for line in content.split('\n'):
            line = line.strip()
            if line and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if value:  # Only add non-empty values
                    sections[key] = value

        # Normalize score to percentage
        if score_range != 0:
            normalized_score = ((score - min_score) / score_range)
            similarity = int((1 - normalized_score) * 100)
        else:
            similarity = 95 if score < 0.5 else 85

        # Structure the content in a more readable format
        description_parts = []
        
        # Add section overview
        if sections.get('Description'):
            description_parts.append(f"""### Overview
**Section Summary:**
{sections['Description']}

**Purpose and Scope:**
This section of the Indian Penal Code (IPC) establishes the legal framework for addressing specific criminal acts and their consequences under Indian law.""")
        
        # Add offense details
        if sections.get('Offense'):
            description_parts.append(f"""### Offense Details
**Nature of the Offense:**
{sections['Offense']}

**Key Components of the Offense:**
• The act must be committed voluntarily
• There must be a direct connection between the act and the outcome
• The offense must fall within the scope defined by this section

**Legal Interpretation:**
The courts have interpreted this section to require proof of:
1. The actual commission of the offense
2. The intention or knowledge of the accused
3. The resulting consequences as specified in the section""")
        
        # Add punishment details
        if sections.get('Punishment'):
            description_parts.append(f"""### Punishment
**Prescribed Sentence:**
{sections['Punishment']}

**Sentencing Guidelines:**
• The court has the discretion to determine the appropriate sentence within the prescribed limits
• Factors considered include:
  - Severity of the offense
  - Circumstances of the case
  - Impact on the victim
  - Prior criminal record
  - Mitigating factors""")
        
        # Add legal classification and procedures
        legal_info = []
        if sections.get('Cognizable'):
            legal_info.append(f"""**Cognizable Status:**
• Classification: {sections['Cognizable']}
• Police Powers: {'Can arrest without warrant' if 'Cognizable' in sections['Cognizable'] else 'Cannot arrest without warrant'}
• Investigation: {'Immediate police investigation possible' if 'Cognizable' in sections['Cognizable'] else 'Requires court order for investigation'}""")
        
        if sections.get('Bailable'):
            legal_info.append(f"""**Bail Provisions:**
• Status: {sections['Bailable']}
• Implications: {'Accused has a right to bail' if 'Bailable' in sections['Bailable'] else 'Bail is at the discretion of the court'}
• Process: Bail application must be filed following proper legal procedures""")
        
        if sections.get('Court'):
            legal_info.append(f"""**Court Jurisdiction:**
• Competent Court: {sections['Court']}
• Powers: Full authority to try the case and pass judgment
• Jurisdiction: Territorial and subject-matter jurisdiction applies""")
        
        if legal_info:
            description_parts.append(f"""### Legal Classification and Procedures
{chr(10).join(legal_info)}

**Procedural Notes:**
1. The case must be filed in the appropriate court
2. Standard rules of criminal procedure apply
3. Evidence must meet legal standards
4. Rights of both accused and victim must be protected""")
        
        formatted_results.append({
            'section': section,
            'title': f"Section {section}",
            'description': '\n\n'.join(description_parts),
            'relevance': similarity,
            'index': idx,
            'raw_score': score,
            'sections_data': sections
        })
    
    # Sort results by relevance score in descending order
    formatted_results.sort(key=lambda x: x['relevance'], reverse=True)
    return formatted_results 

def analyze_with_llm(user_query, formatted_results):
    """
    Analyze search results with LLM to provide a case-specific summary and analysis.
    Uses LangChain's tracing functionality to track the LLM interactions in LangSmith.
    
    Args:
        user_query (str): The original user query describing the case
        formatted_results (list): The formatted search results from the similarity search
        
    Returns:
        dict: LLM analysis with case summary and section-specific analyses
    """
    try:
        # Create context for the LLM
        sections_context = []
        for result in formatted_results:
            section_info = {
                "section_number": result["section"],
                "title": result["title"],
                "description": result.get("sections_data", {}).get("Description", ""),
                "offense": result.get("sections_data", {}).get("Offense", ""),
                "punishment": result.get("sections_data", {}).get("Punishment", ""),
                "relevance_score": result["relevance"]
            }
            sections_context.append(section_info)
        
        # Format sections context for the prompt
        sections_formatted = ""
        for idx, section in enumerate(sections_context, 1):
            sections_formatted += f"""
            SECTION {idx}: {section['section_number']}
            - Description: {section['description']}
            - Offense: {section['offense']}
            - Punishment: {section['punishment']}
            - System Relevance Score: {section['relevance_score']}%
            """
        
        # Create a unique run ID for tracing
        run_id = str(uuid.uuid4())
        
        # Create prompt template using LangChain
        system_template = (
            "You are a knowledgeable and detail-oriented legal assistant "
            "with expertise in the Indian Penal Code (IPC). Your role is to analyze legal cases, "
            "interpret relevant IPC sections, and provide clear, concise explanations on how these sections apply. "
            "Ensure that your responses are supported by logical reasoning and presented in an easily digestible format."
        )

        human_template = """
        A user has provided the following case description:
        
        USER CASE: {user_query}
        
        Based on this case, the system has identified the following IPC sections as potentially relevant:
        
        {sections_formatted}
        
        Please analyze how these IPC sections apply to the specific case. Provide:
        
        1. An overall summary of how the case relates to the identified IPC sections
        2. For each section, provide:
           - A brief description of the offense in simple terms
           - The potential punishment
           - How this section specifically relates to elements of the described case
           - A relevance assessment (high/medium/low) with explanation
        
        Format your response as bullet points for clarity and conciseness.
        """
        
        # Initialize callbacks list based on tracer availability
        callbacks = []
        if tracer is not None:
            callbacks.append(tracer)
        
        # Initialize the ChatOpenAI model with optional tracing
        chat = ChatOpenAI(
            temperature=0.2,
            model="gpt-3.5-turbo",
            streaming=False,
            callbacks=callbacks if callbacks else None
        )
        
        # Create messages
        messages = [
            SystemMessage(content=system_template),
            HumanMessage(content=human_template.format(user_query=user_query, sections_formatted=sections_formatted))
        ]
        
        logger.info("Sending request to LLM using LangChain...")
        
        # Use LangChain's chat model with optional tracing
        response = chat(messages, callbacks=callbacks if callbacks else None)
        
        tracing_status = "with tracing" if tracer is not None else "without tracing"
        logger.info(f"Successfully generated LLM analysis {tracing_status}")
        
        # Create a structured analysis result
        analysis_result = {
            "summary": response.content,
            "sections_analyzed": sections_context
        }
        
        return analysis_result
            
    except Exception as e:
        logger.error(f"Error analyzing with LLM: {str(e)}")
        # Return basic structure in case of error
        return {
            "summary": "Unable to generate analysis due to an error. Please review the IPC sections directly.",
            "sections_analyzed": []
        }