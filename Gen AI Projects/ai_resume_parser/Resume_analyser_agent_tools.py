# Warning control
import warnings
warnings.filterwarnings('ignore')

#!pip install PyMuPDF
#!pip install python-docxa
#!pip install crewai crewai-tools
import sqlite3
import fitz  # PyMuPDF for PDF processing
import docx  # python-docx for DOCX processing
from crewai_tools import SerperDevTool
import os
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Streamlit must be imported after setting page config
import streamlit as st

# Set page configuration FIRST
st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")

# Custom CSS for background color
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #50a7c7;
        }}
        .title-text {{
            font-size: 1000px;  /* Increased font size */
            font-weight: bold;
            color: white;
            text-align: center;
            margin-bottom: 20px;
        }}
        .subtitle-text {{
            font-style: italic;
            color: white;
            text-align: center;
            margin-bottom: 30px;
        }}
        .upload-box {{
            border: 2px dashed #ffffff;
            padding: 20px;
            text-align: center;
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            margin: 0 auto;
            width: 50%;  /* Reduced width of the upload box */
        }}
        .stTextInput input {{
            max-width: 300px;  /* Reduced width of the text input */
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown('<p class="title-text">Smart Resume Analyzer & Job Matcher</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Expected Runtime: 1 Min</p>', unsafe_allow_html=True)

# Resume upload section
st.markdown("### Upload Your Resume (PDF or DOCX)")
uploaded_file = st.file_uploader("Drop File Here or Click to Upload", type=["pdf", "docx"], help="Upload your resume for analysis.",key="resume_upload")

# Preferred location input
preferred_location = st.text_input("Preferred Location", placeholder="e.g., San Francisco",max_chars=50, key="location_input")


    

# Submit button
submit_button = st.button("Submit",icon="😃")

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file using PyMuPDF."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file using python-docx."""
    doc = docx.Document(file_path)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return "\n".join(fullText)

# Function to extract text from resume
def extract_text_from_resume(file_path):
    """Determines file type and extracts text."""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        return "Unsupported file format."

# Initialize agents and tasks
def initialize_crew():
    search_tool = SerperDevTool()

    # Agent 1: Resume Advisor
    resume_advisor = Agent(
        role="Professional Resume Advisor",
        goal="Give feedback on the resume to make it stand out in the job market.",
        verbose=True,
        backstory="With a strategic mind and an eye for detail, you excel at providing feedback on resumes to highlight the most relevant skills and experiences."
    )

    # Agent 2: Resume Writer
    resume_writer = Agent(
        role="Professional Resume Writer",
        goal="Based on the feedback received from Resume Advisor, make changes to the resume to make it stand out in the job market.",
        verbose=True,
        backstory="With a strategic mind and an eye for detail, you excel at refining resumes based on the feedback to highlight the most relevant skills and experiences."
    )

    # Agent 3: Job Researcher
    job_researcher = Agent(
        role="Senior Recruitment Consultant",
        goal="Find the 5 most relevant, recently posted jobs based on the improved resume received from resume advisor and the location preference.",
        tools=[search_tool],
        verbose=True,
        backstory="""As a senior recruitment consultant, your prowess in finding the most relevant jobs based on the resume and location preference is unmatched. 
        You can scan the resume efficiently, identify the most suitable job roles, and search for the best-suited recently posted open job positions at the preferred location."""
    )

    # Task for Resume Advisor Agent: Give Feedback
    resume_advisor_task = Task(
        description=(
            """Give feedback on the resume to make it stand out for recruiters. 
            Review every section, including the summary, work experience, skills, and education. Suggest to add relevant sections if they are missing.  
            Also, give an overall score to the resume out of 10. This is the resume: {resume}"""
        ),
        expected_output="The overall score of the resume followed by the feedback in bullet points.",
        agent=resume_advisor
    )

    # Task for Resume Writer Agent: Improve Resume
    resume_writer_task = Task(
        description=(
            """Rewrite the resume based on the feedback to make it stand out for recruiters. You can adjust and enhance the resume but don't make up facts. 
            Review and update every section, including the summary, work experience, skills, and education to better reflect the candidate's abilities. This is the resume: {resume}"""
        ),
        expected_output="Resume in markdown format that effectively highlights the candidate's qualifications and experiences.",
        context=[resume_advisor_task],
        agent=resume_writer
    )

    # Task for Job Researcher Agent: Find Jobs
    research_task = Task(
        description="""Find the 5 most relevant recent job postings based on the resume received from resume advisor and location preference. This is the preferred location: {location}. 
        Use the tools to gather relevant content and shortlist the 5 most relevant, recent, job openings.""",
        expected_output="A bullet point list of the 5 job openings, with the appropriate links and detailed description about each job, in markdown format.",
        agent=job_researcher
    )

    # Create Crew
    crew = Crew(
        name="Resume Analysis Crew",
        tasks=[resume_advisor_task, resume_writer_task, research_task],
        agents=[resume_advisor, resume_writer, job_researcher]
    )

    return crew,resume_advisor_task, resume_writer_task, research_task


# Function to run the crew and get results
def resume_agent(file_path, location):
    resume_text = extract_text_from_resume(file_path)

    crew, resume_advisor_task, resume_writer_task, research_task = initialize_crew()
    result = crew.kickoff(inputs={"resume": resume_text, "location": location})

    # Extract outputs
    feedback = resume_advisor_task.output.raw.strip("```markdown").strip("```").strip()
    improved_resume = resume_writer_task.output.raw.strip("```markdown").strip("```").strip()
    job_roles = research_task.output.raw.strip("```markdown").strip("```").strip()

    return feedback, improved_resume, job_roles

# Handle file upload and processing
if submit_button:
    if uploaded_file is None:
        st.error("Please upload a resume before submitting.")
    else:
        with st.spinner("Analyzing your resume and finding job matches..."):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                feedback, improved_resume, job_roles = resume_agent(tmp_file_path, preferred_location)
                
                st.success("Analysis complete!")
                st.markdown("## Resume Feedback:")
                st.markdown(feedback)
                
                st.markdown("## Improved Resume Suggestions:")
                st.markdown(improved_resume)
                
                st.markdown("## Relevant Job Roles:")
                st.markdown(job_roles)
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)