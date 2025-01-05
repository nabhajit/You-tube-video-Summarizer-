import streamlit as st
from dotenv import load_dotenv
import os
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit.components.v1 as components
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from fpdf import FPDF
from pytube import YouTube
import base64
import tempfile
from transformers import pipeline

# Download required NLTK data
def download_nltk_data():
    try:
        # Download all required data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        
        # Additional download for punkt_tab
        nltk.download('punkt_tab', quiet=True)
        
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        st.info("Please run these commands in your Python console:")
        st.code("""
import nltk
nltk.download()  # This will open the NLTK downloader GUI
        """)
        return False

# Initialize NLTK data
if not download_nltk_data():
    st.stop()

# Load environment variables
load_dotenv()

def create_download_link(val, filename):
    try:
        b64 = base64.b64encode(val).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return None

def generate_pdf(transcript, summary, title="YouTube Summary"):
    try:
        # Initialize PDF with A4 format
        pdf = FPDF(format='A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Set margins
        margin = 20
        pdf.set_left_margin(margin)
        pdf.set_right_margin(margin)
        effective_page_width = pdf.w - 2*margin
        
        # Add title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, title, ln=True, align='C')
        pdf.ln(5)
        
        # Add summary section
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Summary:', ln=True)
        pdf.ln(2)
        
        # Process summary
        pdf.set_font('Arial', '', 12)
        summary_lines = summary.replace('•', '-').split('\n\n')
        
        for i, line in enumerate(summary_lines):
            # Handle each bullet point
            line = line.strip()
            if not line:
                continue
            
            # Add bullet point indentation
            pdf.set_x(margin + 5)
            
            # Split line into words for proper wrapping
            words = line.split()
            current_line = []
            first_line = True
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                # Account for bullet point indentation in first line
                current_width = pdf.get_string_width(test_line)
                if first_line:
                    available_width = effective_page_width - 5
                else:
                    available_width = effective_page_width
                
                if current_width < available_width:
                    current_line.append(word)
                else:
                    # Write current line
                    if current_line:
                        if first_line:
                            pdf.cell(5, 5, "-", 0, 0)
                            pdf.multi_cell(effective_page_width - 5, 5, ' '.join(current_line))
                            first_line = False
                        else:
                            pdf.set_x(margin + 10)  # Indent continuation lines
                            pdf.multi_cell(effective_page_width - 10, 5, ' '.join(current_line))
                    current_line = [word]
            
            # Write remaining words
            if current_line:
                if first_line:
                    pdf.cell(5, 5, "-", 0, 0)
                    pdf.multi_cell(effective_page_width - 5, 5, ' '.join(current_line))
                else:
                    pdf.set_x(margin + 10)
                    pdf.multi_cell(effective_page_width - 10, 5, ' '.join(current_line))
            
            pdf.ln(3)  # Add space between bullet points
        
        # Add transcript section
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Transcript Preview:', ln=True)
        pdf.ln(2)
        
        # Process transcript
        pdf.set_font('Arial', '', 12)
        transcript_preview = transcript[:1000] + "..."
        
        # Split transcript into paragraphs
        paragraphs = transcript_preview.split('\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            words = paragraph.split()
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                if pdf.get_string_width(test_line) < effective_page_width:
                    current_line.append(word)
                else:
                    if current_line:
                        pdf.multi_cell(effective_page_width, 5, ' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                pdf.multi_cell(effective_page_width, 5, ' '.join(current_line))
            pdf.ln(3)
        
        # Return the PDF as bytes
        return bytes(pdf.output())
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def download_video(url):
    try:
        yt = YouTube(
            url,
            use_oauth=True,
            allow_oauth_cache=True
        )
        
        # Get the highest quality stream
        video = (yt.streams
                .filter(progressive=True, file_extension='mp4')
                .order_by('resolution')
                .desc()
                .first())
        
        if not video:
            st.error("No suitable video stream found.")
            return None, None
            
        # Create a temporary file with a unique name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            try:
                # Download the video
                video.download(
                    output_path=os.path.dirname(tmp_file.name),
                    filename=os.path.basename(tmp_file.name)
                )
                return tmp_file.name, yt.title
            except Exception as e:
                st.error(f"Download failed: {str(e)}")
                # Provide alternative download link
                st.markdown(f"You can download the video directly from YouTube: {url}")
                return None, None
            
    except Exception as e:
        st.error(f"Error accessing video: {str(e)}")
        st.info("Due to YouTube's restrictions, direct video download might not be available. "
                "You can download the video directly from YouTube.")
        st.markdown(f"Video URL: {url}")
        return None, None

def extract_transcript_details(youtube_video_url):
    try:
        if "v=" in youtube_video_url:
            video_id = youtube_video_url.split("v=")[1].split("&")[0]
        elif "youtu.be" in youtube_video_url:
            video_id = youtube_video_url.split("/")[-1]
        else:
            raise ValueError("Invalid YouTube URL format")

        # Get video title using a simpler approach
        try:
            yt = YouTube(youtube_video_url)
            video_title = yt.title
        except:
            # Fallback: Use video ID as title if title extraction fails
            video_title = f"YouTube Video ({video_id})"

        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([item["text"] for item in transcript_data])
        return transcript, video_id, video_title

    except Exception as e:
        st.error(f"Error extracting transcript: {e}")
        return None, None, None

def generate_summary(transcript_text):
    try:
        # Initialize the summarization pipeline with a smaller, faster model
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-base",  # Using smaller base model
            device=-1  # Use CPU to avoid CUDA issues
        )
        
        # Split text into smaller chunks for faster processing
        max_chunk_length = 512  # Reduced chunk size
        chunks = [transcript_text[i:i + max_chunk_length] for i in range(0, len(transcript_text), max_chunk_length)]
        
        # Summarize each chunk with shorter output
        summaries = []
        for chunk in chunks[:2]:  # Process only first 2 chunks for speed
            summary = summarizer(
                chunk,
                max_length=60,  # Shorter summaries
                min_length=20,
                do_sample=False,
                truncation=True
            )
            summaries.append(summary[0]['summary_text'])
        
        # Format as bullet points
        summary_points = [f"- {text.strip()}" for text in summaries if text.strip()]
        summary = '\n\n'.join(summary_points)
        
        return summary

    except Exception as e:
        st.error(f"Error generating summary: {e}")
        # Fallback to simple extractive summarization if transformer fails
        try:
            sentences = sent_tokenize(transcript_text)
            # Get first and last sentences as a basic summary
            if len(sentences) >= 2:
                summary_points = [
                    f"- {sentences[0].strip()}",
                    f"- {sentences[-1].strip()}"
                ]
                return '\n\n'.join(summary_points)
        except:
            pass
        return None

# Add custom CSS for styling
def add_custom_css():
    custom_css = """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stTextInput>div>input {
        border-radius: 12px;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    """
    components.html(custom_css, height=0)

# Initialize Streamlit app
st.set_page_config(page_title="YouTube Summarizer", layout="wide")
add_custom_css()

# App title
st.title("YouTube Transcript to Detailed Notes Converter")

# Initialize session states
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []
if 'processed' not in st.session_state:
    st.session_state['processed'] = False
if 'transcript_text' not in st.session_state:
    st.session_state['transcript_text'] = None
if 'video_id' not in st.session_state:
    st.session_state['video_id'] = None
if 'summary' not in st.session_state:
    st.session_state['summary'] = None

# Main input
youtube_link = st.text_input("Enter YouTube Video Link:")

# Process button
if youtube_link:
    if st.button("Process Video"):
        with st.spinner("Processing video..."):
            # Extract transcript and get video ID
            transcript_text, video_id, video_title = extract_transcript_details(youtube_link)
            
            if transcript_text and video_id:
                # Store in session state
                st.session_state['transcript_text'] = transcript_text
                st.session_state['video_id'] = video_id
                st.session_state['video_title'] = video_title
                
                # Display video thumbnail
                st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
                
                # Generate summary
                summary = generate_summary(transcript_text)
                
                if summary:
                    # Store summary in session state
                    st.session_state['summary'] = summary
                    st.session_state['processed'] = True
                    
                    # Display transcript and summary
                    st.markdown("## Transcript Preview")
                    st.write(transcript_text[:500] + "...")
                    
                    st.markdown("## Summary")
                    st.markdown(summary)
                    
                    # Add to search history
                    if youtube_link not in st.session_state['search_history']:
                        st.session_state['search_history'].append(youtube_link)

# Add download buttons outside the process button
if st.session_state['processed']:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Watch on YouTube"):
            st.markdown(f"[Open Video on YouTube]({youtube_link})")
    
    with col2:
        if st.button("Download PDF Summary"):
            with st.spinner("Generating PDF..."):
                title = f"Summary of {st.session_state.get('video_title', 'YouTube Video')}"
                pdf_data = generate_pdf(
                    st.session_state['transcript_text'],
                    st.session_state['summary'],
                    title
                )
                if pdf_data:
                    safe_title = "".join(c for c in st.session_state['video_title'] if c.isalnum() or c in (' ', '-', '_'))[:50]
                    st.markdown(
                        create_download_link(pdf_data, f"summary_{safe_title}.pdf"),
                        unsafe_allow_html=True
                    )
                    st.success("PDF ready for download!")
                else:
                    st.error("Failed to generate PDF. Please try again.")

# Display search history in sidebar
st.sidebar.markdown("## Search History")
for i, link in enumerate(st.session_state['search_history']):
    st.sidebar.write(f"{i+1}. {link}")
