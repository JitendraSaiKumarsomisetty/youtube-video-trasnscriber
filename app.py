import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from transformers import BartTokenizer, BartForConditionalGeneration

# Function to extract transcript details using youtube-transcript-api
def extract_transcript_details(youtube_link):
    try:
        video_id = youtube_link.split("v=")[1].split("&")[0]  # Extract video ID
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript)
        return transcript_text
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return ""

# Function to summarize text in chunks
def summarize_text(text, model_name="facebook/bart-large-cnn"):
    # Load the summarization model and tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Define chunk size based on model constraints
    max_chunk_size = tokenizer.model_max_length - 50  # Slightly less than model max length
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_chunk_size)
    summary = ""
    
    for i in range(0, len(inputs.input_ids[0]), max_chunk_size):
        chunk = inputs.input_ids[0][i:i+max_chunk_size]
        chunk_attention_mask = inputs.attention_mask[0][i:i+max_chunk_size]
        chunk_input = {'input_ids': chunk.unsqueeze(0), 'attention_mask': chunk_attention_mask.unsqueeze(0)}
        
        # Generate summary for the chunk
        chunk_summary_ids = model.generate(**chunk_input, max_length=150, min_length=50, do_sample=False)
        chunk_summary = tokenizer.decode(chunk_summary_ids[0], skip_special_tokens=True)
        
        summary += chunk_summary + " "
    
    return summary.strip()

# Streamlit app code
st.title("YouTube Transcript to Detailed Notes Converter")
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    try:
        # Extract video ID from the YouTube link
        video_id = youtube_link.split("v=")[1].split("&")[0]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
    except IndexError:
        st.error("Invalid YouTube link format. Please enter a valid link.")

if st.button("Get Detailed Notes"):
    if youtube_link:
        transcript_text = extract_transcript_details(youtube_link)
        prompt = "Summarize the following transcript:\n"  # Define your prompt
        if transcript_text:
            # Debugging: Print the length and a snippet of the transcript
            st.write(f"Transcript Length: {len(transcript_text)}")
            st.write(f"Transcript Snippet: {transcript_text[:10000]}")  # Display first 500 characters for debugging

            summary = summarize_text(prompt + transcript_text)
            if summary:
                st.markdown("## Detailed Notes:")
                st.write(summary)
        else:
            st.error("Failed to extract transcript details.")
    else:
        st.error("Please enter a YouTube link to proceed.")
