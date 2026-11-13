from pypdf import PdfReader
import os

# 1. Define the path to your PDF file
# IMPORTANT: Change "infosys_recent.pdf" to the EXACT name of the file you downloaded!
pdf_path = "data/raw/infosys_q2_fy25.pdf"

def extract_text_from_pdf(file_path):
    print(f"📖 Reading: {file_path}...")
    
    # Open the PDF file
    reader = PdfReader(file_path)
    
    full_text = ""
    
    # Loop through every page and extract the text
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n--- Page {i+1} ---\n{text}"
            
    return full_text

# 2. Run the function and check the result
if __name__ == "__main__":
    # Check if the file actually exists before trying to read it
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Could not find '{pdf_path}'.")
        print("Please check the filename and make sure it is inside the 'data/raw/' folder.")
    else:
        try:
            # Get all the text
            extracted_text = extract_text_from_pdf(pdf_path)
            
            # Print a preview (first 800 characters) so we don't flood the terminal
            print("\n✅ Success! Here is a preview of the extracted text:")
            print("-" * 60)
            print(extracted_text[:800]) 
            print("-" * 60)
            print(f"\n(Total characters extracted: {len(extracted_text)})")
            
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")