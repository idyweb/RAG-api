import os
import glob
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

def convert_txt_to_pdf():
    # Find all .txt files in the data directory and subdirectories
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "departments")
    txt_files = glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True)
    
    for txt_path in txt_files:
        pdf_path = txt_path.replace(".txt", ".pdf")
        
        print(f"Converting {txt_path} -> {pdf_path}")
        
        # Read the text content
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # Create a basic PDF using ReportLab
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        
        # Simple text wrapping setup
        c.setFont("Helvetica", 12)
        y_position = height - 50
        margin = 50
        
        for line in lines:
            line = line.strip('\n')
            # Handle wrapping for long lines
            wrapped_lines = simpleSplit(line, "Helvetica", 12, width - (2 * margin))
            
            for wrapped in wrapped_lines:
                if y_position < 50:
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y_position = height - 50
                    
                c.drawString(margin, y_position, wrapped)
                y_position -= 15
                
        c.save()

if __name__ == "__main__":
    convert_txt_to_pdf()
    print("✅ Conversion to PDF complete!")
