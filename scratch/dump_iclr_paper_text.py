import fitz
import os

def dump_paper():
    pdf_path = "D:/AI/EquiPhase/doc/ICLR_Nature_2026.pdf"
    txt_path = "C:/Users/Sims/.gemini/antigravity/brain/e20d7f14-205f-4a52-9696-5f6f1c4caac8/scratch/ICLR_Nature_2026.txt"
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return
        
    doc = fitz.open(pdf_path)
    text = ""
    for i, page in enumerate(doc):
        text += f"\n================ PAGE {i+1} ================\n"
        text += page.get_text()
        
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
        
    print(f"Dumped {len(doc)} pages of {pdf_path} to {txt_path}.")

if __name__ == "__main__":
    dump_paper()
