from pathlib import Path
from pypdf import PdfReader

pdf_path = Path.cwd().parents[1] / 'Emergent_Models v14-04-25-1.pdf'
out_path = Path('references') / 'emergent_models_text.txt'

reader = PdfReader(str(pdf_path))
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open('w', encoding='utf-8') as f:
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ''
        f.write(f'--- Page {idx + 1} ---\n')
        f.write(text)
        if not text.endswith('\n'):
            f.write('\n')
        f.write('\n')

print(f'Extracted {len(reader.pages)} pages to {out_path}')
