from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from src.utils.resolve_path import resolve_project_path

def convert_to_pdf(base_path, destination_path, language):
    base_path = resolve_project_path(base_path)
    destination_path = resolve_project_path(destination_path)

    file_count = len(list(base_path.glob(f"{language}*.txt")))

    for i in range(file_count + 1):
        if i == 0 :
            continue
        else:
            textfile = f"{base_path}/{language}_merged_{i}.txt"
            pdffile = f"{destination_path}/{language}_converted_{i}.pdf"

            doc = SimpleDocTemplate(
                pdffile,
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )

            styles = getSampleStyleSheet()
            style = styles["Normal"]
            style.leading = 14

            story = []

            with open(textfile, "r", encoding="utf-8") as f:
                text = f.read().split("\n")

            for line in text:
                # WICHTIG: keine &nbsp; verwenden
                para = Paragraph(line, style)
                story.append(para)
                story.append(Spacer(1, 4))

            doc.build(story)
