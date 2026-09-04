from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from src.config import (BASE_PATH, DESTINATION_PATH, EN, DE, FR)

base_path = BASE_PATH
destination_path = DESTINATION_PATH
languages = [EN, DE, FR]
for i in range(11):
    if i == 0 :
        continue
    else:
        for language in languages:
            textfile = f"{base_path}{language}{i}.txt"
            pdffile = f"{destination_path}{language}{i}.pdf"

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