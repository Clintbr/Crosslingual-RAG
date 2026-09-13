from src.config import BASE_PATH, DE, EN, FR, DOCS_PATH
from src.ingestion.converting.convert_to_pdf import convert_to_pdf
from src.ingestion.converting.merger import merge_txt_files

languages = [DE, EN, FR]
output_dir = BASE_PATH + "/merged"

def merge_and_convert_docs():
    for language in languages:
        input_dir = BASE_PATH + "/raw/" + language
        merge_txt_files(input_dir, output_dir, language, files_num=26)
        convert_to_pdf(output_dir, DOCS_PATH, language)

if __name__ == '__main__':
    merge_and_convert_docs()