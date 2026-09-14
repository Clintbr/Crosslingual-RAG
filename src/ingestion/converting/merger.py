from src.utils.resolve_path import resolve_project_path


def merge_txt_files(directory, target_directory, lang, files_num):
    directory = resolve_project_path(directory)
    target_directory = resolve_project_path(target_directory)

    txt_files = sorted(directory.glob("*.txt"))

    for i in range(0, len(txt_files), files_num):
        batch = txt_files[i:i + files_num]
        output_file = target_directory / f"{lang}_merged_{i // files_num + 1}.txt"

        with output_file.open("w", encoding="utf-8") as outfile:
            for j, txt_file in enumerate(batch):
                content = txt_file.read_text(encoding="utf-8").strip()

                if j > 0:
                    outfile.write("\n")

                outfile.write(content)

        print(f"Created: {output_file}")
