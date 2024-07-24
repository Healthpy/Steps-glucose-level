import os

def remove_first_line_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        if lines[0].startswith('#'):
            lines = lines[1:]
        elif lines[0].strip().isdigit():
            lines = lines[2:]

        with open(file_path, 'w') as file:
            file.writelines(lines)

        print(f"Processed file: {file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def process_directory(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('glucose.txt'):
                file_path = os.path.join(dirpath, filename)
                remove_first_line_from_file(file_path)
