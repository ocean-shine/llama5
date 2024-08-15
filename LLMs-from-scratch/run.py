import os
import json
import re

def update_width_in_ipynb(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        return

    updated = False
    for cell in content.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            for i, line in enumerate(cell.get('source', [])):
                new_line = re.sub(r'width=\"[^"]*\"', 'width=\"2500px\"', line)
                if new_line != line:
                    cell['source'][i] = new_line
                    updated = True

    if updated:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(content, file, ensure_ascii=False, indent=1)

def update_all_ipynb_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.ipynb'):
                file_path = os.path.join(dirpath, filename)
                update_width_in_ipynb(file_path)

# 指定根目录
root_directory = '/home/ocean/Code/llama5/LLMs-from-scratch'
update_all_ipynb_files(root_directory)

