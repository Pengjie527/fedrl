import os

for root, dirs, files in os.walk('.'):
    for dir in dirs:
        if dir == '__pycache__':
            full_path = os.path.join(root, dir)
            print(f"Deleting: {full_path}")
            os.system(f'rmdir /s /q "{full_path}"')
