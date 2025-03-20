import logging
from pathlib import Path
import os

list_of_files=[
    '.github/workflows',
    'README.md',
    'requirements.txt',
    'setup.py',
    'app.py',
    'src/__init__.py',
    'src/components/__init__.py',
    'src/constants/__init__.py',
    'src/entity/__init__.py',
    'src/logger/__init__.py',
    'src/configuration/__init__.py',
    'src/pipline/__init__.py',
    
]

for file in list_of_files:
    file = Path(file)

    file_dir,file_name = os.path.split(file)

    if file_dir !="":
        os.makedirs(file_dir,exist_ok=True)
    
    if (not os.path.exists(file)) or (os.path.getsize(file)==0):
        with open(file,'w') as f:
            pass
    