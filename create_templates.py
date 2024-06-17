import os
from pathlib import Path

project_name = 'trading_bot'

files_list = [
    '.github/workflows/.gitkeep',
    f'{project_name}/__init__.py',
    f'{project_name}/components/__init__.py',
    f'{project_name}/components/data_ingestion.py',
    f'{project_name}/components/data_validation.py',
    f'{project_name}/components/data_transformation.py',
    f'{project_name}/components/data_validation.py',
    f'{project_name}/components/model_trainer.py',
    f'{project_name}/components/model_evaluation.py',
    f'{project_name}/components/model_pusher.py',
    f'{project_name}/data_access/__init__.py',
    f'{project_name}/data_access/mongo_db_connection.py',
    f'{project_name}/constants/__init__.py',
    f'{project_name}/constants/constants.py',
    f'{project_name}/exception/__init__.py',
    f'{project_name}/exception/exception.py',
    f'{project_name}/logger/__init__.py',
    f'{project_name}/logger/logger.py',
    f'{project_name}/pipeline/__init__.py',
    f'{project_name}/pipeline/training_pipeline.py',
    f'{project_name}/utils/__init__.py',
    f'{project_name}/utils/utils.py',
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    'app.py',
    'requirements.txt',
    'requirements_dev.txt',
    'init_setup.sh',
    'setup.py',
    'setup.cfg',
    'pyproject.toml',
    'tox.ini'
]

for filepath in files_list:

    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as file:
            pass
    else:
        print(f"{filename} is already in {filedir}!!!")