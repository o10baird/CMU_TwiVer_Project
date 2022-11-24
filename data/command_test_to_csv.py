import subprocess
import pandas as pd
import urllib.parse as urlparse
from datetime import date, timedelta
from pathlib import Path

path = Path.cwd()
data_Dir = path.joinpath('new_data')
file_list = []
for file in data_Dir.glob('**/*'):
    if file.is_file():
        command = f"twarc2 csv {file} {file.stem}.csv"
    subprocess.run(command.split())
