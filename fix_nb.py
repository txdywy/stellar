import json, os
os.chdir('/Users/yiwei/stellar')

install_cell = {
    'cell_type': 'code',
    'metadata': {},
    'outputs': [],
    'source': [
        'import subprocess\n',
        "subprocess.run(['pip', 'install', '-q', 'torch==2.4.0', 'torchvision==0.19.0', '--index-url', 'https://download.pytorch.org/whl/cu121'], capture_output=True)\n",
        "print('PyTorch installed')"
    ]
}

for nb_path in ['kaggle_nb_123/realmlp_s123.ipynb', 'kaggle_nb_777/realmlp_s777.ipynb']:
    with open(nb_path) as f:
        nb = json.load(f)
    nb['cells'].insert(1, install_cell)
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'Updated {nb_path}')
