import shutil
import subprocess
from tqdm import tqdm
from os import makedirs
from os.path import join as pjoin
from os.path import exists, dirname

import pandas as pd
from scipy.sparse import csr_matrix

from agtool.utils.path_utils import module_path


DIR_BASE = pjoin(dirname(module_path()), 'dataset')
URL_MOVIELENS = 'http://files.grouplens.org/datasets/movielens'
makedirs(DIR_BASE, exist_ok=True)


def _processing(DIR, df):
    mat = csr_matrix((df.rating, (df.uid - 1, df.iid - 1)))
    print(f'Get scipy.csr_matrix, shape={mat.shape}.')
    with open(f'{DIR}/processed/indptr', 'w') as fout:
        for e in tqdm(mat.indptr, desc='Write indptr', ncols=100):
            fout.write(str(e) + '\n')
    with open(f'{DIR}/processed/indices', 'w') as fout:
        for e in tqdm(mat.indices, desc='Write indices', ncols=100):
            fout.write(str(e) + '\n')


def ml_100k():
    data = 'ml-100k'
    DIR = pjoin(DIR_BASE, data)
    if not exists(pjoin(DIR, 'u.data')):
        url = pjoin(URL_MOVIELENS, f'{data}.zip')
        subprocess.run(
            f"wget -O tmp.zip {url} && unzip tmp.zip && rm tmp.zip",
            cwd=DIR_BASE, shell=True, check=True
        )
        makedirs(pjoin(DIR, 'processed'), exist_ok=True)
    elif exists(pjoin(DIR, 'processed', 'indptr')) and \
            exists(pjoin(DIR, 'processed', 'indices')):
        print("Already processed ml-100k.")
        return DIR
    df = pd.read_csv(f'{DIR}/u.data', sep='\t', header=None)
    df.columns = ['uid', 'iid', 'rating', 'timestamp']
    _processing(DIR, df)
    print("Write done.")
    return DIR


def ml_10m():
    data = 'ml-10m'
    DIR = pjoin(DIR_BASE, data)
    if not exists(pjoin(DIR, 'ratings.dat')):
        url = pjoin(URL_MOVIELENS, f'{data}.zip')
        subprocess.run(
            f"wget -O tmp.zip {url} && unzip tmp.zip && rm tmp.zip && mv ml-10M100K ml-10m",
            cwd=DIR_BASE, shell=True, check=True
        )
        makedirs(pjoin(DIR, 'processed'), exist_ok=True)
    elif exists(pjoin(DIR, 'processed', 'indptr')) and \
            exists(pjoin(DIR, 'processed', 'indices')):
        print("Already processed ml-100k.")
        return DIR
    df = pd.read_csv(f'{DIR}/ratings.dat', sep='::', header=None, engine='python')
    df.columns = ['uid', 'iid', 'rating', 'timestamp']
    _processing(DIR, df)
    print("Write done.")
    return DIR


if __name__ == '__main__':
    DIR = ml_100k()
    print(f"Remove {DIR}")
    shutil.rmtree(DIR)
