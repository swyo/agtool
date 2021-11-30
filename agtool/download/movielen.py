import shutil
import subprocess
from os import mkdir
from os.path import exists
from os.path import join as pjoin

import pandas as pd
from scipy.sparse import csr_matrix

from agtool.utils.path_utils import module_path


DIR_BASE = module_path('download')


def ml_100k():
    data = 'ml-100k'
    URL_MOVIELENS = 'http://files.grouplens.org/datasets/movielens'
    DIR = pjoin(DIR_BASE, data)
    data = 'ml-100k'
    if not exists(DIR):
        url = pjoin(URL_MOVIELENS, f'{data}.zip')
        subprocess.run(
            f"wget -O tmp.zip {url} && unzip tmp.zip && rm tmp.zip",
            cwd=DIR_BASE, shell=True, check=True
        )
    if not exists(pjoin(DIR, 'processed')):
        mkdir(pjoin(DIR, 'processed'))
    elif exists(pjoin(DIR, 'processed', 'indptr')) and \
            exists(pjoin(DIR, 'processed', 'indices')):
        print("Already processed ml-100k.")
        return DIR
    df = pd.read_csv(f'{DIR}/u.data', sep='\t', header=None)
    df.columns = ['uid', 'iid', 'rating', 'timestamp']
    mat = csr_matrix((df.rating, (df.uid - 1, df.iid - 1)))
    print(f'Get scipy.csr_matrix, shape={mat.shape}.')
    with open(f'{DIR}/processed/indptr', 'w') as fout:
        for e in mat.indptr:
            fout.write(str(e) + '\n')
    with open(f'{DIR}/processed/indices', 'w') as fout:
        for e in mat.indices:
            fout.write(str(e) + '\n')
    print("Write done.")
    return DIR


if __name__ == '__main__':
    DIR = ml_100k()
    print(f"Remove {DIR}")
    shutil.rmtree(DIR)
