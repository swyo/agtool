# Algorithm Tools

This library serves useful algorithm tools for python. 😄 <br>

Anyone can be install this library from pypi from this link: https://pypi.org/project/agtool/

```
pip install agtool
```

Q. How to manage packages? see from <a href="https://www.youtube.com/watch?v=Motr7UunBT4&list=PLjAFBrXBY3g59hczbnFa-xu1Tqrtzh1Yn&index=1&t=9s" target="_blank"><img src="https://img.shields.io/badge/YouTube-Dol AI-white?style=plastic&logo=youtube&logoColor=red"/></a>


## To Do 

This check lists will be implemented soon. 🔥

- [ ] vanilla vae.
- [ ] plsi, lda model.
- [ ] vanilla gcn.

## Deploy

Deploy to pypi as follows. 🥳
```
# setup.py version up
# doc/conf.py version up
python setup.py bdist_wheel
python -m twine upload dist/*.whl
```

## Documentation

Update documentation using sphinx.
```
sphinx-apidoc -f -o docs agtool
```
And then, `cd docs && make html`.

Serving the documetation.
```
sphinx-autobuild --host [IP] --port [PORT] docs docs/_build/html
```

## Unittest

Install by source build and run pytest.
```
conda env update --file environment.yml --name [ANACONDA ENV NAME]
pip install -r requirements.txt
python setup.py install
conda install pytest
python -m pytest test
```

## Use GPU

If you want to run model with cuda, please install torch as follows by refering [this article](https://pytorch.org/get-started/previous-versions/#commands-for-versions--100). 🚀

E.g., Find proper version from this [website](https://download.pytorch.org/whl/torch_stable.html).
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

If you want to use `torch-sparse, torch-cluster, torch-scatter, torch-spline-curve`, find proper version from this [website](https://data.pyg.org/whl/).
