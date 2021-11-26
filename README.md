# Algorithm Tools

This library serves useful algorithm tools for python. 😄 <br>

Anyone can be install this library from pypi from this link: https://pypi.org/project/agtool/

```
pip install agtool
```

Q. How to manage packages? see from <a href="https://www.youtube.com/watch?v=Motr7UunBT4&list=PLjAFBrXBY3g59hczbnFa-xu1Tqrtzh1Yn&index=1&t=9s" target="_blank"><img src="https://img.shields.io/badge/YouTube-Dol AI-white?style=plastic&logo=youtube&logoColor=red"/></a>


## To Do 

This check lists will be implemented soon. 🔥

- [ ] negative sampling: draw negative samples from rating matrix.


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
sphinx-apidoc -f -o docs altools
```

Serving the documetation.
```
sphinx-autobuild --host [IP] --port [PORT] docs docs/_build/html
```
