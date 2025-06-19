## MOFClassifier: A Machine Learning Approach for Validating Computation-Ready Metal-Organic Frameworks
                                                                                                                                          
[![Static Badge](https://img.shields.io/badge/arXiv.2506.14845v1-brightgreen?style=flat)](https://arxiv.org/abs/2506.14845)
![GitHub repo size](https://img.shields.io/github/repo-size/mtap-research/MOFClassifier?logo=github&logoColor=white&label=Repo%20Size)
[![PyPI](https://img.shields.io/pypi/v/MOFClassifier?logo=pypi&logoColor=white)](https://pypi.org/project/MOFClassifier?logo=pypi&logoColor=white)
[![Requires Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![GitHub license](https://img.shields.io/github/license/mtap-research/MOFClassifier.svg)](https://github.com/mtap-research/MOFClassifier/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/mtap-research/MOFClassifier.svg)](https://GitHub.com/mtap-research/MOFClassifier/issues/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15654431.svg)](https://doi.org/10.5281/zenodo.15654431)
                                                                     
**Developed by:** [Guobin Zhao](https://github.com/sxm13)                                
                                                                                                         
### Installation 
                                     
```sh
pip install MOFClassifier
```

### Examples                                                                                                     
```python
from MOFClassifier import CLscore
cifid, cl_scores, cl_score = CLscore.predict(root_cif="./example.cif")
```
-  **root_cif**: the path of your structure
-  **cifid**: the name of structure
-  **cl_scores**: the CLscore predicted by 100 models (bags)
-  **cl_score**: the mean CLscore of **cl_scores**
                                                                                
### Citation                                          
**Guobin Zhao**, **Pengyu Zhao** and **Yongchul G. Chung**. 2025. **arXiv.2506.14845**.
