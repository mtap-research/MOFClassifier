The original code is derived from [Synthesizability-PU-CGCNN](https://github.com/kaist-amsg/Synthesizability-PU-CGCNN) from [prof. Yousung Jung group](https://micc.snu.ac.kr/professor/) and publish on [JACS](https://doi.org/10.1021/jacs.0c07384) in 2020.                                                                                                     
                                                                 
### Installation

`pip install -r requirements.txt`    

### Prepare dataset              
                                        
**generate graph from crystal file (`./data/cifs)**                                             
                                                 
- `structures.cif`: all CIFs used for training  
- `atom_init.json`: a [JSON](https://github.com/txie-93/cgcnn/blob/master/data/sample-regression/atom_init.json) file that stores the initialization vector for each element                      
- `id_prop.csv`: a [CSV](https://github.com/txie-93/cgcnn/blob/master/data/sample-regression/id_prop.csv) file with two columns. The first column recodes a CIF name for each crystal, and the second column recodes the value of target property (CR: 1; unlabeled: 0).

```sh
python generate_crystal_graph.py \
  --cifs ./data/cifs/ \
  --n 12 \
  --r 8 \
  --f ./data/graph/
```                                                                        

see all arguments
```sh
python generate_crystal_graph.py -h
```
                                                                                  
### Trainning 
                                            
**run a training job**                      
                                               
```sh
python main_PU_learning.py \
--bag 100 \
--graph ./data/graph/ \
--cifs ./data/cifs/ \
--resume ./models/ \
--split ./data/split \
-b 32 > ./data/train.log &
```

see all arguments
```sh
python main_PU_learning.py -h
```
                                                                                                               
### Test

**evaluate the trained models**                          

```sh
python predict_PU_learning.py \
--bag 100 \
--graph ./data/graph/ \
--cifs ./data/cifs/ \
--modeldir ./models/ \
--disable-cuda \
-b 32
```

see all arguments                                                         
```sh
python predict_PU_learning.py -h
```                                                                 
