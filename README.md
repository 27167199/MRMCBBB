# MRMCBBB 
Multi-relational graph neural network with multi-constraints for predicting blood-brain barrier permeability of drugs

## Requirements
  * python==3.8.0
  * torch==1.8.0
  * scikit-learn==1.3.1
  * pandas==2.0.3
  * scipy==1.10.1
  * tqdm==4.66.1
  * numpy==1.24.4
  * urllib3==2.0.5

## File
### data.zip: dataset (unzip it to data), it includes£»
  * graph_drugsim.pt:Drug-protein heterogeneous graph 
  * drug similarity matrix 1.txt: Drug similarity based on chemical-chemical associations
  * drug similarity matrix 2.txt:  Drug similarity based on chemical structures

### model.py Construct the model: multi-relational graph neural network with multi-constraints

### HOCV.py: The code for implementing the Hold-out validation experiment

### 5CV.py: The code for implementing the 5-CV experiment

### 10CV.py: The code for implementing the 10-CV experiment

### utils.py: The code for providing some common functions and classes

## Contact
If you have any questions or comments, please feel free to email Cheng Yan(yancheng01@hnucm.edu.cn).