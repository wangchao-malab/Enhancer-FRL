## Enhancer-FRL

### 1. Description
Enhancers are crucial for precisely regulation of gene expression, while enhancers identification and strength prediction is challenging because of their freely distribution and tremendous of similar fractions in genome. Enhancer-FRL, a two-layer predictor proposed for identifying enhancers (enhancers or non-enhancers) and their activities (strong and weak) using the feature representation learning scheme.

### 2. Availability
#### 2.1. Webserver is available at: http://39.100.246.211:10505/DeepSoluE/
http://lab.malab.cn/~wangchao/softwares/Enhancer-FRL/
http://39.100.246.211:10504/

#### 2.2 Datasets and source code are available at:
https://github.com/wangchao-malab/Enhancer-FRL

#### 2.3 Local running
##### 2.3.1 Environment
Before running, please make sure the following packages are installed in Python environment:
gensim==3.4.0
pandas==1.1.3
python==3.7.3
biopython==1.7.8
numpy==1.19.2
scikit-learn==0.22.1
For convenience, we strongly recommended users to install the Anaconda Python 3.7.3 (or above) in your local computer.
##### 2.3.2 Running
Changing working dir to Enhancer-FRL_master, and then running the following command:
python Enhancer-FRL.py -i input.fasta -o results.csv
-i: name of input_file in fasta format   # folder “sequence” is the default file path of the input_file 
-o name of output_file              # folder “results” is the default file path for result save.

Notes: We have set the default working dir to 'C:\Users\Administer\Desktop\Enhancer-FRL_master', Of course, you can change the working dir by fix the Enhancer-FRL.py scripts according your environment.


