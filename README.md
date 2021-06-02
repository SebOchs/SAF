# KN1-baseline

1. The experiments were run with a conda python environment using python version 3.7.10.
   I built the environment to reproduce our results in the following way (for linux, comment pywin32=300 in requirements.txt):
> conda create --name baseline python=3.7   
> conda activate baseline  
> conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
> pip install -r requirements
> pip install bert-score (forgot to include in requirements)
2. Download and copy the kn1 data set into the repository.
3. Run preprocessing script to preprocess the data set, eventually edit the file path in the script
> python preprocessing.py
4. Run finetuning script to train the models. Adjust variable mode to select if you want to train on scores or verification feedback, you can also edit hyperparameters here
> python finetuning.py
5. A model can be tested by running the testing script. Adjust model path in the script.
> python testing.py
6. Testing saves the model predictions to a file, which can then be utilized to calculate and print out the BERT score. As file convention, the file should start with 'score' or 'ver' as indication for different modes.
> python bert_scoring.py

You may also adjust the seeding in the litT5.py script.

In cooperation with Anna Filighera and the Multimedia Communications Lab at TU Darmstadt.