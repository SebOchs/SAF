# Short Answer Feedback Generation Baseline
This repository contains the dataset and code corresponding to the paper *"Your Answer is incorrect... Would you like to know why? Introducing a Bilingual Short Answer Feedback Dataset"* appearing at ACL 2022.
1. The experiments were run with a conda python environment using python version 3.7.10.
   The environment to reproduce our results can be installed in the following way (for linux, comment pywin32=300 in requirements.txt):
> conda create --name baseline python=3.7   
> conda activate baseline  
> conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
> pip install -r requirements.txt
2. Download and copy your data set into the *data* folder or ensure that SAF is there.
<!---
Download and copy the data set into the repository.
-->
3. Run the preprocessing script to preprocess the data set, optionally edit the file path and hyperparameters in the script
> python preprocessing.py
4. Run the finetuning script to train the models. Adjust variable mode to select which experiment setting you want to use, you can also edit hyperparameters here
> python finetuning.py
5. A model can be tested by running the testing script. Adjust model path, language and data paths in the script.
> python testing.py
6. Testing saves the model predictions to a file, which can then be utilized to calculate and print out the BERT score. As file convention, the file should start with the experiment mode.
> python bert_scoring.py

You may also adjust the seeding in the litT5.py script.

If you found this code or dataset helpful in your research, please consider citing:
<!---
Insert bibtex here, once published
-->
