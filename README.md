# Short Answer Feedback Generation Baseline
**!!! Version 3.0 of this dataset can now be found [Huggingface](https://huggingface.co/Short-Answer-Feedback)!!!**

This repository contains the dataset and code corresponding to the paper *"Your Answer is Incorrect… Would you like to know why? Introducing a Bilingual Short Answer Feedback Dataset"* (Filighera et al., ACL 2022). 
1. The experiments were run with a conda python environment using python version 3.7.10.
   The environment to reproduce our results can be installed in the following way (for windows, uncomment pywin32=300 in requirements.txt):
> conda create --name baseline python=3.7  
> conda activate baseline  
> conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch  
> pip install -r requirements.txt  
> pip install bert-score
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
7. We also provide an example script for inference with a finetuned model (using our code) and a json file that contains a question, a reference answer, and a list of student answers.  
> python inference.py  

You may also adjust the seeding in the litT5.py script.

If you found this code or dataset helpful in your research, please consider citing:
```bibtex
@inproceedings{filighera-etal-2022-answer,
    title = "Your Answer is Incorrect... Would you like to know why? Introducing a Bilingual Short Answer Feedback Dataset",
    author = "Filighera, Anna  and
      Parihar, Siddharth  and
      Steuer, Tim  and
      Meuser, Tobias  and
      Ochs, Sebastian",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.587",
    pages = "8577--8591",
   }
```
