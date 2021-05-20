# KN1-baseline

1. Install the necessary libraries with: 
> pip install -r requirements
2. Download and copy the kn1 data set into the repository.
3. Run preprocessing script to preprocess the data set, eventually edit the file path in the script
> python preprocessing.py
4. Run finetuning script to train the models. Adjust variable mode to select if you want to train on scores or verification feedback
> python finetuning.py
5. A model can be tested by running the testing script. Adjust model path in the script.
> python testing.py
6. Testing saves the model predictions to a file, which can then be utilized to calculate and print out the BERT score. As file convention, the file should start with 'score' or 'ver' as indication for different modes.
> python bert_scoring.py

In cooperation with Anna Filighera and the Multimedia Communications Lab at TU Darmstadt.