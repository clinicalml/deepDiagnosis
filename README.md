# deepDiagnosis
A torch package for learning diagnosis models from temporal patient data.

For more details please check:

1) http://arxiv.org/abs/1511.07938 
Narges Razavian, David Sontag, "Temporal Convolutional Neural Networks for Diagnosis from Lab Tests", ICLR 2016 Workshop track. 

2) [link to be updated]
Narges Razavian, Jake Marcus, David Sontag,"Multi-task Prediction of Disease Onsets from Longitudinal Lab Tests", Machine Learning for Healthcare, 2016

#Installation:

The package has the following dependencies:

Python: [Numpy](http://www.scipy.org/scipylib/download.html), [CPickle](https://pymotw.com/2/pickle/)

LUA: [Torch](http://torch.ch/docs/getting-started.html), [cunn](https://github.com/torch/cunn), [nn](https://github.com/torch/nn), [cutorch](https://github.com/torch/cutorch), [gnuplot](https://github.com/torch/gnuplot), [optim](https://github.com/torch/optim), and [rnn](https://github.com/Element-Research/rnn)

#Usage:

Run the following in order. Creating datasets can be done in parallel over train/test/valid tasks. Up to you.

There are sample input files (./sample_python_data) that you can use to test the package first. 


	1) python create_torch_tensors.py --x  sample_python_data/xtrain.pkl --y sample_python_data/ytrain.pkl --task 'train' --outdir ./sampledata/

	2) python create_torch_tensors.py --x sample_python_data/xtest.pkl --y sample_python_data/ytest.pkl --task 'test' --outdir ./sampledata/

	3) python create_torch_tensors.py --x sample_python_data/xvalid.pkl --y sample_python_data/yvalid.pkl --task 'valid' --outdir ./sampledata/


	4) th create_batches.lua --task=train --input_dir=./sampledata --batch_output_dir=./sampleBatchDir 

	5) th create_batches.lua --task=valid --input_dir=./sampledata --batch_output_dir=./sampleBatchDir 

	6) th create_batches.lua --task=scoretrain --input_dir=./sampledata --batch_output_dir=./sampleBatchDir 

	7) th create_batches.lua --task=test --input_dir=./sampledata --batch_output_dir=./sampleBatchDir


	8) th train_and_validate.lua --task=train --input_batch_dir=./sampleBatchDir --save_models_dir=./sample_models/


Once the model is trained, run the following to get final evaluations on test set: (change the "lstm2016_05_29_10_11_01" into the model directory that you have created in step 8. Training directories have timestamp.)


	9) th train_and_validate.lua --task=test --validation_dir=./sample_models/lstm2016_05_29_10_11_01/

Read the following for details on how to define your cohort and task.

#Input: 
Input should be one of the two formats descrubed below:

![Here is an Imaginary input and output for a single person in 2 input setting.](https://github.com/clinicalml/deepDiagnosis/blob/master/doc/input_formats.png)

Read below for the details:

1) Python nympy arrays (also support cPickle) of size 

xtrain, xvalid, xtest: |labs| x |people| x |cohort time| for creating the input batches
	
ytrain, yvalid, ytest: |diseases| x |people| x |cohort time| for creating the output batches and inclusion/exclusion for each batch member


2) Python numpy arrays (also support cPickle) of size

xtrain, xvalid, xtest: |Labs| x |people| x |cohort time| for the output
	
ytrain, yvalid, ytest: |diseases| x |people| for the output, where we do not have a concept of time.


3) *advanced* shelve databases, for our internal use.

Please refer to https://github.com/clinicalml/ckd_progression for details.


#Prediction Models:

Currently the following models are supported. The details of the architectures are included in the citation paper below.

1) Logistic Regression  (--model=max_logit)

![](https://github.com/clinicalml/deepDiagnosis/blob/master/doc/maxlogit.png )

2) Feedforward network  (--model=mlp)

![](https://github.com/clinicalml/deepDiagnosis/blob/master/doc/mlp.png )

3) Temporal Convolutional neural network over a backward window   (--model=convnet) 

![](https://github.com/clinicalml/deepDiagnosis/blob/master/doc/arch1.png )

4) Convolutional neural network over input and time dimension  (--model=convnet_mix)

![](https://github.com/clinicalml/deepDiagnosis/blob/master/doc/conv_arch2.png )

5) Multi-resolution temporal convolutional neural network  (--model=multiresconvnet)

![](https://github.com/clinicalml/deepDiagnosis/blob/master/doc/conv_arch1.png)

6) LSTM network over the backward window  (--model=lstmlast) (note: a version --model=lstmall is also available but we found training with lstmlast gives better results)

![](https://github.com/clinicalml/deepDiagnosis/blob/master/doc/lstm_last.png )

7) Ensamble of multiple models  (to be added soon)


#Synthetic Input for testing the package

You can use the following to create synthetic numpy arrays to test the package;

	python create_synthetic_data.py --outdir ./sample_python_data --N 6000  --D 15 --T 48 --O 20

This code will create 3 datasets (train, test, valid) in the ./sample_python_data directory, with dimensions of: 5 x  2000 x 48 for each input x (xtrain, xtest, xvalid) and 20 x  2000 x  48 for each outcome set y. This synthetic data correcsponds to input type 1 above. Follow steps 1-9 in the (Run) section above to test with this data, and feel free to test with other synthetic datasets.

#Citation: [Will be updated soon]

1) http://arxiv.org/abs/1511.07938 
Narges Razavian, David Sontag, "Temporal Convolutional Neural Networks for Diagnosis from Lab Tests", ICLR 2016 Workshop track. 

2) [link to be updated]
Narges Razavian, Jake Marcus, David Sontag,"Multi-task Prediction of Disease Onsets from Longitudinal Lab Tests", Machine Learning for Healthcare, 2016

#Contact

For any questions please email:
narges razavian [narges.sharif@gmail.com or https://github.com/narges-rzv/]

