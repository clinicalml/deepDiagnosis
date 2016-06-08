# deepDiagnosis
A torch package for learning diagnosis models from temporal patient data.

For more details please check http://arxiv.org/abs/1511.07938 

Narges Razavian, David Sontag, "Temporal Convolutional Neural Networks for Diagnosis from Lab Tests", ICLR 2016 Workshop track. (To be updatd later with new citation)

----------------------------------------------------
#Installation:

The package has the following dependencies:

Python: [Numpy](http://www.scipy.org/scipylib/download.html), [CPickle](https://pymotw.com/2/pickle/)

LUA: [Torch](http://torch.ch/docs/getting-started.html), [cunn](https://github.com/torch/cunn), [nn](https://github.com/torch/nn), [cutorch](https://github.com/torch/cutorch), [gnuplot](https://github.com/torch/gnuplot), [optim](https://github.com/torch/optim), and [rnn](https://github.com/Element-Research/rnn)

----------------------------------------------------
#Usage:


Run the following in order. Creating datasets can be done in parallel over train/test/valid tasks. Up to you.

There are sample input files (./sample_python_data) that you can use to test the package first. 


	1) python create_torch_tensors.py --x  sample_python_data/xtrain.npy --y sample_python_data/ytrain.npy --task 'train' --outdir ./sampledata/

	2) python create_torch_tensors.py --x sample_python_data/xtest.npy --y sample_python_data/ytest.npy --task 'test' --outdir ./sampledata/

	3) python create_torch_tensors.py --x sample_python_data/xvalid.npy --y .ample_python_data/yvalid.npy --task 'valid' --outdir ./sampledata/

	4) th create_batches.lua --task=train --input_dir=./sampledata --batch_output_dir=./sampleBatchDir 

	5) th create_batches.lua --task=valid --input_dir=./sampledata --batch_output_dir=./sampleBatchDir 

	6) th create_batches.lua --task=scoretrain --input_dir=./sampledata --batch_output_dir=./sampleBatchDir 

	7) th create_batches.lua --task=test --input_dir=./sampledata --batch_output_dir=./sampleBatchDir


	8) th train_and_validate.lua --task=train --input_batch_dir=./sampleBatchDir --save_models_dir=./sample_models/


Once the model is trained, run the following to get final evaluations on test set: (change the "lstm2016_05_29_10_11_01" into the model directory that you have created in step 8. Training directories have timestamp.)


	9) th train_and_validate.lua --task=test --validation_dir=./sample_models/lstm2016_05_29_10_11_01/

Read the following for details on how to define your cohort and task.

----------------------------------------------------
#Input: 

The package has the following options for input cohort.

----

1) Python nympy arrays (also support cPickle) of size 

xtrain, xvalid, xtest: |labs| x |people| x |cohort time| for creating the input batches
	
ytrain, yvalid, ytest: |diseases| x |people| x |cohort time| for creating the output batches and inclusion/exclusion for each batch member

[Here is an Imaginary input and output for a single person in this input setting.](https://github.com/clinicalml/deepDiagnosis/blob/master/doc/overview.pdf "An imaginary input and output for a single person.")If you don't have the temporal aspect on the output, see option 2 for the input below.

----
	
2) Python numpy arrays (also support cPickle) of size

xtrain, xvalid, xtest: |Labs| x |people| x |cohort time| for the output
	
ytrain, yvalid, ytest: |diseases| x |people| for the output, where we do not have a concept of time.

---

3) *advanced* shelve databases, for our internal use.

Please refer to https://github.com/clinicalml/ckd_progression for details.

----------------------------------------------------

#Prediction Models:

Currently the following models are supported. The details of the architectures are included in the citation paper below.


0) Logistic Regression  (--model=max_logit)

1) Feedforward network  (--model=mlp)

2) Temporal Convolutional neural network over a backward window   (--model=convnet) 

3) Convolutional neural network over input and time dimension  (--model=convnet_mix)  [Overview](https://github.com/clinicalml/deepDiagnosis/blob/master/doc/conv_arch2.pdf "Overview")

4) Multi-resolution temporal convolutional neural network  (--model=multiresconvnet)
[Overview](https://github.com/clinicalml/deepDiagnosis/blob/master/doc/conv_arch1.pdf "Overview")

5) LSTM network over the backward window  (--model=lstmlast) (note: a version --model=lstmall is also available but we found training with lstmlast gives better results)
[Overview](https://github.com/clinicalml/deepDiagnosis/blob/master/doc/lstm_last.pdf "Overview")

6) Ensamble of multiple models  (to be added soon)


----------------------------------------------------
#Synthetic Input for testing the package

You can use the following to create synthetic numpy arrays to test the package;

	python create_synthetic_data.py --outdir ./sample_python_data --N 6000  --D 15 --T 48 -O 20

This code will create 3 datasets (train, test, valid) in the ./sample_python_data directory, with dimensions of: 5 x  2000 x 48 for each input x (xtrain, xtest, xvalid) and 20 x  2000 x  48 for each outcome set y. This synthetic data correcsponds to input type 1 above. Follow steps 1-9 in the (Run) section above to test with this data, and feel free to test with other synthetic datasets.

----------------------------------------------------
#Intermediate data

Given the input, create_batches.lua creates mini-batches and save them on disk. These files are created:

1) bix_*_batch_input

size: (batchSize, 1, labcounts, backward_window)

Description: Tensor of the input. Values could be binary (i.e. if disease history, medications)  or continuous (observed lab values)

2) bix_*batch_input_nnx

size:(batchSize, 1, labcounts, backward_window)

Description: Tensor of a binary indicator for whether or not the input was observed.

3) bix_*_batch_target

size: (batchSize, diseasecount, 1, 1)

Description: Tensor of binary indicators for outcomes.

4) bix_*_batch_tobe_excluded_outcomes 

size: (batchSize, diseasecount, 1, 1)

Description: Tensor of binary indicators for whether or not each member in the batch would need to be excluded for each outcome. (We have the option to exclude people who already have an outcome)

5) bix_*_batch_mu

size: (batchSizeTrain, 1, labcounts, backward_window)

Description: Tensor that includes per person per lab, the mean value. This is useful if imputation is used and we want to normalize each time series before imputing. We store the means so that we can scale them back after imputation.

6) bix_*_batch_std

size: (batchSizeTrain, 1, labcounts, backward_window)

Description: Tensor that includes per person per lab, the standard deviation value. This is useful if imputation is used and we want to normalize each time series before imputing. We store the standard deviations so that we can scale them back after imputation.


----------------------------------------------------

#Citation: [Will be updated soon]

Narges Razavian, David Sontag, "Temporal Convolutional Neural Networks 
for Diagnosis from Lab Tests", ICLR 2016 Workshop track.
Link: http://arxiv.org/abs/1511.07938 

----------------------------------------------------
#Contact

For any questions please email:

narges razavian [narges.sharif@gmail.com or https://github.com/narges-rzv/]

