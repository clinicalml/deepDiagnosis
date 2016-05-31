------------------------------------------------------------------------------------------------
-- If using this code or its variant please cite:
-- Narges Razavian, David Sontag, "Temporal Convolutional Neural Networks 
-- for Diagnosis from Lab Tests", ICLR 2016 Workshop track.
-- Link: http://arxiv.org/abs/1511.07938 
-- For questions about the code contact Narges (narges.sharif@gmail.com)
-----------------------------------------------------------------------------------------------

require 'cunn'
require 'cutorch'
require 'rnn'
require 'paths'
require 'optim'
require 'gnuplot'
ROC = dofile('roc.lua') 
dofile('CDivTable_rebust.lua')
require 'os'

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(2)

local opt = lapp[[
	--gpuid           		(default 1),
	--input_batch_dir 		(default './sampleBatchDir')    this directory needs to have train, test, valid and scoretrain subdirectories.
	--save_models_dir 		(default './sampleModel')		per epoch one model will be saved here.
	--outcome_label_file	(default '')					labels for outcomes: One line per label.
	--model 		  		(default 'lstmlast')			model should be one of these: convnet|convnet_mix|multiresconvnet|lstmlast|mlp|max_logit
	--task			  		(default 'train')				task can be either train or test. If test, validation_dir should point to the directory that the models were stored.
	--maxEpoch				(default 4000),
	--verbose				(default 0),
	--exclude_already_onset (default 1),
	--imputation_type    	(default 'none'),
	--scale_imputed_data   	(default 1),
	--autoweight 	  		(default 1),
	--use_batch_normalization (default 1),	
	--learningRate  		(default 0.1),
	--learningRateDecay 	(default 0.01),
	--dropout_prob 			(default 0.5),
	--momentum 				(default 0.01),
	--nesterov 				(default 0),
	--weightDecay 			(default 0.1),
	--validation_dir  		(default ''),

	--conv_prediction_depth 	(default  2),
	--conv_num_features 		(default {64, 64}), 
	--conv_k_size_horiz 		(default {12, 3}),
	--conv_k_size_horiz_stride 	(default {1, 1}),
	--conv_k_size_vert  		(default {1, 1}),
	--batch_normalization_epsilon (default  0.0001),
	--conv_pool_horizontal 		(default {3, 3}),
	--conv_pool_horizontal_stride (default {1, 1}),
	--number_hidden_nodes_in_shared_net (default 100),

	--num_filters_temporal_conv (default 64),
	--num_filters_labmix 		(default 64),
	--ncn_kernel_width 			(default  {1, 12}),
	--ncn_kernel_horiz_stride 	(default  {1, 1}),
	--ncn_max_pool_horiz 		(default  3),
	--ncn_max_pool_horiz_stride (default  3),

	--resolution_1_poolwidth 	(default {3, 3}),
	--resolution_1_poolstride 	(default {3, 3}),
	--resolution_1_convkwidth 	(default {3}),
	--resolution_1_convkstride 	(default {1}),

	--resolution_2_poolwidth 	(default {3}),
	--resolution_2_poolstride 	(default {3}),
	--resolution_2_convkwidth 	(default {6}),
	--resolution_2_convkstride 	(default {1}),

	--resolution_3_convkwidth 	(default {6, 6}),
	--resolution_3_convkstride 	(default {1, 1}),
	--resolution_3_poolwidth 	(default {3}),
	--resolution_3_poolstride 	(default {3}),

	--lstm_depth 				(default  2),
	--lstmhiddenSize 			(default 500),

	--resolution_depth 			(default  3) ,
	--num_of_multires_filters 	(default  64),
	--depth_hidden_multires 	(default  2),
	--mlp_depth_shared 			(default  2),
	--mlp_layer_width_shared 	(default  {80, 80})

]]

cutorch.setDevice(opt.gpuid)
print(opt)

function init( )
	learningRate = opt.learningRate
	learningRateDecay = opt.learningRateDecay
	rootbatches = opt.input_batch_dir	
	validation_dir = opt.validation_dir

	load_pretrain_train_network_dir = opt.pretrain_dir
	save_train_network_dir = opt.save_models_dir .. '/' .. opt.model..string.gsub(os.date("%Y_%m_%d_%X"),':','_')

	os.execute('mkdir -p '.. save_train_network_dir ..'/')
	os.execute('cp ./train_and_validate.lua ' .. save_train_network_dir .. '/')
	option_file_to_save = io.open(save_train_network_dir..'/options_cmd.txt', "a")	
	for k,v in pairs(opt) do 
		option_file_to_save:write(k .. ':' .. v .. '\n') 
	end
	option_file_to_save:flush()
	option_file_to_save:close()

	batches_network_dir = rootbatches .. '/'.. opt.task ..'/'
	batches_network_dir_valid = rootbatches .. '/valid/'
	batches_network_dir_scoretrain = rootbatches .. '/scoretrain/'
	batches_network_dir_test = rootbatches .. '/test/'
	
	print('Checking the input dir:' .. batches_network_dir)
	batch_0_output = torch.load(batches_network_dir..'bix1_batch_target')
	batch_0_input = torch.load(batches_network_dir..'bix1_batch_input')
	print ('Succss!')
	batchSize = batch_0_input:size(1)	 
	backward_window = batch_0_input:size(4) 
	labcounts = batch_0_input:size(3)
	diseasecount = batch_0_output:size(2)
	print ('batchsize:'.. batchSize .. '\nbackward_window:'..backward_window..'\ninput signals:'..labcounts..'\noutcome counts:'..diseasecount)

	print ('loading outcome labels from' .. opt.outcome_label_file)
	diseaseLabels = {}
	if (opt.outcome_label_file ~= '') then
		labelsFile  = io.open(opt.outcome_label_file)
		local line = labelsFile:read("*l")	
		while (line ~= nil) do
			table.insert(diseaseLabels,line)		
			line = labelsFile:read("*l")
		end
		labelsFile:close()
	else
		for i = 1, diseasecount do
			table.insert(diseaseLabels,"outcome "..i)
		end
	end
	if (opt.verbose == 1) then
		print(diseaseLabels)
	end
	print('done')
	
	--load the labels later...
	batch_lists = scandir(batches_network_dir..'/bix*_batch_input_nnx')	
	totalBatchCntTrain = (#batch_lists)
	print('total of '.. totalBatchCntTrain .. ' batches available for task: '.. opt.task)

	batch_lists_valid = scandir(batches_network_dir_valid..'/bix*')
	totalBatchCntValid = (#batch_lists_valid)/6

	batch_lists_scrTrain = scandir(batches_network_dir_scoretrain..'/bix*')
	totalBatchCntScrTrain = (#batch_lists_scrTrain)/6

	batch_lists_test = scandir(batches_network_dir_test..'bix*')
	totalBatchCntScrTest = (#batch_lists_test)/6

	---
	print('Loading outcome batches to compute weights for weighted log likelihood. This may take a few seconds...')
	read_disease_frequencies()
	print('done')
	---

	maxEpoch = opt.maxEpoch
	imputation_type = opt.imputation_type
	scale_imputed_data = opt.scale_imputed_data
	autoweight = opt.autoweight
	use_batch_normalization = opt.use_batch_normalization
	exclude_already_onset = opt.exclude_already_onset --previously 0
	
	loss_train_values = {0}
	loss_mini_validate_values = {0}
	loss_mini_train_values = {0}

	----shared hyper-parameter for all networks---------------------------------------
	number_hidden_nodes_in_shared_net = {opt.number_hidden_nodes_in_shared_net}
	dropout_prob = opt.dropout_prob
	batch_normalization_epsilon = opt.batch_normalization_epsilon

	----Convolution---------------------------------------
	conv_prediction_depth = opt.conv_prediction_depth -- the depth of the convolution network
	conv_num_features = parse_option(opt.conv_num_features)
	conv_k_size_horiz = parse_option(opt.conv_k_size_horiz)  --length of filters 
	conv_k_size_horiz_stride = parse_option(opt.conv_k_size_horiz_stride) 
	conv_k_size_vert  = parse_option(opt.conv_k_size_vert)
	conv_pool_horizontal =  parse_option(opt.conv_pool_horizontal)
	conv_pool_horizontal_stride = parse_option(opt.conv_pool_horizontal_stride)
	
	----Vertical Convolution------------------------------
	num_filters_temporal_conv = opt.num_filters_temporal_conv
	num_filters_labmix = opt.num_filters_labmix

	ncn_n_filters = {num_filters_labmix, num_filters_labmix, num_filters_temporal_conv}
	ncn_kernel_width = parse_option(opt.ncn_kernel_width)
	ncn_kernel_height = {labcounts, 1}
	ncn_kernel_horiz_stride = parse_option(opt.ncn_kernel_horiz_stride)
	ncn_max_pool_horiz = opt.ncn_max_pool_horiz
	ncn_max_pool_horiz_stride = opt.ncn_max_pool_horiz_stride

	----Multiresolution Convolution-----------------------
	resolution_1_poolwidth = parse_option(opt.resolution_1_poolwidth)
	resolution_1_poolstride = parse_option(opt.resolution_1_poolstride)
	resolution_1_convkwidth = parse_option(opt.resolution_1_convkwidth)
	resolution_1_convkstride = parse_option(opt.resolution_1_convkstride)

	resolution_2_poolwidth = parse_option(opt.resolution_2_poolwidth)
	resolution_2_poolstride = parse_option(opt.resolution_2_poolstride)
	resolution_2_convkwidth = parse_option(opt.resolution_2_convkwidth)
	resolution_2_convkstride = parse_option(opt.resolution_2_convkstride)

	resolution_3_convkwidth = parse_option(opt.resolution_3_convkwidth)
	resolution_3_convkstride = parse_option(opt.resolution_3_convkstride)
	resolution_3_poolwidth = parse_option(opt.resolution_3_poolwidth)
	resolution_3_poolstride = parse_option(opt.resolution_3_poolstride)

	-----LSTM -------------------------------------------
	lstm_depth = opt.lstm_depth
	lstmhiddenSize = opt.lstmhiddenSize

	-----Feedforward (MLP)-------------------------------
	resolution_depth = opt.resolution_depth -- how many different resolutions do we build. 
	num_of_multires_filters = opt.num_of_multires_filters --for multi resolution convnet model
	depth_hidden_multires = opt.depth_hidden_multires --for multi resolution convnet model
	mlp_depth_shared = opt.mlp_depth_shared
	mlp_layer_width_shared = parse_option(opt.mlp_layer_width_shared)

end

function read_disease_frequencies()
	print('Processing total batches for estimating weight and class frequency of each outcome:' .. totalBatchCntScrTrain)
	disease_cnter_pos = torch.CudaTensor(diseasecount):fill(0)
	disease_cnter_all = torch.CudaTensor(diseasecount):fill(0)
	class_weight = torch.CudaTensor(2, diseasecount):fill(1)

	for bcntr = 1, totalBatchCntScrTrain do		
		batch_target = torch.load(batches_network_dir_scoretrain..'bix'..bcntr..'_batch_target'):cuda()
		batch_tobe_excluded_outcomes = torch.load(batches_network_dir_scoretrain..'bix'..bcntr..'_batch_tobe_excluded_outcomes'):cuda()		
		disease_cnter_pos = disease_cnter_pos + (torch.cmul(batch_target, batch_tobe_excluded_outcomes:ne(1))):clone():sum(1):clone():view(diseasecount):clone()
		disease_cnter_all = disease_cnter_all + (batch_tobe_excluded_outcomes:ne(1)):sum(1):clone():view(diseasecount):clone()
	end
	disease_freqs = torch.cdiv(disease_cnter_pos, disease_cnter_all)
	disease_weight = disease_freqs / (disease_freqs:sum())
	for i = 1, diseasecount do
		if (opt.verbose == 1) then
			print (diseaseLabels[i] ..' | '..disease_cnter_pos[i] ..' '..disease_cnter_all[i] ..' '..  disease_freqs[i] .. ' ' .. disease_weight[i])
		end
		if (autoweight == 1) then
			class_weight[{{1},{i}}] = disease_cnter_pos[i]/disease_cnter_all[i] --negative class is weighted by frequency of positive class
			class_weight[{{2},{i}}] = 1 - (disease_cnter_pos[i]/disease_cnter_all[i]) --positive class is weighted by frequency of negative class
		end
	end
end

function tofile(fname, x) --Credit for this function goes to Jure Zbontar
   tfile = torch.DiskFile(fname .. '.type', 'w')
   if x:type() == 'torch.FloatTensor' then
      tfile:writeString('float32')
      torch.DiskFile(fname, 'w'):binary():writeFloat(x:storage())
   elseif x:type() == 'torch.LongTensor' then
      tfile:writeString('int64')
      torch.DiskFile(fname, 'w'):binary():writeLong(x:storage())
   elseif x:type() == 'torch.IntTensor' then
      tfile:writeString('int32')
      torch.DiskFile(fname, 'w'):binary():writeInt(x:storage())
   end
   dimfile = torch.DiskFile(fname .. '.dim', 'w')
   for i = 1,x:dim() do
      dimfile:writeString(('%d\n'):format(x:size(i)))
   end
end

function save_data_for_logit_python_train()
	big_x = torch.CudaTensor(totalBatchCntScrTrain*batchSize, labcounts, backward_window ):fill(0)
	big_y = torch.CudaTensor(totalBatchCntScrTrain*batchSize, diseasecount):fill(0)
	big_y_exclusionflag = torch.CudaTensor(totalBatchCntScrTrain*batchSize, diseasecount):fill(0)

	for bcntr = 1, totalBatchCntScrTrain do
		print (bcntr)		
		batch_target = torch.load(batches_network_dir_scoretrain..'bix'..bcntr..'_batch_target'):cuda()
		batch_tobe_excluded_outcomes = torch.load(batches_network_dir_scoretrain..'bix'..bcntr..'_batch_tobe_excluded_outcomes'):cuda()		
		batch_input = torch.load(batches_network_dir_scoretrain..'bix'..bcntr..'_batch_input'):cuda()
		big_x[{{(bcntr-1)*batchSize+1, bcntr*batchSize}, {}, {}}] = batch_input:view(batchSize, labcounts, backward_window):clone()
		big_y[{{(bcntr-1)*batchSize+1, bcntr*batchSize}, {}}] = batch_target:view(batchSize, diseasecount):clone()
		big_y_exclusionflag[{{(bcntr-1)*batchSize+1, bcntr*batchSize}, {}}] = batch_tobe_excluded_outcomes:view(batchSize, diseasecount):clone()
	end
	print('saving train')
	filename = paths.concat('xtrain_for_python.np')	
	os.execute('mkdir -p ' .. sys.dirname(filename))
	tofile(filename, big_x:float())
	filename = paths.concat('ytrain_for_python.np')	
	os.execute('mkdir -p ' .. sys.dirname(filename))
	tofile(filename, big_y:float())
	filename = paths.concat('ymask_for_python.np')
	os.execute('mkdir -p ' .. sys.dirname(filename))
	tofile(filename, big_y_exclusionflag:float())
	os.exit()
end

function save_data_for_logit_python_test()
	big_x = torch.CudaTensor(totalBatchCntScrTest*batchSize, labcounts, backward_window ):fill(0)
	big_y = torch.CudaTensor(totalBatchCntScrTest*batchSize, diseasecount):fill(0)
	big_y_exclusionflag = torch.CudaTensor(totalBatchCntScrTest*batchSize, diseasecount):fill(0)

	for bcntr = 1, totalBatchCntScrTest do
		print (bcntr)		
		batch_target = torch.load(batches_network_dir_test..'bix'..bcntr..'_batch_target'):cuda()
		batch_tobe_excluded_outcomes = torch.load(batches_network_dir_test..'bix'..bcntr..'_batch_tobe_excluded_outcomes'):cuda()		
		batch_input = torch.load(batches_network_dir_test..'bix'..bcntr..'_batch_input'):cuda()
		big_x[{{(bcntr-1)*batchSize+1, bcntr*batchSize}, {}, {}}] = batch_input:view(batchSize, labcounts, backward_window):clone()
		big_y[{{(bcntr-1)*batchSize+1, bcntr*batchSize}, {}}] = batch_target:view(batchSize, diseasecount):clone()
		big_y_exclusionflag[{{(bcntr-1)*batchSize+1, bcntr*batchSize}, {}}] = batch_tobe_excluded_outcomes:view(batchSize, diseasecount):clone()
	end
	print('saving test')
	filename = paths.concat('xtest_for_python.np')	
	os.execute('mkdir -p ' .. sys.dirname(filename))
	tofile(filename, big_x:float())
	filename = paths.concat('ytest_for_python.np')	
	os.execute('mkdir -p ' .. sys.dirname(filename))
	tofile(filename, big_y:float())
	filename = paths.concat('ytestmask_for_python.np')
	os.execute('mkdir -p ' .. sys.dirname(filename))
	tofile(filename, big_y_exclusionflag:float())
	os.exit() 
end

function save_data_for_logit_python_validate()
	big_x = torch.CudaTensor(totalBatchCntValid*batchSize, labcounts, backward_window ):fill(0)
	big_y = torch.CudaTensor(totalBatchCntValid*batchSize, diseasecount):fill(0)
	big_y_exclusionflag = torch.CudaTensor(totalBatchCntValid*batchSize, diseasecount):fill(0)

	for bcntr = 1, totalBatchCntValid do
		print (bcntr)		
		batch_target = torch.load(batches_network_dir_valid..'bix'..bcntr..'_batch_target'):cuda()
		batch_tobe_excluded_outcomes = torch.load(batches_network_dir_valid..'bix'..bcntr..'_batch_tobe_excluded_outcomes'):cuda()		
		batch_input = torch.load(batches_network_dir_valid..'bix'..bcntr..'_batch_input'):cuda()
		big_x[{{(bcntr-1)*batchSize+1, bcntr*batchSize}, {}, {}}] = batch_input:view(batchSize, labcounts, backward_window):clone()
		big_y[{{(bcntr-1)*batchSize+1, bcntr*batchSize}, {}}] = batch_target:view(batchSize, diseasecount):clone()
		big_y_exclusionflag[{{(bcntr-1)*batchSize+1, bcntr*batchSize}, {}}] = batch_tobe_excluded_outcomes:view(batchSize, diseasecount):clone()
	end
	print('saving validation set')
	filename = paths.concat('xvalid_for_python.np')	
	os.execute('mkdir -p ' .. sys.dirname(filename))
	tofile(filename, big_x:float())
	filename = paths.concat('yvalid_for_python.np')	
	os.execute('mkdir -p ' .. sys.dirname(filename))
	tofile(filename, big_y:float())
	filename = paths.concat('yvalidmask_for_python.np')
	os.execute('mkdir -p ' .. sys.dirname(filename))
	tofile(filename, big_y_exclusionflag:float())
	os.exit() 
end

function build_model() --we are releasing the prediction model at the moment. There will be afuture imputation model as well.	
    big_prediction_model = nn.Sequential()
	--------------------  part 1 of the network is shared between tasks (model_features_shared) -----------------
    model_features_shared = nn.Sequential()    
    effective_backward_window = backward_window
    current_layer_width = backward_window
    vertical_dimension = labcounts
    
    if (opt.model == 'convnet') then
        for depth_ix = 1, conv_prediction_depth do
        	if depth_ix == 1 then
        		prev_number_of_features = 1        		
        	else
        		prev_number_of_features =  conv_num_features[depth_ix-1]
        	end
        	local convolution_module = nn.SpatialConvolutionMM(prev_number_of_features, conv_num_features[depth_ix], conv_k_size_horiz[depth_ix], conv_k_size_vert[depth_ix], conv_k_size_horiz_stride[depth_ix], 1, 0, 0)
        	model_features_shared:add(convolution_module)
        	model_features_shared:add(nn.ReLU())
        	current_layer_width = 1 + math.floor((current_layer_width - conv_k_size_horiz[depth_ix])/conv_k_size_horiz_stride[depth_ix])
        	effective_backward_window = 1 + math.floor((effective_backward_window - conv_k_size_horiz[depth_ix]) / conv_k_size_horiz_stride[depth_ix])

	        if use_batch_normalization == 1 then
	        	model_features_shared:add(nn.SpatialBatchNormalization(conv_num_features[depth_ix], batch_normalization_epsilon))
	     	end	        
	        --- pooling ---
	        local pad_width = 0; 
	        model_features_shared:add(nn.SpatialMaxPooling(conv_pool_horizontal[depth_ix], 1, conv_pool_horizontal_stride[depth_ix], 1, 0, 0 )) --pad_width
	        current_layer_width = 1+ math.floor((current_layer_width - conv_pool_horizontal[depth_ix]) / conv_pool_horizontal_stride[depth_ix])
	        effective_backward_window = 1+ math.floor((effective_backward_window - conv_pool_horizontal[depth_ix]) / conv_pool_horizontal_stride[depth_ix])
	        print(' model convnet at depth:'..depth_ix .. ' prev_layer_length:' ..current_layer_width ..' effective_backward_window:'..effective_backward_window)        	
	    end

	    model_features_shared:add(nn.Reshape(conv_num_features[conv_prediction_depth] * vertical_dimension * effective_backward_window))
		model_features_shared:add(nn.Dropout(dropout_prob))
		model_features_shared:add(nn.Linear(conv_num_features[conv_prediction_depth] * vertical_dimension * effective_backward_window, number_hidden_nodes_in_shared_net[1]))
		model_features_shared:add(nn.ReLU())

		size_of_shared_network_output = number_hidden_nodes_in_shared_net[1]
	end

	if (opt.model == 'convnet_mix') then
		------	 conv 1 across input dimensions
		model_features_shared:add(nn.SpatialConvolutionMM(1, ncn_n_filters[1], 1, labcounts, ncn_kernel_horiz_stride[1], 1, 0, 0))
		model_features_shared:add(nn.ReLU())		
		if use_batch_normalization == 1 then
        	model_features_shared:add(nn.SpatialBatchNormalization(ncn_n_filters[1], batch_normalization_epsilon))
     	end
		model_features_shared:add(nn.Reshape(1, ncn_n_filters[1], backward_window))
		------ conv 2 across conv1 filter outputs (not temporal yet)
		model_features_shared:add(nn.SpatialConvolutionMM(1, ncn_n_filters[2], 1, ncn_n_filters[1], ncn_kernel_horiz_stride[1], 1, 0, 0))
		model_features_shared:add(nn.ReLU())		
		if use_batch_normalization == 1 then
        	model_features_shared:add(nn.SpatialBatchNormalization(ncn_n_filters[2], batch_normalization_epsilon))
     	end
		model_features_shared:add(nn.Reshape(1, ncn_n_filters[2], backward_window))		
		------ temporal pooling
		model_features_shared:add(nn.SpatialMaxPooling(ncn_max_pool_horiz, 1, ncn_max_pool_horiz_stride, 1, 0, 0 ))
		effective_backward_window = torch.ceil(backward_window/ncn_max_pool_horiz)
		------ temporal convolution
		model_features_shared:add(nn.SpatialConvolutionMM(1, ncn_n_filters[3], ncn_kernel_width[2], ncn_kernel_height[2], ncn_kernel_horiz_stride[2], 1, 0, 0))
		model_features_shared:add(nn.ReLU())
		if use_batch_normalization == 1 then
        	model_features_shared:add(nn.SpatialBatchNormalization(ncn_n_filters[3], batch_normalization_epsilon))
     	end
		effective_backward_window = effective_backward_window - (ncn_kernel_width[2]) + 1		
		size_of_shared_network_output = ncn_n_filters[2] * ncn_n_filters[3] * effective_backward_window
	    model_features_shared:add(nn.Reshape(size_of_shared_network_output))
	    ------ hidden layer (still shared)
	    model_features_shared:add(nn.Linear(size_of_shared_network_output, number_hidden_nodes_in_shared_net[1]))
		model_features_shared:add(nn.ReLU())
		-- model_features_shared:add(nn.Dropout(dropout_prob))

		size_of_shared_network_output = number_hidden_nodes_in_shared_net[1]
	end

	if (opt.model == 'lstmlast') then
		model_features_shared:add(nn.Sequencer(nn.Linear(labcounts, lstmhiddenSize)))
		for i = 1, lstm_depth do
			model_features_shared:add(nn.Sequencer(nn.LSTM(lstmhiddenSize, lstmhiddenSize)))
		end
		model_features_shared:add(nn.SelectTable(backward_window))

		model_features_shared:add(nn.Linear(lstmhiddenSize, number_hidden_nodes_in_shared_net[1]))
		model_features_shared:add(nn.ReLU())
		model_features_shared:add(nn.Dropout(dropout_prob))
		model_features_shared:add(nn.Linear(number_hidden_nodes_in_shared_net[1], number_hidden_nodes_in_shared_net[1]))
		effective_backward_window = 1

		size_of_shared_network_output = number_hidden_nodes_in_shared_net[1]
	end

	if (opt.model == 'lstmall') then
		model_features_shared:add(nn.Sequencer(nn.Linear(labcounts, lstmhiddenSize)))
		for i = 1, lstm_depth do
			model_features_shared:add(nn.Sequencer(nn.LSTM(lstmhiddenSize, lstmhiddenSize)))
		end
		model_features_shared:add(nn.JoinTable(2,2)) --this way we already take care of Reshape
		effective_backward_window = backward_window

		model_features_shared:add(nn.Linear(lstmhiddenSize * effective_backward_window, number_hidden_nodes_in_shared_net[1]))
		model_features_shared:add(nn.ReLU())
		model_features_shared:add(nn.Dropout(dropout_prob))
		model_features_shared:add(nn.Linear(number_hidden_nodes_in_shared_net[1], number_hidden_nodes_in_shared_net[1]))
		size_of_shared_network_output = number_hidden_nodes_in_shared_net[1]
	end

	if (opt.model == 'multiresconvnet') then  	
		local model_tmp1 = nn.ConcatTable()
		
		
		

		model_resolution_1 = nn.Sequential()
		model_resolution_1:add(nn.SpatialMaxPooling(resolution_1_poolwidth[1], 1, resolution_1_poolstride[1], 1, 0, 0))
			effective_len1 = (1+ math.floor((backward_window - resolution_1_poolwidth[1]) / resolution_1_poolstride[1]))
		model_resolution_1:add(nn.SpatialMaxPooling(resolution_1_poolwidth[2], 1, resolution_1_poolstride[2], 1, 0, 0))
			effective_len1 = (1+ math.floor((effective_len1 - resolution_1_poolwidth[2]) / resolution_1_poolstride[2]))
		model_resolution_1:add(nn.SpatialConvolutionMM(1, num_of_multires_filters , resolution_1_convkwidth[1], 1, resolution_1_convkstride[1], 1, 0, 0))
			effective_len1 = (1+ math.floor((effective_len1 - resolution_1_convkwidth[1]) / resolution_1_convkstride[1]))		
		if use_batch_normalization == 1 then
			model_resolution_1:add(nn.SpatialBatchNormalization(num_of_multires_filters, batch_normalization_epsilon))
		end
		model_resolution_1:add(nn.ReLU()) 

		

		local model_resolution_2 = nn.Sequential()
		model_resolution_2:add(nn.SpatialMaxPooling(resolution_2_poolwidth[1], 1, resolution_2_poolstride[1], 1, 0, 0))
			effective_len2 = (1+ math.floor((backward_window - resolution_2_poolwidth[1]) / resolution_2_poolstride[1]))
		model_resolution_2:add(nn.SpatialConvolutionMM(1, num_of_multires_filters , resolution_2_convkwidth[1], 1, resolution_2_convkstride[1], 1, 0, 0))
			effective_len2 = (1+ math.floor((effective_len2 - resolution_2_convkwidth[1]) / resolution_2_convkstride[1]))
		if use_batch_normalization == 1 then
			model_resolution_2:add(nn.SpatialBatchNormalization(num_of_multires_filters, batch_normalization_epsilon))
		end
		model_resolution_2:add(nn.ReLU())      	

		local model_resolution_3 = nn.Sequential()
		model_resolution_3:add(nn.SpatialConvolutionMM(1, num_of_multires_filters, resolution_3_convkwidth[1], 1, resolution_3_convkstride[1], 1, 0, 0))
			effective_len3 = (1+ math.floor((backward_window - resolution_3_convkwidth[1]) / resolution_3_convkstride[1]))
		if use_batch_normalization == 1 then
			model_resolution_3:add(nn.SpatialBatchNormalization(num_of_multires_filters, batch_normalization_epsilon))
		end
		model_resolution_3:add(nn.ReLU())      	
		model_resolution_3:add(nn.SpatialMaxPooling(resolution_3_poolwidth[1], 1, resolution_3_poolstride[1], 1, 0, 0))
			effective_len3 = (1+ math.floor((effective_len3 - resolution_3_poolwidth[1]) / resolution_3_poolstride[1]))
		model_resolution_3:add(nn.SpatialConvolutionMM(num_of_multires_filters, num_of_multires_filters , resolution_3_convkwidth[2], 1, resolution_3_convkstride[2], 1, 0, 0))
			effective_len3 = (1+ math.floor((effective_len3 - resolution_3_convkwidth[2]) / resolution_3_convkstride[2]))
		if use_batch_normalization == 1 then
			model_resolution_3:add(nn.SpatialBatchNormalization(num_of_multires_filters, batch_normalization_epsilon))
		end
		model_resolution_3:add(nn.ReLU())     
		model_tmp1:add(model_resolution_1)
		model_tmp1:add(model_resolution_2)
		model_tmp1:add(model_resolution_3)

		model_features_shared:add(model_tmp1)
		model_features_shared:add(nn.JoinTable(3,3))

		if (effective_len1 <1 or effective_len2<1 or effective_len3<1) then
			print('Fatal Error: For multiresolution convolution with this specifications you do not have enough backward window.')
			print('with your current parameters here is how the length of the result will be theoretically:')
			print('level 1(lowest resolution):'..effective_len1)
			print('level 2(middle resolution):'..effective_len2)
			print('level 3(highest resolution):'..effective_len3)
			os.exit()
		end
		current_layer_width = effective_len3 + effective_len2 + effective_len1
		effective_backward_window = current_layer_width
		
		model_features_shared:add(nn.Reshape(num_of_multires_filters * vertical_dimension * effective_backward_window))
		model_features_shared:add(nn.Dropout(dropout_prob))
		model_features_shared:add(nn.Linear(num_of_multires_filters * vertical_dimension * effective_backward_window, number_hidden_nodes_in_shared_net[1]))
		model_features_shared:add(nn.ReLU())

		-- model_features_shared:add(nn.Dropout(dropout_prob))
		-- model_features_shared:add(nn.Linear(number_hidden_nodes_in_shared_net[1], number_hidden_nodes_in_shared_net[1]))
		-- if use_batch_normalization == 1 then
		-- 	model_features_shared:add(nn.BatchNormalization(number_hidden_nodes_in_shared_net[1]))
		-- end
		-- model_features_shared:add(nn.ReLU())

		for dd = 2, depth_hidden_multires do
			print ('multires convnet depth:'..dd)
			model_features_shared:add(nn.Dropout(dropout_prob))
			model_features_shared:add(nn.Linear(number_hidden_nodes_in_shared_net[1], number_hidden_nodes_in_shared_net[1]))
			if use_batch_normalization == 1 then
				model_features_shared:add(nn.BatchNormalization(number_hidden_nodes_in_shared_net[1]))
			end
			model_features_shared:add(nn.ReLU())
		end
		size_of_shared_network_output = number_hidden_nodes_in_shared_net[1]
	end

	if (opt.model == 'logit') then
		model_features_shared:add(nn.Identity())		
		effective_backward_window = backward_window
		if use_batch_normalization == 1 then
			model_features_shared:add(nn.BatchNormalization(1))
		end
		model_features_shared:add(nn.Reshape(vertical_dimension * effective_backward_window))
		size_of_shared_network_output = vertical_dimension * effective_backward_window
	end

	if (opt.model == 'max_logit') then
		model_features_shared:add(nn.Max(4))
		model_features_shared:add(nn.Reshape(vertical_dimension))

		effective_backward_window = 1
		size_of_shared_network_output = vertical_dimension	
	end

	if (opt.model == 'mlp') then
		model_features_shared:add(nn.Reshape(vertical_dimension * effective_backward_window))	
		model_features_shared:add(nn.Dropout(dropout_prob))
		model_features_shared:add(nn.Linear(vertical_dimension * effective_backward_window, mlp_layer_width_shared[1]))
		model_features_shared:add(nn.ReLU())
		if use_batch_normalization == 1 then
			model_features_shared:add(nn.BatchNormalization(mlp_layer_width_shared[1]))
		end
		for i = 1, mlp_depth_shared - 1 do
			model_features_shared:add(nn.Dropout(dropout_prob))
			model_features_shared:add(nn.Linear(mlp_layer_width_shared[i], mlp_layer_width_shared[i+1]))
			model_features_shared:add(nn.ReLU())
			if use_batch_normalization == 1 then
				model_features_shared:add(nn.BatchNormalization(mlp_layer_width_shared[i+1]))
			end
		end
		size_of_shared_network_output = mlp_layer_width_shared[mlp_depth_shared]		
	end
	
	-----------------   part 2 of the network is output specific. We use concatTable for this part.  -----------------
	per_icd9_model_parallel = nn.ConcatTable()
	for icd9ix = 1, diseasecount do		
		local per_icd9_model_parallel_i = nn.Sequential()
		per_icd9_model_parallel_i:add(nn.Dropout(dropout_prob))
		per_icd9_model_parallel_i:add(nn.Linear(size_of_shared_network_output, 2))
		if use_batch_normalization == 1 then
			per_icd9_model_parallel_i:add(nn.BatchNormalization(2))
		end
		per_icd9_model_parallel_i:add(nn.LogSoftMax())
		per_icd9_model_parallel:add(per_icd9_model_parallel_i)		
	end

	big_prediction_model:add(model_features_shared)
	big_prediction_model:add(per_icd9_model_parallel)

	return big_prediction_model:cuda()
end

function normalize(input)
	local inputnnx = input:ne(0):clone()
	local mean = torch.cdiv(input:sum(4):clone(), inputnnx:sum(4):clone()):squeeze() --size 18	
	mean[inputnnx:sum(4):clone():squeeze():abs():eq(0)] = 0.0	

	local std = torch.cdiv(torch.pow(input,2):clone():sum(4):clone(), inputnnx:sum(4):clone()):squeeze():clone() - torch.cmul(mean,mean):clone()
	std[inputnnx:sum(4):clone():squeeze():eq(0)] = 1.0 -- we dont' divide or mult by zero std. replace it by 1.0
	std[std:lt(0)] = 1.0 --sometimes it's -0.0 don't know why..
	std = torch.sqrt(std):clone()
	
	std = std:view(std:size(1),1):clone() -- size 18x1
	mean = mean:view(mean:size(1),1):clone() --size 18x1

	input = input - mean:repeatTensor(1, input:size(4)):clone()
	input = torch.cmul(input, inputnnx)
	stdtmp = std:clone()
	stdtmp[stdtmp:lt(0.2)] = 1.0 --if std is small, don't divide and then don't multiply. it's ok if the range is a bit high..
	input = torch.cdiv(input, stdtmp:repeatTensor(1,input:size(4)):clone()):clone()
	return input:clone(), inputnnx:clone(), mean:clone(), stdtmp:clone()
end

function train(model_predicter)
	print('trainig begins')
	collectgarbage()
	model_predicter:training()	

	train_avg_auc_table = {0}
	valid_avg_auc_table = {0}

	local allparameters, allgradients = nn.Container():cuda():add(model_predicter):getParameters()

	local batch_input = torch.CudaTensor(batchSize, 1, labcounts, backward_window):fill(0)
	local batch_input_nnx = torch.CudaTensor(batchSize, 1, labcounts, backward_window):fill(0)
	local batch_target = torch.CudaTensor(batchSize, diseasecount, 1, 1):fill(0)
	local batch_tobe_excluded_outcomes = torch.CudaTensor(batchSize, diseasecount, 1, 1):fill(0)
	local batch_mu = torch.CudaTensor(batchSize, 1, labcounts, backward_window):fill(0)
	local batch_std = torch.CudaTensor(batchSize, 1 ,labcounts, backward_window):fill(0)
	local bix = 0
	local bcntr = 0
	local training_stage = 0
	
	local old_valdauc = old_valdauc or torch.Tensor(diseasecount):fill(0.5)	

	optimizationstate = {
    	learningRate = opt.learningRate,
    	momentum = opt.momentum,
    	learningRateDecay = opt.learningRateDecay,
    	weightDecay = opt.weightDecay
	}

	for epoch = 1, maxEpoch do
		print('epoch' .. epoch)
		model_predicter:training()	
		
		print(totalBatchCntTrain)
		local shuffled_bix = torch.randperm(totalBatchCntTrain)
		
		for bcntr = 1, totalBatchCntTrain do
			local bix = shuffled_bix[batchix]
			print('---'.. bcntr ..'---')	
			collectgarbage()

			local networkForwardBackward = function(argparameters)
				if argparameters ~= allparameters  then
					allparameters:copy(argparameters)
				end
				allgradients:zero()


				batch_input = torch.load(batches_network_dir..'bix'..bcntr..'_batch_input'):cuda()
				if (opt.augment_input == 1) then
					batch_input = augment_input(batch_input)
				end
				batch_input_nnx = torch.load(batches_network_dir..'bix'..bcntr..'_batch_input_nnx'):cuda()
				batch_target = torch.load(batches_network_dir..'bix'..bcntr..'_batch_target'):cuda()
				batch_tobe_excluded_outcomes = torch.load(batches_network_dir..'bix'..bcntr..'_batch_tobe_excluded_outcomes'):cuda()
				batch_mu = torch.load(batches_network_dir..'bix'..bcntr..'_batch_mu'):cuda()
				batch_std = torch.load(batches_network_dir..'bix'..bcntr..'_batch_std'):cuda()
						
				-- model_predicter:zeroGradParameters()
				-- model_imputer:zeroGradParameters()
										
				local batch_scaled
				if scale_imputed_data == 1 then	
					batch_scaled = torch.cmul(batch_input, batch_std) + batch_mu
				else
					batch_scaled = batch_input:clone() 
				end

				if (opt.model == 'lstmlast' or opt.model == 'lstmall') then
					local input_table = {}
					for tix = 1, batch_scaled:size(4) do
						table.insert(input_table, batch_scaled[{{},{1},{},{tix}}]:squeeze():clone())
					end
					batch_scaled = input_table
				end

				local batch_predict = model_predicter:forward(batch_scaled)	--predict returns: {log P(y_i == 0 | xi) and log P(y_i == 1 | xi)} table of size diseasecount
				local batch_log_loss_gd = {}

				local loss_bt_sum= 0
				for dx = 1, diseasecount do					
					local criterion = nn.ClassNLLCriterion( class_weight[{{},{dx}}]:clone():squeeze() ):cuda()
					if (exclude_already_onset == 1) then							
						batch_predict[dx]:cmul(batch_tobe_excluded_outcomes[{{},{dx},{1},{1}}]:eq(0):view(batchSize,1):clone():repeatTensor(1,2))							
					end					
					local loss1 = criterion:forward(batch_predict[dx], batch_target[{{},{dx},{},{}}]:clone():view(batchSize):clone() + 1 )
					loss_bt_sum = loss_bt_sum + loss1
					
					local loss1_gd = criterion:backward(batch_predict[dx], batch_target[{{},{dx},{},{}}]:clone():view(batchSize):clone() + 1 )
					if (exclude_already_onset == 1) then							
						loss1_gd:cmul(batch_tobe_excluded_outcomes[{{},{dx},{1},{1}}]:eq(0):view(batchSize,1):clone():repeatTensor(1,2))
					end
					table.insert(batch_log_loss_gd, disease_weight[dx] * loss1_gd:clone())
				end	
				table.insert(loss_train_values, loss_bt_sum/diseasecount); 
				print(loss_bt_sum/diseasecount)

				local predictor_gd = model_predicter:backward(batch_scaled, batch_log_loss_gd)

				return allloss, allgradients
			end			
			optim.adadelta(networkForwardBackward, allparameters, optimizationstate)
		end

		save_model(model_predicter, epoch, bcntr, 1)
		
		validate_auc = validate_prediction(model_predicter, 0, 1, epoch)
		print('*************** Average validate AUC: '.. validate_auc:mean())
		table.insert(valid_avg_auc_table, validate_auc:mean())
		train_auc = validate_prediction(model_predicter, 1, 1, epoch)
		print('*************** Average train AUC: '.. train_auc:mean())
		table.insert(train_avg_auc_table, train_auc:mean())

		gnuplot.figure(1)
		gnuplot.plot({'train avg auc (model:' .. save_train_network_dir..')', torch.Tensor(train_avg_auc_table)}, {'validate avg auc', torch.Tensor(valid_avg_auc_table)})
	end	
end

function save_model( model1, epoch, bcntr, flag_sofar )
	if flag_sofar == 1 then
		filename = paths.concat(save_train_network_dir .. '/predictor_epoch'..epoch .. '_bcntr'.. bcntr..'_bestsofar'..'.net')
	else
		filename = paths.concat(save_train_network_dir .. '/predictor_epoch'..epoch .. '_bcntr'.. bcntr..'.net')
	end
	os.execute('mkdir -p ' .. sys.dirname(filename))
	if paths.filep(filename) then
	  os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
	end
	print('Saving network to '..filename)
	torch.save(filename, model1)
end

function load_predictor_model_from_file( epoch, bcntr, flag_sofar )
	if opt.model == 'logit' then
		flag_sofar = 0
	end
	--/sontag-md3400/m/users/narges/IBC/health/code/users/narges/deep/code/convnet_baseline/nov5/net_dpout_exc1_scl1_impnone_imptstnone_trkr0_blnc1_jntkr0_Lrate0.05/mlp/predictor_epoch12_bcntr4400.netESC
	if flag_sofar == 1 then
		filename = paths.concat(load_pretrain_train_network_dir .. '/predictor_epoch'..epoch .. '_bcntr'.. bcntr..'_bestsofar'..'.net')
	else
		filename = paths.concat(load_pretrain_train_network_dir .. '/predictor_epoch'..epoch .. '_bcntr'.. bcntr..'.net')
	end
	local model1 = torch.load(filename)
	return model1
end

function backup_model(model1, ixx)
	if (ixx == -1) then
		model_predicter_backup = model1:clone()
	elseif opt.model == 'logit' or opt.model == 'multiresconvnet' or opt.model == 'convnet' then
		model_predicter_backup:get(2):get(ixx):get(2).weight = model1:get(2):get(ixx):get(2).weight:clone()
	elseif opt.model == 'mlp' then
		model_predicter_backup:get(2):get(ixx):get(2).weight = model1:get(2):get(ixx):get(2).weight:clone()
	end
end

function restore_model_disease_i(model1, ixx)
	if opt.model == 'logit' or opt.model == 'multiresconvnet' or opt.model == 'convnet' then
		model1:get(2):get(ixx):get(2).weight = model_predicter_backup:get(2):get(ixx):get(2).weight:clone()
	elseif opt.model == 'mlp' then
		model1:get(2):get(ixx):get(2).weight = model_predicter_backup:get(2):get(ixx):get(2).weight:clone()
	end
end

function validate_prediction(model_predicter, mode_train_or_valid, write_to_file, epoch)	
	model_predicter:evaluate()
	collectgarbage()

	if mode_train_or_valid == 0 then
		print('-----validating----')
		total_count = totalBatchCntValid
		log_filename_test = save_train_network_dir .. '/validate__aucs_all.log'
	elseif mode_train_or_valid == 1 then
		print('-----scoring the train----')
		total_count = totalBatchCntScrTrain
		log_filename_test = save_train_network_dir .. '/scoretrain__aucs_all.log'
	elseif mode_train_or_valid == 2 then
		print('-----scoring the TEST set----')
		total_count = totalBatchCntScrTest
		log_filename_test = validation_dir .. '/test__aucs_all.log'
	end

	total_size = total_count * batchSize

	local bix = 0
	local bcntr = 0
	local minivalidation_auc = torch.Tensor(diseasecount):fill(0.5) --so, if we see NaN, we keep 0.5
	local all_predictions = torch.Tensor(diseasecount, total_size):fill(0)
	local all_mask = torch.Tensor(diseasecount,total_size):fill(0)
	local all_target_adjusted = torch.Tensor(diseasecount, total_size):fill(0)
	local all_predictions_cntr = 0
	local last_batch_flag = 0

	for bcntr = 1, total_count do
		collectgarbage()
		if mode_train_or_valid == 0 then --valid
			batch_input = torch.load(batches_network_dir_valid..'bix'..bcntr..'_batch_input'):cuda()
			batch_input_nnx = torch.load(batches_network_dir_valid..'bix'..bcntr..'_batch_input_nnx'):cuda()
			batch_target = torch.load(batches_network_dir_valid..'bix'..bcntr..'_batch_target'):cuda()
			batch_tobe_excluded_outcomes = torch.load(batches_network_dir_valid..'bix'..bcntr..'_batch_tobe_excluded_outcomes'):cuda()
			batch_mu = torch.load(batches_network_dir_valid..'bix'..bcntr..'_batch_mu'):cuda()
			batch_std = torch.load(batches_network_dir_valid..'bix'..bcntr..'_batch_std'):cuda()
		elseif mode_train_or_valid == 1 then
			batch_input = torch.load(batches_network_dir_scoretrain..'bix'..bcntr..'_batch_input'):cuda()
			batch_input_nnx = torch.load(batches_network_dir_scoretrain..'bix'..bcntr..'_batch_input_nnx'):cuda()
			batch_target = torch.load(batches_network_dir_scoretrain..'bix'..bcntr..'_batch_target'):cuda()
			batch_tobe_excluded_outcomes = torch.load(batches_network_dir_scoretrain..'bix'..bcntr..'_batch_tobe_excluded_outcomes'):cuda()
			batch_mu = torch.load(batches_network_dir_scoretrain..'bix'..bcntr..'_batch_mu'):cuda()
			batch_std = torch.load(batches_network_dir_scoretrain..'bix'..bcntr..'_batch_std'):cuda()
		elseif mode_train_or_valid == 2 then
			batch_input = torch.load(batches_network_dir_test..'bix'..bcntr..'_batch_input'):cuda()
			batch_input_nnx = torch.load(batches_network_dir_test..'bix'..bcntr..'_batch_input_nnx'):cuda()
			batch_target = torch.load(batches_network_dir_test..'bix'..bcntr..'_batch_target'):cuda()
			batch_tobe_excluded_outcomes = torch.load(batches_network_dir_test..'bix'..bcntr..'_batch_tobe_excluded_outcomes'):cuda()
			batch_mu = torch.load(batches_network_dir_test..'bix'..bcntr..'_batch_mu'):cuda()
			batch_std = torch.load(batches_network_dir_test..'bix'..bcntr..'_batch_std'):cuda()
		end

		batch_nottobe_excluded_outcomes = batch_tobe_excluded_outcomes:eq(0)

		local batch_imputed_scaled = nil
		if scale_imputed_data == 1 then
			batch_imputed_scaled = torch.cmul(batch_input, batch_std) + batch_mu
		else
			batch_imputed_scaled = batch_input:clone() 
		end		

		if (opt.model == 'lstmlast' or opt.model == 'lstmall') then
			local input_table = {}
			for tix = 1, batch_imputed_scaled:size(4) do
				table.insert(input_table, batch_imputed_scaled[{{},{1},{},{tix}}]:squeeze():clone())
			end
			batch_imputed_scaled = input_table
		end

		local batch_predict = model_predicter:forward(batch_imputed_scaled)					
		
		all_target_adjusted[{{},{all_predictions_cntr+1, all_predictions_cntr+batchSize}}] = batch_target:transpose(1,2):clone():view(diseasecount, batchSize):clone():float()
		all_mask[{{}, {all_predictions_cntr+1, all_predictions_cntr+batchSize}}] = batch_nottobe_excluded_outcomes:transpose(1,2):clone():view(diseasecount, batchSize):clone():float()

		local loss_bt_sum = 0
		for dxi = 1, diseasecount do
			local tmp1 = batch_nottobe_excluded_outcomes[{{},{dxi},{1},{1}}]:clone():view(batchSize,1):clone() --repeat 1 time in 1st dim, 2 times in 2nd dim
			if (exclude_already_onset == 1) then							
				batch_predict[dxi]:cmul(tmp1:repeatTensor(1,2):clone())
			end
			local criterion = nn.ClassNLLCriterion( class_weight[{{},{dxi}}]:clone():squeeze() ):cuda()
			local loss1 = criterion:forward(batch_predict[dxi], batch_target[{{},{dxi},{},{}}]:clone():view(batchSize):clone() + 1)												
			loss_bt_sum = loss1 + loss_bt_sum
			all_predictions[{{dxi},{all_predictions_cntr+1, all_predictions_cntr+batchSize}}] = batch_predict[dxi][{{},{2}}]:clone():view(1, batchSize):clone():float():clone() --logP(y=1) only
		end
		if mode_train_or_valid == 1 then
			table.insert(loss_mini_train_values, loss_bt_sum/diseasecount)
		else
			table.insert(loss_mini_validate_values, loss_bt_sum/diseasecount)
		end
		print(loss_bt_sum/diseasecount)
		all_predictions_cntr = all_predictions_cntr + batchSize
	end

	line_to_write = ''
	for dxi = 1, diseasecount do
		collectgarbage()
		local select_i = all_mask[{{dxi},{}}]:clone():view(-1):clone()
		local auc_icd9 = 0.5

		if (select_i:sum() > 1) then
			local pred_vals_logppos = all_predictions[{{dxi},{}}]:clone():view(-1):clone()[select_i:eq(1)]:clone()
			local target_vals = all_target_adjusted[{{dxi},{}}]:clone():view(-1):clone()[select_i:eq(1)]:clone()
			
			if mode_train_or_valid == 2 then --task=test
				torch.save(validation_dir .. '/' .. dxi ..'_pred_vals_logppos.th', pred_vals_logppos)
				torch.save(validation_dir .. '/' .. dxi .. '_target_values.th', target_vals)
			end

			local roc_points = ROC.points( pred_vals_logppos, target_vals) --roc code can handle 0/1 target
			print(dxi..' '.. diseaseLabels[dxi])
			print(' total '..select_i:sum().. ' pos '..target_vals:sum())
			auc_icd9 = ROC.area(roc_points)
			
			if (auc_icd9 > 100) or (auc_icd9 < -100) then -- in case of numeric bug in roc :(
				auc_icd9 = 0.5
			end
			if (auc_icd9 > 0.5)	or (auc_icd9 < 0.5) then
				minivalidation_auc[dxi] = auc_icd9
			end
		end
		print(diseaseLabels[dxi]..'|'.. auc_icd9)
		line_to_write = line_to_write .. '|' .. auc_icd9
	end	
	line_to_write = epoch .. '|0|' .. minivalidation_auc:mean().. '|' .. line_to_write 
	if write_to_file == 1 then
		log_file_open = io.open(log_filename_test, "a")	
		log_file_open:write(line_to_write .. '\n')
		log_file_open:flush()
		log_file_open:close()
	end
	return minivalidation_auc
end

function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    for filename in popen('ls -a '..directory):lines() do
        i = i + 1
        t[i] = filename
    end
    return t
end

function parse_option( inputstr )
	local res = {}
	local items = string.gsub(string.gsub(inputstr, '{', ''), '}',''):split(',')
	for i=1,#items do
		table.insert(res,tonumber(items[i]))
	end
	return res
end

if opt.task == 'train' then
	init()
	model_predicter = build_model()
	train(model_predicter)	
end

if opt.task == 'test' then
	init()
	collectgarbage()
	xtrain, ytrain = load_data(opt.task)
	collectgarbage()
	valid_log_filename = validation_dir .. '/validate__aucs_all.log'
	best_epoch_ix = torch.Tensor(diseasecount):fill(0)
	best_epoch_val = torch.Tensor(diseasecount):fill(0)
	best_epoch_ix_mean = 0
	best_epoch_auc_mean = 0
	for line in io.lines(valid_log_filename) do		
		local epoch_values = line:split("|")
		local epoch = tonumber(epoch_values[1])
		local average_epoch_auc = tonumber(epoch_values[3])
		if average_epoch_auc > best_epoch_auc_mean then
			best_epoch_ix_mean = epoch
			best_epoch_auc_mean = average_epoch_auc
		end
	end
	print('best epoch was epoch: '..best_epoch_ix_mean)
	print('best AUC achieved at that epoch was: '..best_epoch_auc_mean)
	
	model_to_load_filename = validation_dir .. '/predictor_epoch' .. best_epoch_ix_mean .. '_bcntr0_bestsofar'..'.net'
	model_to_load_predictor = torch.load(model_to_load_filename)
	model_to_load_predictor:cuda()
	if (opt.verbose == 1) then
		print(model_to_load_predictor)
	end
	validate_prediction(model_to_load_predictor, 2, 1, best_epoch_ix_mean)
end