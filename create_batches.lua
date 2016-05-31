------------------------------------------------------------------------------------------------
-- If using this code or its variant please cite:
-- Narges Razavian, David Sontag, "Temporal Convolutional Neural Networks 
-- for Diagnosis from Lab Tests", ICLR 2016 Workshop track.
-- Link: http://arxiv.org/abs/1511.07938 
-- For questions about the code contact Narges (narges.sharif@gmail.com)
-----------------------------------------------------------------------------------------------

require 'cutorch'
require 'os'

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(2)

local opt = lapp[[
	--gpuid           		(default 1),
	--task			  		(default 'train'),
	--exclude_already_onset (default 1),
	--imputation_type    	(default 'none'),
	--normalize_each_timeseries   	(default 1),
	--input_dir				(default './sampledata'),
	--batch_output_dir 		(default './sampleBatchDir/'),
	--batchSize 			(default 256),	
	--diagnosis_gap 		(default 3),
	--diagnosis_window 		(default 24),
	--min_icd9_cnt 			(default 2),
	--min_icd9_cnt_exclude  (default 1),
	--backward_window 		(default 36),
	--min_lab_months_measured (default 3),
	--staged_training		(default 1),
	--min_data_for_training_stage_y  (default {20, 5, 0}),
	--min_data_for_training_stage_x  (default {50, 10, 0}),
]]
cutorch.setDevice(opt.gpuid)
print(opt)

function init()
	input_dir = opt.input_dir
	batch_output_dir = opt.batch_output_dir .. '/' ..opt.task ..'/'

	if (opt.task == 'train' or opt.task =='test' or opt.task =='valid') then
		data_torch_bin_filename = input_dir..'/'.. opt.task ..'/x'.. opt.task ..'_normalized.bin'
		data_torch_bin_filename_outcomes = input_dir..'/'.. opt.task ..'/y'.. opt.task ..'_outcomes_binary.bin'
	elseif (opt.task == 'scoretrain') then
		data_torch_bin_filename = input_dir..'/train/xtrain_normalized.bin'
		data_torch_bin_filename_outcomes = input_dir..'/train/ytrain_outcomes_binary.bin'
	end

	os.execute('mkdir -p '.. batch_output_dir)
	os.execute('cp ./create_batches.lua ' .. batch_output_dir..'/../')

	batchSize = opt.batchSize
	imputation_type = opt.imputation_type
	normalize_each_timeseries = opt.normalize_each_timeseries
	diagnosis_gap = opt.diagnosis_gap --3 months -previously it was 0
	diagnosis_window = opt.diagnosis_window  --2 years -previously it was 1
	exclude_already_onset = opt.exclude_already_onset 
	min_icd9_cnt = opt.min_icd9_cnt
	min_icd9_cnt_exclude = opt.min_icd9_cnt_exclude
	backward_window = opt.backward_window -- backward window is the how many months we will look for certain patterns. It is 3 years now
	min_lab_months_measured = opt.min_lab_months_measured --i.e. at least measured 3+1 times in the backward window
	staged_training = opt.staged_training
	if (staged_training == 1) then
		min_data_for_training_stage_y = string.gsub(string.gsub(opt.min_data_for_training_stage_y, '{', ''), '}',''):split(',') --{20, 5, 0}
		min_data_for_training_stage_x = string.gsub(string.gsub(opt.min_data_for_training_stage_x, '{', ''), '}',''):split(',') --{50, 10, 0}	
	else
		min_data_for_training_stage_y = {-1}
		min_data_for_training_stage_x = {-1}
	end
	if (opt.task=='scoretrain' or opt.task=='test' or opt.task=='valid') then
		min_data_for_training_stage_y = {-1}
		min_data_for_training_stage_x = {-1}		
	end

	print('attempting to load data from:\n'.. data_torch_bin_filename ..'\nand:'..data_torch_bin_filename_outcomes..'\nThis might take some time. Please wait...')

	datax = fromfile(data_torch_bin_filename)
	datay = fromfile(data_torch_bin_filename_outcomes)
	
	print('data successfully loaded!')
	print(datax:size())
	print(datay:size())

	labcounts = datax:size(1)
	peoplecounts = datax:size(2)
	timecounts = datax:size(3)
	diseasecount = datay:size(1)

	if (datay:size():size() == 2) then
		input_format = 2
		print('Y is detected to be of size |diseases| x |people|')
		print('The following options are not used anymore: staged_training, exclusion, min_data_for_training_stage_y, min_icd9_cnt, diagnosis_gap, diagnosis_window,min_icd9_cnt_exclude.')	
	elseif (timecounts == datay:size(3)) then
		input_format = 1
		print('Y is detected to be of size |diseases| x |people| x |cohort time|.')
		print('If specified, staged_training, exclusion and min_data_for_training_stage_y and min_icd9_cnt and min_icd9_cnt_exclude will be used appropriately.')
	end
	if (peoplecounts ~= datay:size(2) and input_format == 2 ) then
		print ('There is a problem with the size. X should be of size |labs| x |people| x |cohort time| and y should be of size |diseases| x |people|. Aborting.')
		os.exit()
	end
	if (peoplecounts ~= datay:size(2) and input_format == 1 ) then
		print ('There is a problem with the size. X should be of size |labs| x |people| x |cohort time| and y should be of size |diseases| x |people| x |time|. Aborting.')
		os.exit()
	end
end

function normalize(input)
	local inputnnx = input:ne(0):clone():typeAs(input)
	local mean = torch.cdiv(input:sum(4):clone(), inputnnx:sum(4):clone()):squeeze() --size labcounts	
	mean[inputnnx:sum(4):clone():squeeze():abs():eq(0)] = 0.0	
	local std = torch.cdiv(torch.pow(input,2):clone():sum(4):clone(), inputnnx:sum(4):clone()):squeeze():clone() - torch.cmul(mean,mean):clone()
	std[inputnnx:sum(4):clone():squeeze():eq(0)] = 1.0 -- we dont' divide or mult by zero std. replace it by 1.0
	std[std:lt(0)] = 1.0 --sometimes it's -0.0 don't know why..
	std = torch.sqrt(std):clone()
	std = std:view(std:size(1),1):clone() -- size labcountsx1
	mean = mean:view(mean:size(1),1):clone() --size labcountsx1 
	input = input - mean:repeatTensor(1, input:size(4)):clone()
	input = torch.cmul(input, inputnnx)
	stdtmp = std:clone()
	stdtmp[stdtmp:lt(0.2)] = 1.0 --if std is small, don't divide and then don't multiply. it's ok if the range is a bit high.
	input = torch.cdiv(input, stdtmp:repeatTensor(1,input:size(4)):clone()):clone()
	return input:clone(), inputnnx:clone(), mean:clone(), stdtmp:clone()
end

function build()
	print('creating batches for task ' .. opt.task..' begins')
	collectgarbage()

	local batch_input = torch.Tensor(batchSize, 1, labcounts, backward_window):fill(0)
	local batch_input_nnx = torch.Tensor(batchSize, 1, labcounts, backward_window):fill(0)
	local batch_target = torch.Tensor(batchSize, diseasecount, 1, 1):fill(0)
	local batch_tobe_excluded_outcomes = torch.Tensor(batchSize, diseasecount, 1, 1):fill(0)
	local batch_mu = torch.Tensor(batchSize, 1, labcounts, backward_window):fill(0)
	local batch_std = torch.Tensor(batchSize, 1 ,labcounts, backward_window):fill(0)
	local bix = 0
	local bcntr = 0
	
	if input_format == 1 then
		local training_stage = 0

		for epoch = 1, #min_data_for_training_stage_x do
			print('epoch:' .. epoch .. 'batch counter:'..bcntr)
			
			training_stage = training_stage + 1
			if training_stage > #min_data_for_training_stage_x then 
				training_stage = #min_data_for_training_stage_x
			end
			
			local shuffled_ix = torch.randperm(peoplecounts)
			local shuffled_it = torch.randperm(timecounts)
			
			for ox = 1, peoplecounts * timecounts -1 do
				local tx = math.fmod(ox, timecounts); if tx == 0 then; tx = timecounts; end;
				local ix = math.floor(ox/timecounts) + 1
				local t = shuffled_it[tx]
				local i = shuffled_ix[ix]

				if (exclude_already_onset == 1) then
					tmax = timecounts + 1 - diagnosis_gap - diagnosis_window
				else
					tmax = timecounts + 1
				end

				if  (i > 0) and (t > backward_window) 
				and (t < tmax) 
				and (datax[{{},{i},{t}}]:ne(0):sum() > 0)
				and (datax[{{},{i},{t-backward_window + 1, t-1}}]:ne(0):sum(1):clone():ne(0):sum() > min_lab_months_measured)
				and (torch.sum(datay[{{},{i},{t+diagnosis_gap, t+diagnosis_gap+diagnosis_window}}]:clone()) > tonumber(min_data_for_training_stage_y[training_stage])) 
				and (torch.sum(datax[{{},{i},{t-backward_window + 1, t}}]:clone():ne(0):clone())> tonumber(min_data_for_training_stage_x[training_stage] )) 
				then
					
					local input1 = datax[{{},{i},{t- backward_window + 1, t}}]:clone():view(1, 1, labcounts, backward_window)								
					
					local input = input1:clone()
					local inputnnx = input:ne(0):clone():typeAs(input)
					local mean = torch.Tensor(labcounts, 1):fill(0)
					local std = torch.Tensor(labcounts, 1):fill(0)
					
					if (opt.normalize_each_timeseries == 1) then
						input, inputnnx, mean, std = normalize(input1)
					end

					local target = datay[{{},{i}, {t+diagnosis_gap, t+diagnosis_gap+diagnosis_window}}]:clone():sum(3):clone():view(1,diseasecount,1,1):ge(min_icd9_cnt)
					local tobe_excluded_outcomes = datay[{{},{i}, {t- backward_window + 1, t+diagnosis_gap}}]:clone():sum(3):clone():view(1,diseasecount,1,1):ge(min_icd9_cnt_exclude)

					bix = bix + 1
					batch_input[{{bix},{1},{},{}}] = input:clone()
					batch_input_nnx[{{bix},{1},{},{}}] = inputnnx:clone()
					batch_target[{{bix},{},{1},{}}] = target:clone()
					batch_tobe_excluded_outcomes[{{bix},{},{1},{}}] = tobe_excluded_outcomes:clone()
					batch_mu[{{bix},{1},{},{}}] = mean:view(1,labcounts,1,1):clone():repeatTensor(1, 1, 1, backward_window):clone()
					batch_std[{{bix},{1},{},{}}] = std:view(1,labcounts,1,1):clone():repeatTensor(1, 1, 1, backward_window):clone()

					if (bix == batchSize or ox == datax:size(2)*datax:size(3) - 1) then					
						collectgarbage()
						print('---'.. bcntr ..'---')
						bix = 0					
						bcntr = bcntr + 1
						torch.save(batch_output_dir..'bix'..bcntr..'_batch_input', batch_input)
						torch.save(batch_output_dir..'bix'..bcntr..'_batch_input_nnx', batch_input_nnx)
						torch.save(batch_output_dir..'bix'..bcntr..'_batch_target', batch_target)
						torch.save(batch_output_dir..'bix'..bcntr..'_batch_tobe_excluded_outcomes', batch_tobe_excluded_outcomes)
						torch.save(batch_output_dir..'bix'..bcntr..'_batch_mu', batch_mu)
						torch.save(batch_output_dir..'bix'..bcntr..'_batch_std', batch_std)
					end
				end			
			end
		end
	end


	if input_format == 2 then
		print('epoch:' .. 0 .. ' batch counter:'..bcntr)
		
		local shuffled_ix = torch.randperm(peoplecounts)
		local shuffled_it = torch.randperm(timecounts)
			
		for ox = 1, peoplecounts * timecounts -1 do
			local tx = math.fmod(ox, timecounts); if tx == 0 then; tx = timecounts; end;
			local ix = math.floor(ox/timecounts) + 1
			local t = shuffled_it[tx]
			local i = shuffled_ix[ix]

			tmax = timecounts + 1
			
			if  (i > 0)	and (t > backward_window) and (t < tmax) 
			and (datax[{{},{i},{t}}]:ne(0):sum() > 0)
			and (datax[{{},{i},{t-backward_window + 1, t-1}}]:ne(0):sum(1):clone():ne(0):sum() > min_lab_months_measured)			
			then
				
				local input1 = datax[{{},{i},{t- backward_window + 1, t}}]:clone():view(1, 1, labcounts, backward_window)
				local input = input1:clone()
				local inputnnx = input:ne(0):clone():typeAs(input)
				local mean = torch.Tensor(labcounts, 1):fill(0)
				local std = torch.Tensor(labcounts, 1):fill(0)
				
				if (opt.normalize_each_timeseries == 1) then
					input, inputnnx, mean, std = normalize(input1)
				end

				local target = datay[{{},{i}}]:clone():view(1,diseasecount,1,1):clone()
				local tobe_excluded_outcomes = target:clone():fill(0)

				bix = bix + 1
				batch_input[{{bix},{1},{},{}}] = input:clone()
				batch_input_nnx[{{bix},{1},{},{}}] = inputnnx:clone()
				batch_target[{{bix},{},{1},{}}] = target:clone()
				batch_tobe_excluded_outcomes[{{bix},{},{1},{}}] = tobe_excluded_outcomes:clone()
				batch_mu[{{bix},{1},{},{}}] = mean:view(1,labcounts,1,1):clone():repeatTensor(1, 1, 1, backward_window):clone()
				batch_std[{{bix},{1},{},{}}] = std:view(1,labcounts,1,1):clone():repeatTensor(1, 1, 1, backward_window):clone()

				if (bix == batchSize or ox == peoplecounts*timecounts - 1) then					
					collectgarbage()
					print('---'.. bcntr ..'---')
					bix = 0					
					bcntr = bcntr + 1
					torch.save(batch_output_dir..'bix'..bcntr..'_batch_input', batch_input)
					torch.save(batch_output_dir..'bix'..bcntr..'_batch_input_nnx', batch_input_nnx)
					torch.save(batch_output_dir..'bix'..bcntr..'_batch_target', batch_target)
					torch.save(batch_output_dir..'bix'..bcntr..'_batch_tobe_excluded_outcomes', batch_tobe_excluded_outcomes)
					torch.save(batch_output_dir..'bix'..bcntr..'_batch_mu', batch_mu)
					torch.save(batch_output_dir..'bix'..bcntr..'_batch_std', batch_std)
				end
			end	
		end
	end
end

function fromfile(fname) --Credit for this function goes to Jure Zbontar
   local file = io.open(fname .. '.type')
   local type = file:read('*all')

   local x
   if type == 'float32' then
      x = torch.FloatTensor(torch.FloatStorage(fname))
   elseif type == 'int32' then
      x = torch.IntTensor(torch.IntStorage(fname))
   elseif type == 'int64' then
      x = torch.LongTensor(torch.LongStorage(fname))
   else
      print(fname, type)
      assert(false)
   end
   local file = io.open(fname .. '.dim')
   local dim = {}
   for line in file:lines() do
      table.insert(dim, tonumber(line))
   end
   x = x:reshape(torch.LongStorage(dim))
   return x:float()
end

init()
build()

