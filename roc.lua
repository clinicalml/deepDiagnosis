local ROC = {}

-- the original version had a problem, as it was not using all the thresholds. In Sklearn they use all the unique values as threshold, and thus the results are slightly lower, but (I think) more correct.
-- anyway now this script matches the sklearn so the reported results are comparable. You can change it back to the old version by uncommenting lines 20-22 and 81-86. --Narges


-- auxiliary method that quickly simulates the ROC curve computation 
-- just to estimate how many points the curve will have,
-- in order to later allocate just that much memory
local function determine_roc_points_needed(responses_sorted, labels_sorted)
   	local npoints = 1
   	local i = 1
   	local nsamples = responses_sorted:size()[1]

   	while i<nsamples do
		local split = responses_sorted[i]
		while i <= nsamples and responses_sorted[i] == split do
			i = i+1
		end
		-- while i <= nsamples and labels_sorted[i] == -1 do
		-- 	i = i+1	
		-- end
		npoints = npoints + 1
   	end
   	return npoints + 2
end


function ROC.points(responses1, labels1)

	--responses are 100x1
	--targets are 100x1 

	responses = responses1:clone():squeeze():float()
	labels = labels1:clone():squeeze():float()


	-- print(responses)
	-- print(labels)
	-- --{turning labels from 0,1 to -1,1}

	labels[torch.lt(labels,0.5)]= -1
	
	-- assertions about the data format expected
	assert(responses:size():size() == 1, "responses should be a 1D vector")
	assert(labels:size():size() == 1 , "labels should be a 1D vector")

   	-- assuming labels {-1, 1}
   	local npositives = torch.sum(torch.eq(labels,  1))
   	local nnegatives = torch.sum(torch.eq(labels, -1))
   	local nsamples = npositives + nnegatives

   	-- print(nsamples)
   	assert(nsamples == responses:size()[1], "labels should have same length as responses")
   	
   	-- sort by response value
   	local responses_sorted, indexes_sorted = torch.sort(responses)   	
   	local labels_sorted = labels:index(1, indexes_sorted)
   	

   	-- one could allocate a lua table and grow its size dynamically
   	-- and at the end convert to torch tensor, but here I am chosing
   	-- to allocate only the exact memory needed, and doing two passes 
   	-- over the data to estimate first how many points will need
  	local roc_num_points = determine_roc_points_needed(responses_sorted, labels_sorted)
   	local roc_points = torch.Tensor(roc_num_points, 2)
   	
   	roc_points[1][1], roc_points[1][2] = 0.0, 0.0

   	local npoints = 1
	local true_negatives = 0
	local false_negatives = 0   	
   	local i = 1

   	while i<nsamples do
		local split = responses_sorted[i]
		-- if samples have exactly the same response, can't distinguish
		-- between them with a threshold in the middle
		while i <= nsamples and responses_sorted[i] == split do
			if labels_sorted[i] == -1 then
				true_negatives = true_negatives + 1
			else
				false_negatives = false_negatives + 1
			end
			i = i+1
		end
		-- while i <= nsamples and labels_sorted[i] == -1 do
		-- 	print(i)
		-- 	true_negatives = true_negatives + 1
		-- 	print('tnn')
		-- 	i = i+1	
		-- end
		npoints = npoints + 1
		local false_positives = nnegatives - true_negatives 
		local true_positives = npositives - false_negatives 
		local false_positive_rate = (1.0*false_positives)/nnegatives
		local true_positive_rate = (1.0*true_positives)/npositives
		roc_points[roc_num_points - npoints + 1][1] = false_positive_rate
		roc_points[roc_num_points - npoints + 1][2] = true_positive_rate
   	end

   	roc_points[roc_num_points][1], roc_points[roc_num_points][2]  = 1.0, 1.0

   	return roc_points
end


function ROC.area(roc_points)

	local area = 0.0 
	local npoints = roc_points:size()[1]

	for i=1, npoints-1 do
		local width = (roc_points[i+1][1] - roc_points[i][1])
		local avg_height = (roc_points[i][2]+roc_points[i+1][2])/2.0
		area = area + width*avg_height
	end

	return area
end

   
return ROC
