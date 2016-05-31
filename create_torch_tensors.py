###############################################################################
# If using this code or its variant please cite:
# Narges Razavian, David Sontag, "Temporal Convolutional Neural Networks 
# for Diagnosis from Lab Tests", ICLR 2016 Workshop track.
# Link: http://arxiv.org/abs/1511.07938 
# For questions about the code contact Narges (narges.sharif@gmail.com)
###############################################################################

import numpy as np
import cPickle
import os
import sys

def tofile(fname, x):
    
    x.tofile(fname)
    open(fname + '.type', 'w').write(str(x.dtype))
    open(fname + '.dim', 'w').write('\n'.join(map(str, x.shape)))

if __name__=='__main__':
	args = sys.argv[:]
	args.pop(0)
	while args:
		arg = args.pop(0)
		if arg == '-h':
			usage()
		elif arg=='--x':
			input_x = args.pop(0)
		elif arg=='--y':
			input_y = args.pop(0)
		elif arg=='--outdir':
			output_dir = args.pop(0)
		elif arg=='--task':
				task = args.pop(0)
	try: 
		x = cPickle.load(open(input_x, 'rb'))
		y = cPickle.load(open(input_y, 'rb'))		
	except:
		try:
			x = np.load(input_x)
			y = np.load(input_y)
		except:
			print 'error', input_x, ' could not be loaded either as pickle or numpy format'
			exit()

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	if not os.path.exists(output_dir+'/' + task ):
		os.makedirs(output_dir+'/' + task)
	tofile( output_dir + '/' + task + '/x'+ task +'_normalized.bin', x.astype('float32'))
	tofile( output_dir + '/' + task + '/y'+ task +'_outcomes_binary.bin', y.astype('int32'))