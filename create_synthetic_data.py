import numpy as np
import cPickle
import os
import sys

if __name__=='__main__':
	args = sys.argv[:]
	args.pop(0)

	N = 10000
	D = 15
	T = 60
	O = 20
	seed1 = 2

	while args:
		arg = args.pop(0)
		if arg == '-h':
			usage()
		elif arg=='--N':
			N = int(args.pop(0))
		elif arg=='--D':
			D = int(args.pop(0))
		elif arg=='-T':
			T = int(args.pop(0))
		elif arg=='--O':
			O = int(args.pop(0))
		elif arg=='--seed':
			seed1 = int(args.pop(0))
		elif arg=='--outdir':
			output_dir = args.pop(0)

	np.random.seed(seed=seed1)
	print 'Creating dataset X of dimensions:', D, N, T, ' and Y of dimensions:', O, N, T
	print 'People:', N , ' input_dimensions:', D, ' time_dimension:', T, ' num_outputs:', O

	x = np.zeros((D,N,T), dtype=float)
	y = np.zeros((O,N,T), dtype=int)
	y_shifted = np.zeros((O,N,T), dtype=int)

	onsetTimes = np.random.randint(0, T*5, size=(O,N))
	onsetTimeShif1 =  np.random.randint(-1*int(T/5), 0, size=(O,1))

	for i in range(0,O):
		for j in range(0,N):
			if (onsetTimes[i,j] < T):
				y[i,j,onsetTimes[i,j]:] = 1
				if (onsetTimes[i,j] + onsetTimeShif1[i,0] > 0):
					y_shifted[i,j,onsetTimes[i,j]+onsetTimeShif1[i,0]:] = 1

	print 'Done creating Y. Now creating X. It will take a while.'

	def ident(x):
		return x

	def inver(x):
		return 1.0/(x+0.01)

	random_functions = [np.sin, np.cos, np.square, ident, inver]

	relationship = np.random.randint(0, len(random_functions), size=(D,O))
	coef1_rel = np.random.uniform(0, 10, size=(D,O))
	coef2_rel = np.random.uniform(0, 10, size=(D,O))
	shif1 =  np.random.uniform(0, 10, size=(D,O))
	noise_coef = np.random.uniform(0, 10, size=(D,O))

	for i in range(0,D):
		if (i % 10) == 0:
			print i, ' dims out of ', D, ' total is finished for X..'
		for j in range(0,O):
			x[i,:,:] += (coef1_rel[i,j] * random_functions[relationship[i,j]]( coef2_rel[i,j] * y_shifted[j,:,:] + shif1[i,j]) + noise_coef[i,j]*np.random.normal(size=(1,N,T)))[0,:,:]

	ix_all = np.arange(N)
	np.random.shuffle(ix_all)
	ix_test = ix_all[0:N/3]
	ix_valid = ix_all[N/3:(2*N)/3]
	ix_train = ix_all[(2*N)/3:]

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	print 'Done. Dividing X and Y into train, test, valid sets (one third each).'
	print 'Saving as cPickle file...'
	cPickle.dump(x[:, ix_train , :], open(output_dir+'/xtrain.pkl', 'wb'), -1)
	cPickle.dump(y[:, ix_train , :], open(output_dir+'/ytrain.pkl', 'wb'), -1)
	cPickle.dump(x[:, ix_valid, :], open(output_dir+'/xvalid.pkl', 'wb'), -1)
	cPickle.dump(y[:, ix_valid, :], open(output_dir+'/yvalid.pkl', 'wb'), -1)	
	cPickle.dump(x[:, ix_test, :], open(output_dir+'/xtest.pkl', 'wb'), -1)
	cPickle.dump(y[:, ix_test, :], open(output_dir+'/ytest.pkl', 'wb'), -1)
	print 'Done. Synthetic data is now in ',output_dir