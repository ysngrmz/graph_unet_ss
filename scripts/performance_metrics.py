import numpy as np
import sklearn.metrics as metrics
def matrix2vector(targets,predicts,nlabel):
	nPro = np.shape(targets)[0]
	prob2class = []
	for ind in range(nPro):
		line = predicts[ind]
		temp_class = []
		for elem in line:
			c_label = np.where(elem==np.max(elem))

			nMax = np.shape(c_label)
			if(nMax[1]>1):
				c_label = c_label[0]

			c_label = int(c_label[0])
			temp_class.append(c_label)
		prob2class.append(temp_class)
	targets_ = []
	predicts_ = []

	for ind in range(nPro):
		t = np.reshape(targets[ind],np.shape(targets[ind])[0])
		p = prob2class[ind]
		for elem in range(len(p)):
			if(t[elem] == nlabel):
				break
			targets_.append(t[elem])
			predicts_.append(p[elem])
	return targets_,predicts_

def clean_prob(targets,predicts,nlabel):
	nPro = np.shape(targets)[0]
	cleanProb = []
	targets_ = []
	for ind in range(nPro):
		t = np.reshape(targets[ind],np.shape(targets[ind])[0])
		p = predicts[ind]
		for elem in range(np.shape(p)[0]):
			if(t[elem] == nlabel):
				break
			targets_.append(t[elem])
			cleanProb.append(p[elem])
	return targets_,cleanProb
def calc_acc_full_predict(targets,predicts,nlabel):
	t,p = matrix2vector(targets,predicts,nlabel)
	acc = metrics.accuracy_score(t,p)
	#perf_metrics = metrics.precision_recall_fscore_support(t,p)
	return acc

