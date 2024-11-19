import numpy as np

def generate_dataset_helper(ws,type,nlabel,ds,task):
	data_read_dir = "/cta/users/ygormez/projects/phd_thesis/data/" + task +"/" +ds + "/"
	ws = int(ws)
	fread = open(data_read_dir + type + "_set.lstm","r")
	nofpro = int(fread.readline())

	Targets = []
	Inputs = []
	fs = 0

	for proN in range(nofpro):
		nofaa = int(fread.readline())
		if(nofaa <= ws ):
			temp_label_list = []
			temp_fea_list = []
			n_of_zero = ws - nofaa
			for nAA in range(nofaa):
				line = fread.readline().replace("\n","")
				line = line.split()
				clabel = int(line[0])
				temp_label_list.append(clabel)
				templine = []
				fs = len(line) - 1
				for elem in range(1,len(line)):
					templine.append(float(line[elem]))
				templine.append(1)
				temp_fea_list.append(templine)
			for nZero in range(n_of_zero):
				temp_label_list.append(nlabel)
				templine = []
				for elem in range(fs+1):
					templine.append(0)
				temp_fea_list.append(templine)
			Targets.append(temp_label_list)
			Inputs.append(temp_fea_list)
		else:
			n_of_part = int(nofaa / ws) + 1
			for part in range(n_of_part-1):
				temp_label_list = []
				temp_fea_list = []
				for nAA in range(ws):
					line = fread.readline().replace("\n","")
					line = line.split()
					clabel = int(line[0])
					temp_label_list.append(clabel)
					fs = len(line)-1
					templine = []
					for elem in range(1,len(line)):
						templine.append(float(line[elem]))
					templine.append(1)
					temp_fea_list.append(templine)
				Inputs.append(temp_fea_list)
				Targets.append(temp_label_list)
			temp_label_list = []
			temp_fea_list = []
			n_of_zero = ws * n_of_part - nofaa
			n_of_elem = ws - n_of_zero
			for nAA in range(n_of_elem):
				line = fread.readline().replace("\n","")
				line = line.split()
				clabel = int(line[0])
				temp_label_list.append(clabel)
				templine = []
				fs = len(line) - 1
				for elem in range(1,len(line)):
					templine.append(float(line[elem]))
				templine.append(1)
				temp_fea_list.append(templine)
			for nZero in range(n_of_zero):
				temp_label_list.append(nlabel)
				templine = []
				for elem in range(fs+1):
					templine.append(0)
				temp_fea_list.append(templine)
			Targets.append(temp_label_list)
			Inputs.append(temp_fea_list)
	return np.array(Inputs),np.array(Targets)


def generate_dataset(ws,nlabel,ds,task):
	trInp,trTar = generate_dataset_helper(ws,"train",nlabel,ds,task)
	teInp,teTar = generate_dataset_helper(ws,"test",nlabel,ds,task)
	vlInp,vlTar = generate_dataset_helper(ws,"validation",nlabel,ds,task)
	return trInp,trTar,teInp,teTar,vlInp,vlTar
