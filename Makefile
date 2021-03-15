.PHONY: train help test generate corner

PYTHON=python3
data = 'datafile.hdf5'
model = 'model.pkl'

all: help

help:
	@echo "to generate the dataset type make generate data=<dataname.hdf5>"
	@echo "to train the network type make train data=<dataname.hdf5> model=<modelname.pkl>"
	@echo "to test the network type make test data=<dataname.hdf5> model=<modelname.pkl>"
	@echo "to produce a corner plot type make corner data=<dataname.hdf5> model=<modelname.pkl>"
	@echo "default: data=datafile.hdf5 model=model.pkl"

train: 
	chmod u+x scripts/train.py
	$(PYTHON) scripts/train.py $(data) $(model)

test:
	chmod u+x scripts/test.py
	$(PYTHON) scripts/test.py $(data) $(model)
    
corner:
	chmod u+x scripts/corner_plot.py
	$(PYTHON) scripts/corner_plot.py $(data) $(model)
    
generate:
	chmod u+x scripts/generate.py
	$(PYTHON) scripts/generate.py $(data)