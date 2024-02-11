## makefile automates the build and deployment for python projects


## Build system
#
PROJ_TYPE =		python
PROJ_MODULES =		git python-resources python-cli python-doc python-doc-deploy
INFO_TARGETS +=		appinfo
PY_DEP_POST_DEPS +=	modeldeps


## Project build
#
# additional cleanup
ADD_CLEAN +=		amr-bank-struct-v3.0-scored.csv \
			corpus/amr-bank-struct-v3.0-parsed.txt
ADD_CLEAN_ALL +=	amr_graph models
CLEAN_DEPS +=		cleanexample
CLEAN_ALL_DEPS +=	cleanalldep

# file, models and entry point
MODEL_CONF_DIR = 	train-config
MODEL_FILE =		corpus/amr-bank-struct-v3.0.txt
EVAL_FILE = 		target/corp/stage/dev/dev.txt
ABIN =			./amr
INF_CONF = 		resources/model/inference.conf


## Project data
#
# text to parse for run examples
TEST_TEXT = 		"Barack Obama is an American politician who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, he was the first African-American president of the United States."


## Includes
#
include ./zenbuild/main.mk


## Targets
#
.PHONY:			appinfo
appinfo:
			@echo "app-resources-dir: $(RESOURCES_DIR)"

# download [spacy models](https://spacy.io/models/en)
.PHONY:			modeldeps
modeldeps:
			$(PIP_BIN) install $(PIP_ARGS) -r $(PY_SRC)/requirements-model.txt

# requirements for scoring (i.e. WLK)
.PHONY:			scoredeps
scoredeps:
			$(PIP_BIN) install $(PIP_ARGS) -r $(PY_SRC)/requirements-score.txt

# test parsing text
.PHONY:			testparse
testparse:
			$(ABIN) parse $(TEST_TEXT)

# test plotting text
.PHONY:			testplot
testplot:
			$(ABIN) plot $(TEST_TEXT)

# run all examples
.PHONY:			testexample
testexample:
			( for i in example/*.py ; do PYTHONPATH=src/python ./$$i ; done )

# unit and integration testing
.PHONY:			testall
testall:		test testparse testplot testexample

# generate AMR plots of the little prince and the biomedical corpora
.PHONY:			renderexamples
renderexamples:
			$(ABIN) plotfile $(MODEL_FILE)

# train on the little prince corpus
.PHONY:			trainmodel
trainmodel:
			$(ABIN) --config $(MODEL_CONF_DIR)/parse-spring.conf \
				train

# inference a new trained model
.PHONY:			testmodel
testmodel:
			$(ABIN) --config $(INF_CONF) parse $(TEST_TEXT)

# evaluation model "EVAL_MODEL"
.PHONY:			evalmodel
evalmodel:
			$(ABIN) prep
			$(ABIN) parsefile $(EVAL_FILE) --config $(INF_CONF) \
				--override amr_default.parse_model=$(EVAL_MODEL)
			$(ABIN) score $(EVAL_FILE) --config $(INF_CONF) \
				--override amr_default.parse_model=$(EVAL_MODEL)

# evaluate the corpus on the trained little prince corpus (test)
.PHONY:			evalspring
evalspring:
			make EVAL_MODEL=zsl_spring evalmodel

# stop any training
.PHONY:			stop
stop:
			ps -eaf | grep python | grep $(ABIN) | \
				awk '{print $2}' | xargs kill

# additional clean up using the harness/API (data dir)
.PHONY:			cleanalldep
cleanalldep:
			rm -fr data

# clean data generated by examples
.PHONY:			cleanexample
cleanexample:		pycleancache
			rm -fr example/data

# remove the ./data and ./corpus dir
.PHONY:			vaporize
vaporize:		cleanall
			rm -rf corpus
