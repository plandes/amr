# Examples

This directory has API coding and configuration examples.


## API examples

* `appctx.py`: shows how to use the AMR API using a custom application context
* `simple.py`: uses the parser from the the command line application


## Configuration

The `train` directory contains configuration files to train AMR models and
inference.  These are used in the `../makefile` build automation to both train
(`trainlp`) and test (`evallp`).  The example uses the freely distributed
*Little Prince* ISI AMR annotations.

* `lp.conf`: the shared file that configures the corpus download
* `parse-*-lp.conf`: training config files for several AMR parsing models
* `generate-t5wtense-lp.conf`: training config for a graph to text model
* `inference.conf`: config to use for inferencing text to AMR (set the model to
  use in `amr_default`)
