# Configuration

This directory contains configuration files to train AMR models and inference.
These are used in the `../makefile` build automation to both train
(`trainmodelp`) and test (`evalmodel`).  The default configuration uses the
freely distributed *Little Prince* and *Bio AMR* corpora [ISI] AMR annotations.

* `lp.conf`: the shared file that configures the corpus download
* `parse-*.conf`: training config files for several AMR parsing models
* `generate-t5wtense.conf`: training config for a graph to text model
* `inference.conf`: config to use for inferencing text to AMR (set the model to
  use in `amr_default`)


## Customization

The models are named with a generic `zsl` (short for `zensols`) string followed
by the model.  This directory can be copied and `zsl` global replaced to your
own string to customize your own models.


<!-- links -->
ISI: https://amr.isi.edu/download.html
