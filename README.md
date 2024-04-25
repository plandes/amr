# AMR annotation and feature generation

[![PyPI][pypi-badge]][pypi-link]
[![Python 3.10][python310-badge]][python310-link]
[![Python 3.11][python311-badge]][python311-link]

Provides support for AMR graph manipulation, annotations and feature
generation.

Features:
* Annotation in AMR metadata.  For example, sentence types found in the Proxy
  report AMR corpus.
* AMR token alignment as [spaCy] components.
* Integrates natural language parsing and features with Zensols
  [zensols.nlparse] library.
* A scoring API that includes [Smatch] and [WLK], which extends a more general
  [NLP scoring module].
* AMR parsing ([amrlib]) and AMR co-reference ([amr_coref]).
* Command line and API utilities for AMR graph Penman graphs, debugging and
  files.
* Tools for [training and evaluating](training) new AMR parse (text to graph)
  and generation (graph to text) models.
* A method for re-indexing and updating AMR graph variables so that all in a
  document collection are unique.


## Documentation

* [Full documentation](https://plandes.github.io/amr/index.html).
* [API reference](https://plandes.github.io/amr/api.html)


## Obtaining

The easiest way to install the command line program is via the `pip` installer:
```bash
pip3 install zensols.amr
```

Binaries are also available on [pypi].


## Usage

```python
from penman.graph import Graph
from zensols.nlp import FeatureDocument, FeatureDocumentParser
from zensols.amr import AmrDocument, AmrSentence, Dumper, ApplicationFactory

sent: str = """

He was George Washington and first president of the United States.
He was born On February 22, 1732.

""".replace('\n', ' ').strip()

# get the AMR document parser
doc_parser: FeatureDocumentParser = ApplicationFactory.get_doc_parser()

# the parser creates a NLP centric feature document as provided in the
# zensols.nlp package
doc: FeatureDocument = doc_parser(sent)

# the AMR object graph data structure is provided in the feature document
amr_doc: AmrDocument = doc.amr

# dump a human readable output of the AMR document
amr_doc.write()

# get the first AMR sentence instance
amr_sent: AmrSentence = amr_doc.sents[0]
print('sentence:')
print(' ', amr_sent.text)
print('tuples:')

# show the Penman graph representation
pgraph: Graph = amr_sent.graph
print(f'variables: {", ".join(pgraph.variables())}')
for t in pgraph.triples:
    print(' ', t)
print('edges:')
for e in pgraph.edges():
    print(' ', e)

# visualize the graph as a PDF
dumper: Dumper = ApplicationFactory.get_dumper()
dumper(amr_doc)
```

Per the example, the [t5.conf](test-resources/t5.conf) and
[gsii.conf](test-resources/gsii.conf) configuration show how to include
configuration needed per AMR model.  These files can also be used directly with
the `amr` command using the `--config` option.

However, the other resources in the example must be imported unless you
redefine them yourself.


### Library

When adding the `amr` spaCy pipeline component, the `doc._.amr` attribute is
set on the `Doc` instance.  You can either configure spaCy yourself, or you can
use the configuration files in [test-resources](test-resources) as an example
using the [zensols.util configuration framework].  The command line application
provides an example how to do this, along with the [test
case](test/python/test_amr.py).


### Command Line

This library is written mostly to be used by other program, but the command
line utility `amr` is also available to demonstrate its usage and to generate
ARM graphs on the command line.

To parse:
```lisp
$ amr parse -c test-resources/t5.conf 'This is a test of the AMR command line utility.'
# ::snt This is a test of the AMR command line utility.
(t / test-01
   :ARG1 (u / utility
            :mod (c / command-line)
            :name (n / name
                     :op1 "AMR"
                     :toki1 "6")
            :toki1 "9")
   :domain (t2 / this
               :toki1 "0")
   :toki1 "3")
```

To generate graphs in PDF format:
```bash
$ amr plot -c test-resources/t5.conf 'This is a test of the AMR command line utility.'
wrote: amr-graph/this-is-a-test-of-the-amr-comm.pdf
```


## Training

This package uses the [amrlib] training, but adds a command line and
downloadable corpus aggregation / API.  To train:

1. Choose a model (i.e. SPRING, T5).
1. Optionally edit the [train configuration](train-config) directory of the
   model you choose.
1. Optionally edit the `resources/train.yml` to select/add more corpora (see
   [Adding Corpora](adding-corpora)).
1. Train the model: `./amr --config train-config/<model>.conf`


### Pretrained Models

This library was used to train all of the [amrlib] models (using the same
checkpoints as [amrlib]), except the T5 Base v1 model, with additional
examples from publicly available human annotated corpora.  The differences of
these trained models include:

* None of the models were tested against a training set, only the development
  SMATCH scores are available.  This was intentional to provide more training
  examples.
* The AMR Release 3.0 ([LDC2020T02]) test set was added to the training set.
* The [Little Prince and Bio AMR](https://amr.isi.edu/download.html) corpora
  where used to train the models.  The first 85% of the AMR sentences were
  added to training set and the remaining 15% were added to the development
  set.
* The mini-batch size changed for `generate-t5wtense-base` due to memory
  constraints.
* The number of training epochs were increased to account for the additional
  number of training examples.
* Models have the same naming conventions but are prefixed with `zsl`.
* Generative models were trained on graphs metadata annotated by the Sci spaCy
  `en_core_sci_md` model.

The performance of these models:

| Model Name           | Model Type | Checkpoint             | Performance   |
|----------------------|------------|------------------------|---------------|
| `zsl_spring`         | parse      | [facebook/bart-large]  | SMATCH: 81.26 |
| `zsl_xfm_bart_base`  | parse      | [facebook/bart-base]   | SMATCH: 80.5  |
| `zsl_xfm_bart_large` | parse      | [facebook/bart-large]  | SMATCH: 82.7  |
| `zsl_t5wtense_base`  | generative | [t5-base]              | BLEU: 42.20   |
| `zsl_t5wtense_large` | generative | [google/flan-t5-large] | BLEU: 44.01   |

These models are available upon request.


### Adding Corpora

You can retrain your own model and add additional training corpora by modifying
the list of `${amr_prep_manager:preppers}` in `resources/train.yml`.  This file
defines downloaded corpora for the Little Prince and Bio AMR corpora.  To use
the AMR 3.0 release, add the LDC downloaded file to (a new) `download`
directory.


## Attribution

This project, or reference model code, uses:

* Python 3.11
* [amrlib] for AMR parsing.
* [amr_coref] for AMR co-reference
* [spaCy] for natural language parsing.
* [zensols.nlparse] for natural language features.
* [Smatch] (Cai and Knight. 2013) and [WLK] (Opitz et. al. 2021) for scoring.


## Citation

If you use this project in your research please use the following BibTeX entry:

```bibtex
@inproceedings{landes-etal-2023-deepzensols,
	title = "{D}eep{Z}ensols: A Deep Learning Natural Language Processing Framework for Experimentation and Reproducibility",
	author = "Landes, Paul  and
	  Di Eugenio, Barbara  and
	  Caragea, Cornelia",
	editor = "Tan, Liling  and
	  Milajevs, Dmitrijs  and
	  Chauhan, Geeticka  and
	  Gwinnup, Jeremy  and
	  Rippeth, Elijah",
	booktitle = "Proceedings of the 3rd Workshop for Natural Language Processing Open Source Software (NLP-OSS 2023)",
	month = dec,
	year = "2023",
	address = "Singapore, Singapore",
	publisher = "Empirical Methods in Natural Language Processing",
	url = "https://aclanthology.org/2023.nlposs-1.16",
	pages = "141--146"
}
```


## Changelog

An extensive changelog is available [here](CHANGELOG.md).


## Community

Please star this repository and let me know how and where you use this API.
Contributions as pull requests, feedback and any input is welcome.


## License

[MIT License](LICENSE.md)

Copyright (c) 2021 - 2024 Paul Landes


<!-- links -->
[pypi]: https://pypi.org/project/zensols.amr/
[pypi-link]: https://pypi.python.org/pypi/zensols.amr
[pypi-badge]: https://img.shields.io/pypi/v/zensols.amr.svg
[python37-badge]: https://img.shields.io/badge/python-3.7-blue.svg
[python37-link]: https://www.python.org/downloads/release/python-370
[python38-badge]: https://img.shields.io/badge/python-3.8-blue.svg
[python38-link]: https://www.python.org/downloads/release/python-380
[python310-badge]: https://img.shields.io/badge/python-3.10-blue.svg
[python310-link]: https://www.python.org/downloads/release/python-3100
[python311-badge]: https://img.shields.io/badge/python-3.11-blue.svg
[python311-link]: https://www.python.org/downloads/release/python-3110

[spaCy]: https://spacy.io
[amrlib]: https://github.com/bjascob/amrlib
[amr_coref]: https://github.com/bjascob/amr_coref
[Smatch]: https://github.com/snowblink14/smatch
[WLK]: https://github.com/flipz357/weisfeiler-leman-amr-metrics
[zensols.nlparse]: https://github.com/plandes/nlparse
[zensols.util configuration framework]: https://plandes.github.io/util/doc/config.html
[NLP scoring module]: https://plandes.github.io/nlparse/api/zensols.nlp.html#zensols-nlp-score
[LDC2020T02]: https://catalog.ldc.upenn.edu/LDC2020T02

[facebook/bart-large]: https://huggingface.co/facebook/bart-large
[facebook/bart-base]: https://huggingface.co/facebook/bart-base
[t5-base]: https://huggingface.co/google-t5/t5-base
[google/flan-t5-large]: https://huggingface.co/google/flan-t5-large
