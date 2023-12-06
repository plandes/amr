# AMR annotation and feature generation

[![PyPI][pypi-badge]][pypi-link]
[![Python 3.10][python310-badge]][python310-link]
[![Python 3.11][python311-badge]][python311-link]


Provides support for AMR annotations and feature generation.

Features:
- Annotation in AMR metadata.  For example, sentence types found in the Proxy
  report AMR corpus.
- AMR token alignment as [spaCy] components.
- A scoring API that includes [Smatch] and [WLK], which extends a more general
  [NLP scoring module].
- AMR parsing ([amrlib]) and AMR co-reference ([amr_coref]).
- Command line and API utilities for AMR graph Penman graphs, debugging and
  files.
- Tools for training and evaluating AMR parse models.


## Documentation

* [full documentation](https://plandes.github.io/amr/index.html).
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
from zensols.amr import AmrDocument, AmrSentence, ApplicationFactory

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
```bash
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

## Performance (Smatch)

This repo is configured to download and train on the AMR bio-medical corpus.
The results of the scores using amrlib's default smatch score is:

| Corpus | Model           | Precision          | Recall             | F-score             |
|--------|-----------------|--------------------|--------------------|---------------------|
| bio    | amrlib t5       | 0.5613647022821542 | 0.4799029769470724 | 0.5174473330001563  |
| bio    | amrlib t5 + bio | 0.6792187759112143 | 0.6164372669678633 | 0.6463069704295633  |



## Attribution

This project, or reference model code, uses:

* Python 3
* [spaCy] for natural language parsing.
* [zensols.nlparse] for natural language features.
* [amrlib] for AMR parsing.
* [amr_coref] for AMR co-reference
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

Copyright (c) 2021 - 2023 Paul Landes


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
