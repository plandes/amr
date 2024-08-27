# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]


### Changed
- Reset the default AMR parser to the [amrlib] XFM Bart Base model.  This was
  switched from the GSII model since the authors have not made that model
  downloadable by robots.  This was done for usage of the library and fix the
  unit tests in CLI GitHub workflows.
- Fix spaCy adapted document missing token index when adding AMR metadata.
  This happens when tokens are removed by a filter, as happened with a stop
  word filter in a unit test case.  This also fixes that unit test case.


## [0.1.4] - 2024-07-03
### Added
- `AmrDocument` infers its document ID from the first contained sentence.
- Added feature sentence and documents from AMR analogues in
  `AnnotatedAmrFeatureDocumentFactory`.


## [0.1.3] - 2024-05-16
### Added
- "A utility class to reindex variables in an `AmrDocument` and a decorator for
  it.
- Integration test automation.

### Changed
- A new feature, when enabled, robustly deals with alignment errors.
- Standardized resource library paths.
- Move AMR token decorated literals formatted attributes inside the quotes.
- Fix missing epigraph data for new token decorated nodes.


## [0.1.2] - 2024-04-15
### Added
- Feature to robustly handle co-reference errors by stashing `AmrFailure`s
  instead of the co-reference when errors are raised.
- Token annotator (`TokenAnnotationFeatureDocumentDecorator`) can add to the
  graph epidata or to the node text in addition to adding new attributes.

### Changed
- Invalidate and update graph string after alignment removal.


## [0.1.1] - 2024-03-29
### Added
- AMR parsing and generation model trainers and workflow.
- A framework parser that uses the [zensols.amrspring] (a Docker original
  SPRING parser client/server) as a client.

### Removed
- Pipeline (non-client) classes modules from default imports.

### Changed
- Upgrade model checkpoint training to library [amrlib] 0.7.1.
- Upgrade API to [amrlib] 0.8.0.
- Abstract [amrlib] parsers.


## [0.1.0] - 2023-12-05
### Changed
- Upgrade to [zensols.util] version 1.14.

### Added
- Support for Python 3.11.

### Removed
- Support for Python 3.9.


## [0.0.1] - 2021-09-02
### Added
- Initial version.


<!-- links -->
[Unreleased]: https://github.com/Paul Landes/amr/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/Paul Landes/amr/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/Paul Landes/amr/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Paul Landes/amr/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Paul Landes/amr/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Paul Landes/amr/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/Paul Landes/amr/compare/v0.0.0...v0.0.1

[amrlib]: https://github.com/bjascob/amrlib
[zensols.amrspring]: https://github.com/plandes/amrspring
