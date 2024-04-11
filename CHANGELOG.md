# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]


### Added
- Feature to robustly handle co-reference errors by stashing `AmrFailure`s
  instead of the co-reference when errors are raised.


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
[Unreleased]: https://github.com/Paul Landes/amr/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Paul Landes/amr/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Paul Landes/amr/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/Paul Landes/amr/compare/v0.0.0...v0.0.1

[amrlib]: https://github.com/bjascob/amrlib
[zensols.amrspring]: https://github.com/plandes/amrspring
