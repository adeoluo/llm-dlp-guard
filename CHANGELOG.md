# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.1.1] 2026-11-08

### Documentation

- Complete citation metadata and bump it on release (#10)
- Map the threat model to OWASP LLM 2026 and MITRE ATLAS v2026.10 (#12)
- Add standards keywords to the citation metadata (#13)

### Maintenance

- Latest toolchain and CI versions (#9)
- Release script keeps hard-coded version strings in step (#11)

## [0.1.0] 2026-11-07

### Features

- Pattern engine, validators and CRF tagger with a Rust core (#1)
- Policy engine, reversible vault and streaming redaction (#2)
- OpenAI-compatible proxy, HTTP service and CLI (#3)
- Evaluation benchmarks, docs and ADRs (#4)
- Scaling benchmarks, intervals, ablations and seeded secrets benchmark (#5)

### Performance

- Throughput scaling curves and median of three timings (#7)

### Maintenance

- Scaffold project
- Release tooling, agent notes and a make evaluation target (#6)
- Build the sdist and wheel as release assets (#8)
