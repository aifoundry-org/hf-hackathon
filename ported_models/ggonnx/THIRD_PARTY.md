# Third-Party Source

## GGONNX

- Upstream: `https://github.com/marty1885/ggonnx`
- Vendored revision: `c0453fa516af3e2ae4140fafa798fccb6daff46c`
- Vendored date: 2026-05-02
- Vendored path: `ported_models/ggonnx/src/ggonnx/`
- License at time of vendoring: **none declared**

### License status

At the time this source tree was vendored, the upstream public repository did
not include a LICENSE file and the GitHub repository metadata did not declare a
license. The author (Martin Chang, marty1885) is a collaborator on the nekko
ecosystem — the vendor name `"nekko"` is baked into
`src/ggml_execution_provider.cc` — and his other ET-ecosystem repositories are
typically Apache-2.0 licensed (`et-testdrive`, `etTopoScan`, `etBenchBed`,
`etVecAdd`, `tarnet`, `nina`, `et-profile-convert`) or MIT (his `llama.cpp` fork).
The absence of a LICENSE file in ggonnx appears to be an oversight, not intent.

### Action required

Replace this note with a canonical license attribution as soon as the upstream
repository receives a LICENSE commit. Expected outcome (matching Martin's
other ET work): Apache-2.0. Once that lands:

1. Bump the vendored copy to the SHA that includes the LICENSE file.
2. Copy the upstream `LICENSE` file into this folder alongside the vendored
   `src/ggonnx/LICENSE`.
3. Rewrite this section with the canonical SPDX identifier and attribution.

This source remains the property of the original author (Martin Chang). It is
vendored here with the author's involvement as a nekko-ecosystem collaborator;
the maintainers are securing an explicit license grant for redistribution
(expected Apache-2.0, matching the author's other ET work). Until the upstream
LICENSE lands, attribution and redistribution terms are as recorded in this
file.
