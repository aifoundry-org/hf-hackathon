# Third-Party Source

## ET llama.cpp

- Source: `https://github.com/aifoundry-org/llama.cpp.git`
- Branch: `et`
- Pinned revision: `13da97147c0c8c04f0df410e77c8d202cebee3fb`
- Local pointer: `src/llama.cpp-et`
- License: MIT

The MIT license permits copying, modifying, publishing, distributing,
sublicensing, and selling copies of the software, provided the copyright notice
and permission notice are included in all copies or substantial portions.

We keep the framework as a pinned submodule instead of vendoring a raw source
copy. A raw vendor copy would add roughly 134 MB and more than 2,500 files to
this repository before history, while making framework updates harder to audit.

**CI builds llama.cpp from the committed submodule.** The benchmark runner has
no URL-clone fallback for framework source — what's pinned in the PR is what
runs on the board.

### TODO: uberkernel variant

Some earlier board runs used a private working tree with the "uberkernel"
optimization enabled. That variant is not in the public upstream — no public
`uberkernel` branch exists in the ET fork. Once such a branch is published (or
the optimization is upstreamed), repin this submodule to it so CI runs that
variant instead of the public `et` branch.

## GGONNX

GGONNX is vendored as its own port under `ported_models/ggonnx/`. Its licensing
record lives in `ported_models/ggonnx/THIRD_PARTY.md` — refer to that file for the
authoritative status. Do not duplicate the licensing terms here.
