/*
 * Single translation unit for the board CI leaderboard build, which compiles
 * one source file per configured model. The scripts in scripts/ compile these
 * units separately with the generated manifest on the include path; this file
 * pins the committed full-graph manifest by relative path instead, so the CI
 * build does not depend on the working directory it runs from. It builds the
 * same three units the script build does, which keeps the benchmarked binary
 * identical to the one validated by tools/compare_full.py.
 *
 * YR_SLICE_MANIFEST_PREINCLUDED tells the units the manifest is already in
 * scope. YR_CONV_TENSOR_STRONG_PRESENT drops the weak yr_conv_tensor() stubs in
 * ref_runtime.c, because the strong definitions in yr_conv_tensor_et.c cannot
 * share a translation unit with the weak ones they replace; the separate
 * compilation used by the scripts still resolves that override at link time.
 * The tensor path stays inert either way while YR_CONV_TENSOR_ENABLED is 0.
 *
 * The three included units share no static names, so combining them changes no
 * linkage.
 */
#include "../generated/full_graph/slice_manifest.h"
#define YR_SLICE_MANIFEST_PREINCLUDED 1
#define YR_CONV_TENSOR_STRONG_PRESENT 1

#include "ref_runtime.c"
#include "yr_conv_tensor_et.c"
#include "et_slice_runner.c"
