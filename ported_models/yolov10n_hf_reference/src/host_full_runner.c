#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ref_runtime.h"
#include "slice_manifest.h"


static int read_exact(const char *path, uint8_t *destination, size_t expected)
{
    FILE *file = fopen(path, "rb");
    size_t got;
    int extra;
    if (file == NULL) {
        fprintf(stderr, "error: cannot open %s: errno=%d\n", path, errno);
        return 0;
    }
    got = fread(destination, 1, expected, file);
    extra = fgetc(file);
    if (fclose(file) != 0 || got != expected || extra != EOF) {
        fprintf(stderr, "error: %s size mismatch; expected %zu bytes\n",
                path, expected);
        return 0;
    }
    return 1;
}


static int write_exact(const char *path, const uint8_t *source, size_t bytes)
{
    FILE *file = fopen(path, "wb");
    size_t wrote;
    if (file == NULL) {
        fprintf(stderr, "error: cannot create %s: errno=%d\n", path, errno);
        return 0;
    }
    wrote = fwrite(source, 1, bytes, file);
    if (fclose(file) != 0 || wrote != bytes) {
        fprintf(stderr, "error: short write to %s\n", path);
        return 0;
    }
    return 1;
}


int main(int argc, char **argv)
{
    uint8_t *memory;
    struct yr_result_header *result;
    uint32_t status;
    if (argc != 4) {
        fprintf(stderr, "usage: %s INPUTS.BIN WEIGHTS.BIN DUMP.BIN\n", argv[0]);
        return 2;
    }
    memory = (uint8_t *)calloc(1, YR_MEM_SIZE);
    if (memory == NULL) {
        fprintf(stderr, "error: allocation of %u bytes failed\n", YR_MEM_SIZE);
        return 2;
    }
    if (!read_exact(argv[1], memory + YR_INPUT_DEVICE_OFFSET,
                    YR_INPUT_BLOB_BYTES)
        || !read_exact(argv[2], memory + YR_WEIGHT_DEVICE_OFFSET,
                       YR_WEIGHT_BLOB_BYTES)) {
        free(memory);
        return 2;
    }
    result = (struct yr_result_header *)(memory + YR_RESULT_DEVICE_OFFSET);
    status = yr_prepare_result(memory, result);
    if (status != YR_STATUS_OK) {
        fprintf(stderr, "error: generated full manifest is invalid\n");
        free(memory);
        return 1;
    }
    status = yr_run_selected(memory, result);
    yr_finalize_result(memory, result);
    if (!write_exact(argv[3], memory, YR_DUMP_SIZE)) {
        free(memory);
        return 2;
    }
    printf(
        "HOST_FULL %s nodes=N%03u:N%03u status=%u "
        "workspace_fnv1a=%016llx dump=%s\n",
        status == YR_STATUS_OK ? "PASS" : "FAIL",
        YR_FIRST_NODE, YR_LAST_NODE, status,
        (unsigned long long)result->workspace_fnv1a, argv[3]);
    free(memory);
    return status == YR_STATUS_OK ? 0 : 1;
}
