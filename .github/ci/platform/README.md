# ET job platform

Secure job queue for ET-SoC1 hackathon benchmarks (sim + optional board).

## Quick test on your PC (~10s)

```bash
cd .github/ci/platform
./deploy/run-local.sh --test
```

Uses `ET_JOBS_DRY_RUN=1` by default (`deploy/config.local.env`). API: http://127.0.0.1:18080/docs

## Quick test on board-host

```bash
cp deploy/config.env.example deploy/config.env
# edit WORKER_TOKEN, paths

./deploy/install-board-host.sh deploy/config.env
source deploy/config.env
export PYTHONPATH="$(pwd)"

# terminal 1
python3 -m et_jobs api

# terminal 2
python3 -m et_jobs worker --pool board

# terminal 3
./deploy/test-e2e.sh
```

## Public deploy

```bash
export WORKER_TOKEN=$(openssl rand -hex 32)
docker compose -f deploy/docker-compose.public.yml up -d
```

Board host: set `JOBS_API_URL=https://your-public-api` and run `worker --pool board`.

## Board recovery policy

CI never resets or power-cycles an ET-SoC1. Each direct board session runs a
checksum-pinned SDK `empty.elf` preflight, verifies a deterministic memory dump,
and fails closed if the card is not healthy. A runtime or firmware error writes
`/var/lib/et-soc1-ci/quarantine`; every later job refuses to touch the card.

Recovery is manual: stop the Actions runner, diagnose the first failure, perform
one external power cycle, then run
`.github/ci/platform/deploy/clear-board-quarantine.sh --after-external-power-cycle`.
The command refuses to clear quarantine unless the host boot ID changed and the
new boot has no ET errors. Re-enable the runner only after the deterministic
preflight passes.

The root-owned Actions service must also be provisioned with
`deploy/install-actions-runner-safety.sh`. Its systemd policy makes kernel
controls read-only, removes mount/reboot/module capabilities and syscalls, and
denies network access to the iBoot controller. The ET SDK and source tree under
`/opt` are also read-only to the runner. The policy applies to every process
launched by CI. The installer never starts or restarts the runner. Board
workflows also require local transport so they cannot escape that service
sandbox through a second SSH session.

The llama.cpp ET backend drains its stream before 4,096 queued kernels, far
below the runtime's 65,536-value event namespace. This is defense in depth, not
a substitute for the allocator fix merged in `aifoundry-org/et-platform#134`.
Every board job verifies the exact source revision plus both the shared
`libetrt.so` and static `libetrt_static.a` hashes in
`.github/ci/reference/et_runtime.json` before building candidate code. The
static archive matters because `libggml-et.so` incorporates it at link time.
Provision those audited libraries with `deploy/install-et-runtime-contract.sh`;
the installer archives the previous libraries, writes a root-owned manifest,
and never starts the runner or accesses the card.
