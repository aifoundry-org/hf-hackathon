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
