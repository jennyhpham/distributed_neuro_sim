# Running the K3d Concurrent Load Balancing Benchmark

This benchmark demonstrates K3d's Service load balancing under concurrent SNN
inference requests (RQ2). It runs three scenarios back-to-back:

| Scenario | Replicas | Clients | Expected result |
|---|---|---|---|
| Baseline | 1 | 1 | ~900ms p50, ~1 req/s |
| Queuing | 1 | 3 | ~2700ms p50 (×3), same throughput |
| Scaling | 3 | 3 | ~900ms p50 (back to baseline), ~2× throughput |

Output: `results/k3d_concurrent.json`

---

## Prerequisites

- Docker Desktop running on Windows
- `k3d` and `kubectl` installed on the Windows host
- `neuro-sim:latest` Docker image built (`docker build -t neuro-sim:latest .`)
- This repo cloned, with the devcontainer running in VS Code

---

## Part 1 — Cluster setup (Windows PowerShell, once only)

### 1. Create the cluster

```powershell
k3d cluster create neuro-sim --port "9090:80@loadbalancer" --api-port 0.0.0.0:6550
```

### 2. Fix the kubeconfig to use 127.0.0.1

```powershell
k3d kubeconfig merge neuro-sim --kubeconfig-switch-context
kubectl config set-cluster k3d-neuro-sim --server=https://127.0.0.1:6550
```

Verify connectivity:

```powershell
kubectl cluster-info
# Expected: "Kubernetes control plane is running at https://127.0.0.1:6550"
```

### 3. Import the Docker image into the cluster

K3d cannot pull images from the local Docker daemon automatically — they must
be imported explicitly:

```powershell
k3d image import neuro-sim:latest -c neuro-sim
```

> **Note:** Repeat this step every time you rebuild `neuro-sim:latest`.

### 4. Apply Kubernetes manifests

```powershell
kubectl apply -f k3d/namespace.yaml
kubectl apply -f k3d/pvc.yaml
kubectl apply -f k3d/deployment.yaml
kubectl apply -f k3d/service.yaml
kubectl apply -f k3d/ingress.yaml
```

### 5. Wait for the pod to be ready

```powershell
kubectl get pods -n neuro-sim -w
```

Wait until the pod shows `1/1 Running`. This takes ~30–60s (readiness probe
has a 20s initial delay, then checks every 10s).

### 6. Verify the endpoint is reachable

```powershell
curl http://localhost:9090/health
# Expected: {"status":"healthy","model":"IncreasingInhibitionNetwork",...}
```

---

## Part 2 — Run the benchmark (devcontainer terminal)

The script is interactive — it pauses between scenarios so you can scale
replicas on the host.

```bash
python scripts/run_concurrent_benchmark.py --url http://host.docker.internal:9090
```

### Step 1 of 3 — 1 replica / 1 client

The script will print:

```
[Step 1/3] Ensure 1 replica is running, then press Enter…
  kubectl scale deployment neuro-sim --replicas=1 -n neuro-sim
  kubectl wait --for=condition=available --timeout=60s deployment/neuro-sim -n neuro-sim
```

Switch to a **host PowerShell terminal** and run:

```powershell
kubectl scale deployment neuro-sim --replicas=1 -n neuro-sim
kubectl wait --for=condition=available --timeout=60s deployment/neuro-sim -n neuro-sim
```

Then press **Enter** in the devcontainer terminal. The benchmark runs 50
sequential requests and prints latency + throughput.

### Step 2 of 3 — 1 replica / 3 clients

The script pauses again. No scaling needed — just press **Enter**. Three
concurrent workers fire simultaneously at the single pod. Expect latency to
roughly triple and throughput to stay flat (requests queue).

> This step takes ~2–3 minutes with the 1000m CPU limit.

### Step 3 of 3 — 3 replicas / 3 clients

The script prints:

```
[Step 3/3] Scale to 3 replicas, then press Enter…
  kubectl scale deployment neuro-sim --replicas=3 -n neuro-sim
  kubectl wait --for=condition=available --timeout=120s deployment/neuro-sim -n neuro-sim
```

Switch to **host PowerShell** and run:

```powershell
kubectl scale deployment neuro-sim --replicas=3 -n neuro-sim
kubectl wait --for=condition=available --timeout=120s deployment/neuro-sim -n neuro-sim
```

Then press **Enter**. The three workers each hit a separate pod. Expect latency
to drop back toward the baseline and throughput to increase.

---

## Part 3 — Generate plots (devcontainer)

```bash
python scripts/plot_rq2.py
```

Output figures are saved to `results/figures/`:

| Figure | Content |
|---|---|
| `rq2_latency_comparison.png` | Baseline vs K3d 1r vs K3d 3r (split panels) |
| `rq2_unified_overhead.png` | Overhead across all scenarios (log scale) |
| `rq2_replica_scaling.png` | Throughput + p50 for 1r vs 3r |
| `rq2_concurrent_scaling.png` | Latency and throughput across the 3 concurrent scenarios |

---

## Troubleshooting

**`curl http://localhost:9090/health` returns 404**
Traefik is running but has no route. Apply the ingress:
```powershell
kubectl apply -f k3d/ingress.yaml
```

**Pod stuck in `Pending`**
```powershell
kubectl describe pod -n neuro-sim
```
Usually means the image wasn't imported. Re-run:
```powershell
k3d image import neuro-sim:latest -c neuro-sim
```

**`kubectl cluster-info` times out**
The kubeconfig has a stale API server address. Re-run:
```powershell
k3d kubeconfig merge neuro-sim --kubeconfig-switch-context
kubectl config set-cluster k3d-neuro-sim --server=https://127.0.0.1:6550
```

**Step 2 (1r/3c) returns 404 mid-run**
The pod was marked unready (readiness probe missed under CPU load). Check:
```powershell
kubectl get pods -n neuro-sim
kubectl describe pod <pod-name> -n neuro-sim
```
Wait for it to become ready again, then re-run the script.

---

## Tearing down

```powershell
k3d cluster delete neuro-sim
```

This removes all K3d containers. The `neuro-sim:latest` Docker image is
unaffected.
