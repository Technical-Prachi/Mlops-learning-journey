# Load Testing, Autoscaling and Observability

**Goal:** Find out how the deployed API behaves under heavy load.

**What I did**
- Load tested the API with `wrk` (30 s, 4 threads). Raw output is in `load-testing-wrk-results/`.
- Configured a Horizontal Pod Autoscaler (`hpa.yaml`, CPU target 50%).
- Observed CPU, memory and request logs in Cloud Monitoring and Cloud Logging.
- Compared two scenarios: autoscaling up to 3 pods versus capped at 1 pod.

**Measured results**

| Connections | Requests/sec | Avg latency | Timeouts |
|---|---|---|---|
| 1000 | 80.1 | 1.13 s | 413 |
| 2000 | 106.8 | 1.64 s | 3090 |

**Findings**
- Under heavy load the service became capacity-constrained: latency rose and many requests timed out.
- With autoscaling capped at one pod, CPU saturated and timeouts increased. Allowing the HPA to add pods improved throughput and reduced errors, according to my notes from the test run.

**What I learned**
- Autoscaling limits directly decide how much load a service can absorb.
- Load testing reveals bottlenecks that functional tests never show.
