# Acceptance Checklist

1.  [ ] `helm upgrade --install xorb-stack ./charts/xorb-stack -f values.single-vps.yaml` succeeds.
2.  [ ] `kubectl get pods` all `Running`; total RSS < 12 Gi.
3.  [ ] `curl https://api.xorb.example.com/auth/token` returns JWT (TLS cert valid).
4.  [ ] Grafana dashboards show API request rate & p99 latency.
5.  [ ] Tempo search returns traces for `/targets` workflow path.
6.  [ ] Flux reconciles within 1 min after pushing a new tag.
