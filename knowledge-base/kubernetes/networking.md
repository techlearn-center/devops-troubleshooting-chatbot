# Kubernetes Networking Issues

## Service Not Reachable

### Symptoms
- Application returns connection timeout
- `curl` to service IP fails
- DNS resolution fails

### Debugging Steps
```bash
# 1. Check service exists
kubectl get svc

# 2. Check endpoints (pods backing the service)
kubectl get endpoints <service-name>
# If endpoints are empty, check pod labels match service selector!

# 3. Test DNS resolution
kubectl exec <pod> -- nslookup <service-name>

# 4. Test connectivity
kubectl exec <pod> -- curl -v <service-name>:<port>
```

### Common Causes

#### No Endpoints
**Problem:** Service selector doesn't match any pod labels.
```bash
# Check service selector
kubectl describe svc my-service
# Selector: app=myapp

# Check pod labels
kubectl get pods --show-labels
# Make sure labels match!
```

#### Wrong Port
**Problem:** Service port doesn't match container port.
```yaml
# Service
spec:
  ports:
    - port: 80         # Service port (what clients use)
      targetPort: 8080  # Container port (where app listens)
```

## DNS Issues

### Error: `Could not resolve host`

```bash
# Check CoreDNS is running
kubectl get pods -n kube-system -l k8s-app=kube-dns

# Test DNS
kubectl exec <pod> -- nslookup kubernetes.default

# Check DNS policy in pod spec
spec:
  dnsPolicy: ClusterFirst  # Default, uses cluster DNS
```

## Network Policies

### Pods Can't Communicate
**Cause:** Network policy is blocking traffic.

```bash
# List network policies
kubectl get networkpolicies

# Check if any policy affects your pod
kubectl describe networkpolicy <name>
```

### Allow Specific Traffic
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend
spec:
  podSelector:
    matchLabels:
      app: backend
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: frontend
      ports:
        - port: 8080
```

## Ingress Issues

### 404 Not Found from Ingress

```bash
# Check ingress configuration
kubectl describe ingress <name>

# Common fixes:
# 1. Verify backend service exists
# 2. Check path routing rules
# 3. Verify ingress controller is running
kubectl get pods -n ingress-nginx
```
