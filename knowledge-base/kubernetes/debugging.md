# Kubernetes Debugging Guide

## Essential Debugging Commands

### Pod Status
```bash
# Get all pods with status
kubectl get pods -o wide

# Get pods in all namespaces
kubectl get pods -A

# Watch pods in real-time
kubectl get pods -w

# Get pod details with events
kubectl describe pod <pod-name>
```

### Container Logs
```bash
# View current logs
kubectl logs <pod-name>

# Follow logs in real-time (like tail -f)
kubectl logs -f <pod-name>

# View logs from a previous crashed container
kubectl logs <pod-name> --previous

# View logs from specific container in multi-container pod
kubectl logs <pod-name> -c <container-name>

# View last 100 lines
kubectl logs <pod-name> --tail=100

# View logs from last 5 minutes
kubectl logs <pod-name> --since=5m
```

### Shell Access
```bash
# Get a bash shell in a running container
kubectl exec -it <pod-name> -- /bin/bash

# If bash not available, try sh
kubectl exec -it <pod-name> -- /bin/sh

# Run a specific command
kubectl exec <pod-name> -- cat /app/config.yaml

# Multi-container pod
kubectl exec -it <pod-name> -c <container-name> -- /bin/sh
```

## Debugging Ephemeral Containers

For distroless or minimal images with no shell:
```bash
# Add a debug container to a running pod
kubectl debug -it <pod-name> --image=busybox --target=<container-name>

# Create a debugging copy of the pod
kubectl debug <pod-name> --copy-to=debug-pod --image=ubuntu
```

## Resource Issues

### Check Node Resources
```bash
# See resource usage per node
kubectl top nodes

# See resource usage per pod
kubectl top pods

# Check what's allocated on a node
kubectl describe node <node-name> | grep -A 5 "Allocated"
```

### Check Resource Limits
```bash
# See resource requests/limits for a pod
kubectl get pod <pod-name> -o jsonpath='{.spec.containers[*].resources}'
```

## Common Debugging Workflow

1. **What's the status?** `kubectl get pods`
2. **What happened?** `kubectl describe pod <name>`
3. **What did it say?** `kubectl logs <name>`
4. **Can I get in?** `kubectl exec -it <name> -- /bin/sh`
5. **What's using resources?** `kubectl top pods`
6. **What's the network?** `kubectl get svc,ep`
