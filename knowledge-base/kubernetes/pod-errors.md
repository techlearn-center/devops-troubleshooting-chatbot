# Kubernetes Pod Errors and Solutions

## Error: CrashLoopBackOff

**Error Message:**
```
NAME        READY   STATUS             RESTARTS   AGE
my-pod      0/1     CrashLoopBackOff   5          3m
```

**What It Means:**
The container keeps crashing and Kubernetes keeps restarting it. The "backoff" means Kubernetes is waiting longer between each restart attempt.

**Common Causes:**
1. Application crashes on startup
2. Missing environment variables or secrets
3. Configuration errors
4. Failed health checks
5. Insufficient resources (OOM killed)
6. Missing dependencies or files

**Diagnosis Steps:**

### Step 1: Check pod logs
```bash
# Current logs
kubectl logs my-pod

# Previous container logs (after crash)
kubectl logs my-pod --previous

# Follow logs in real-time
kubectl logs -f my-pod
```

### Step 2: Describe the pod
```bash
kubectl describe pod my-pod
```

Look for:
- **Events section** at the bottom
- **Exit Code** in container status
- **Reason** for container termination

### Step 3: Check exit codes
| Exit Code | Meaning |
|-----------|---------|
| 0 | Success (but pod restarted?) |
| 1 | General application error |
| 137 | OOM Killed (128 + 9 SIGKILL) |
| 139 | Segmentation fault |
| 143 | SIGTERM received |

**Solutions:**

### Solution 1: Fix application errors
```bash
# Check logs for errors
kubectl logs my-pod --previous

# Common fixes:
# - Fix code bugs
# - Add proper error handling
# - Check startup scripts
```

### Solution 2: Add missing environment variables
```yaml
# Check if env vars are set correctly
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: app
    image: my-app:latest
    env:
    - name: DATABASE_URL
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: url
    - name: LOG_LEVEL
      value: "info"
```

### Solution 3: Increase resource limits (if OOM)
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"  # Increase this
    cpu: "500m"
```

### Solution 4: Fix health checks
```yaml
# Make startup probe more lenient
startupProbe:
  httpGet:
    path: /health
    port: 8080
  failureThreshold: 30
  periodSeconds: 10

livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30  # Give app time to start
  periodSeconds: 10
```

---

## Error: ImagePullBackOff

**Error Message:**
```
NAME        READY   STATUS             RESTARTS   AGE
my-pod      0/1     ImagePullBackOff   0          2m
```

**What It Means:**
Kubernetes cannot pull the container image from the registry.

**Common Causes:**
1. Image name or tag is wrong
2. Image doesn't exist in registry
3. No permission to pull (private registry)
4. Registry is unreachable
5. ImagePullSecrets not configured

**Diagnosis:**

```bash
kubectl describe pod my-pod
```

Look for messages like:
- `Failed to pull image`
- `unauthorized: authentication required`
- `manifest unknown`

**Solutions:**

### Solution 1: Verify image name and tag
```bash
# Check if image exists
docker pull my-registry/my-app:v1.0.0

# Common mistakes:
# - Typo in image name
# - Tag doesn't exist (use 'latest' or specific version)
# - Wrong registry URL
```

### Solution 2: Create ImagePullSecret for private registries
```bash
# Create secret
kubectl create secret docker-registry my-registry-secret \
  --docker-server=my-registry.example.com \
  --docker-username=myuser \
  --docker-password=mypass \
  --docker-email=me@example.com

# Use in pod
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  imagePullSecrets:
  - name: my-registry-secret
  containers:
  - name: app
    image: my-registry.example.com/my-app:v1.0.0
```

### Solution 3: Use correct image pull policy
```yaml
containers:
- name: app
  image: my-app:latest
  imagePullPolicy: Always  # Always pull, even if local
  # imagePullPolicy: IfNotPresent  # Use local if exists
  # imagePullPolicy: Never  # Never pull, must be local
```

---

## Error: Pending State

**Error Message:**
```
NAME        READY   STATUS    RESTARTS   AGE
my-pod      0/1     Pending   0          5m
```

**What It Means:**
The pod is waiting to be scheduled to a node.

**Common Causes:**
1. Insufficient cluster resources (CPU/memory)
2. Node selector doesn't match any node
3. Taints and tolerations mismatch
4. PersistentVolumeClaim not bound
5. ResourceQuota exceeded

**Diagnosis:**

```bash
kubectl describe pod my-pod
```

Look for events like:
- `FailedScheduling`
- `Insufficient cpu`
- `0/3 nodes are available`

**Solutions:**

### Solution 1: Check cluster resources
```bash
# Check node resources
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check what's using resources
kubectl top nodes
kubectl top pods --all-namespaces
```

### Solution 2: Reduce resource requests
```yaml
resources:
  requests:
    memory: "128Mi"  # Reduce from 256Mi
    cpu: "100m"      # Reduce from 250m
```

### Solution 3: Fix node selector
```bash
# Check node labels
kubectl get nodes --show-labels

# Ensure selector matches
nodeSelector:
  kubernetes.io/os: linux  # Must match node label
```

### Solution 4: Add tolerations for taints
```bash
# Check node taints
kubectl describe node my-node | grep Taints

# Add toleration
tolerations:
- key: "dedicated"
  operator: "Equal"
  value: "database"
  effect: "NoSchedule"
```

### Solution 5: Check PVC status
```bash
# Check PVC
kubectl get pvc

# If Pending, check storage class
kubectl get storageclass
```

---

## Error: CreateContainerConfigError

**Error Message:**
```
NAME        READY   STATUS                       RESTARTS   AGE
my-pod      0/1     CreateContainerConfigError   0          1m
```

**What It Means:**
Kubernetes cannot create the container due to configuration issues.

**Common Causes:**
1. ConfigMap or Secret doesn't exist
2. ConfigMap/Secret key doesn't exist
3. ServiceAccount doesn't exist
4. Invalid container configuration

**Diagnosis:**

```bash
kubectl describe pod my-pod
```

Look for:
- `configmap "xxx" not found`
- `secret "xxx" not found`
- `key "xxx" not found in secret`

**Solutions:**

### Solution 1: Create missing ConfigMap
```bash
# Check if ConfigMap exists
kubectl get configmap my-config

# Create it if missing
kubectl create configmap my-config \
  --from-literal=DATABASE_HOST=localhost \
  --from-literal=DATABASE_PORT=5432
```

### Solution 2: Create missing Secret
```bash
# Check if Secret exists
kubectl get secret my-secret

# Create it
kubectl create secret generic my-secret \
  --from-literal=password=mysecretpassword
```

### Solution 3: Verify key names
```bash
# List keys in ConfigMap
kubectl get configmap my-config -o yaml

# List keys in Secret
kubectl get secret my-secret -o yaml

# Make sure your pod references correct keys
envFrom:
- configMapRef:
    name: my-config  # Must exist
- secretRef:
    name: my-secret  # Must exist
```

---

## Error: ErrImageNeverPull

**Error Message:**
```
NAME        READY   STATUS              RESTARTS   AGE
my-pod      0/1     ErrImageNeverPull   0          30s
```

**What It Means:**
The imagePullPolicy is set to "Never" but the image doesn't exist locally on the node.

**Solution:**
```yaml
# Change pull policy
containers:
- name: app
  image: my-app:latest
  imagePullPolicy: IfNotPresent  # or Always
```

---

## Error: OOMKilled

**Event:**
```
Reason: OOMKilled
Exit Code: 137
```

**What It Means:**
The container used more memory than its limit and was killed by Kubernetes.

**Solutions:**

### Solution 1: Increase memory limit
```yaml
resources:
  limits:
    memory: "1Gi"  # Increase from 512Mi
```

### Solution 2: Fix memory leak in application
- Profile the application
- Check for memory leaks
- Optimize memory usage

### Solution 3: Use horizontal scaling
Instead of giving one pod more memory, use more pods:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Quick Reference: Common kubectl Debug Commands

```bash
# Get pod status
kubectl get pods

# Get detailed pod info
kubectl describe pod <pod-name>

# Get pod logs
kubectl logs <pod-name>
kubectl logs <pod-name> --previous
kubectl logs <pod-name> -c <container-name>

# Execute command in pod
kubectl exec -it <pod-name> -- /bin/sh

# Get events
kubectl get events --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods
kubectl top nodes

# Debug with ephemeral container
kubectl debug <pod-name> -it --image=busybox
```
