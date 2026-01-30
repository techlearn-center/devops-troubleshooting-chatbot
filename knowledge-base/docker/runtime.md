# Docker Runtime Errors

## Container Exits Immediately

### Exit Code 0
**Meaning:** Container completed successfully.
**Problem:** Usually means the main process finished (no long-running process).
**Fix:** Ensure the CMD/ENTRYPOINT runs a foreground process.
```dockerfile
# Wrong: process runs in background and container exits
CMD ["nginx"]

# Right: run in foreground
CMD ["nginx", "-g", "daemon off;"]
```

### Exit Code 1
**Meaning:** Application error.
**Fix:** Check application logs.
```bash
docker logs <container-id>
```

### Exit Code 137 (OOMKilled)
**Meaning:** Container was killed due to out of memory.
**Fix:** Increase memory limit.
```bash
docker run --memory=512m myapp
```

### Exit Code 139 (Segfault)
**Meaning:** Segmentation fault in the application.
**Fix:** Debug the application code. Check for null pointer dereferences.

## Port Binding Issues

### Error: `Bind for 0.0.0.0:80 failed: port is already allocated`
```bash
# Find what's using the port
docker ps -a | grep 80
# Or on the host:
lsof -i :80

# Stop the conflicting container
docker stop <container-id>

# Or use a different port
docker run -p 8080:80 myapp
```

## Volume Mount Issues

### Permission Denied on Mounted Volume
```bash
# Fix: Match container user to host directory permissions
docker run -u $(id -u):$(id -g) -v /host/dir:/container/dir myapp

# Or set permissions in Dockerfile
RUN chown -R appuser:appuser /app
USER appuser
```

### Data Not Persisting
**Problem:** Not using volumes or bind mounts.
```bash
# Named volume (managed by Docker)
docker run -v mydata:/app/data myapp

# Bind mount (host directory)
docker run -v /host/path:/container/path myapp
```

## Networking Issues

### Container Can't Reach Internet
```bash
# Check DNS
docker exec <container> nslookup google.com

# Check network settings
docker inspect <container> | grep -A 10 "Networks"

# Try with host networking
docker run --network host myapp
```

### Containers Can't Talk to Each Other
```bash
# Create a shared network
docker network create mynetwork

# Run containers on the same network
docker run --network mynetwork --name app1 myapp1
docker run --network mynetwork --name app2 myapp2

# Now app2 can reach app1 by name: http://app1:8080
```
