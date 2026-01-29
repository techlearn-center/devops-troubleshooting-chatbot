# Docker Build Errors and Solutions

## Error: COPY failed: file not found

**Error Message:**
```
COPY failed: file not found in build context or excluded by .dockerignore
```

**What It Means:**
Docker cannot find the file you're trying to copy into the image.

**Common Causes:**
1. File path is wrong in Dockerfile
2. File is outside the build context
3. File is in .dockerignore
4. Typo in filename
5. Case sensitivity issues (Linux is case-sensitive)

**Solutions:**

### Solution 1: Check file path relative to build context
```dockerfile
# Wrong - absolute path doesn't work
COPY /home/user/app/config.json /app/

# Correct - relative to build context
COPY config.json /app/
COPY ./src/config.json /app/
```

### Solution 2: Verify build context
```bash
# Build context is the path you provide
docker build -t myapp .        # . is the build context
docker build -t myapp ./app    # ./app is the build context

# Files MUST be inside the build context
# Cannot copy files from parent directories
```

### Solution 3: Check .dockerignore
```bash
# View .dockerignore
cat .dockerignore

# Remove the file from .dockerignore if needed
# Or use negation pattern:
# !config.json
```

### Solution 4: List files in build context
```bash
# See what Docker sees
ls -la
find . -name "config.json"
```

---

## Error: Package not found (apt-get)

**Error Message:**
```
E: Unable to locate package some-package
```

**What It Means:**
The package manager cannot find the requested package.

**Common Causes:**
1. Package index not updated
2. Package name is wrong
3. Package doesn't exist in that distribution
4. Using wrong base image

**Solutions:**

### Solution 1: Update package index first
```dockerfile
# Always update before installing
RUN apt-get update && apt-get install -y \
    package-name \
    && rm -rf /var/lib/apt/lists/*
```

### Solution 2: Use correct package name
```bash
# Search for package
apt-cache search keyword

# Common package name differences:
# Debian/Ubuntu: python3-pip
# Alpine: py3-pip
# CentOS: python3-pip
```

### Solution 3: Choose appropriate base image
```dockerfile
# For Python apps
FROM python:3.11-slim  # Debian-based, common packages

# For minimal size
FROM python:3.11-alpine  # Alpine-based, fewer packages

# For Node.js
FROM node:18-slim
```

---

## Error: Permission denied

**Error Message:**
```
permission denied while trying to connect to the Docker daemon socket
```

**What It Means:**
Your user doesn't have permission to use Docker.

**Solutions:**

### Solution 1: Add user to docker group
```bash
# Add current user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or run:
newgrp docker

# Verify
docker run hello-world
```

### Solution 2: Use sudo (not recommended for production)
```bash
sudo docker build -t myapp .
```

---

## Error: no space left on device

**Error Message:**
```
no space left on device
```

**What It Means:**
Docker has run out of disk space.

**Solutions:**

### Solution 1: Clean up unused Docker objects
```bash
# Remove all unused containers, networks, images
docker system prune -a

# Remove just dangling images
docker image prune

# Remove stopped containers
docker container prune

# Remove unused volumes (careful - may delete data!)
docker volume prune
```

### Solution 2: Check disk usage
```bash
# See Docker disk usage
docker system df

# Detailed view
docker system df -v
```

### Solution 3: Optimize Dockerfile to reduce image size
```dockerfile
# Use multi-stage builds
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production image
FROM node:18-slim
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
CMD ["node", "dist/index.js"]
```

---

## Error: failed to compute cache key

**Error Message:**
```
failed to compute cache key: "some-file" not found
```

**What It Means:**
Similar to COPY failed, Docker cannot find the specified file.

**Solutions:**

### Solution 1: Verify file exists
```bash
ls -la some-file
```

### Solution 2: Check Dockerfile path references
```dockerfile
# Ensure paths are relative to build context
COPY ./src/app.py /app/
COPY requirements.txt /app/
```

### Solution 3: Use .dockerignore carefully
```bash
# Check if file is being ignored
cat .dockerignore
```

---

## Error: Cannot connect to the Docker daemon

**Error Message:**
```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
```

**What It Means:**
The Docker daemon (background service) is not running.

**Solutions:**

### Solution 1: Start Docker daemon
```bash
# Linux (systemd)
sudo systemctl start docker
sudo systemctl enable docker  # Start on boot

# Linux (older init)
sudo service docker start

# macOS/Windows
# Start Docker Desktop application
```

### Solution 2: Check Docker status
```bash
# Check if running
sudo systemctl status docker

# View Docker daemon logs
sudo journalctl -u docker
```

---

## Error: Dockerfile parse error

**Error Message:**
```
failed to solve: dockerfile parse error
```

**What It Means:**
There's a syntax error in your Dockerfile.

**Common Causes:**
1. Invalid instruction
2. Missing backslash for line continuation
3. Wrong quotes
4. Invalid escape sequences

**Solutions:**

### Solution 1: Check instruction syntax
```dockerfile
# Correct syntax examples
FROM image:tag
WORKDIR /path
COPY source dest
RUN command
ENV KEY=value
EXPOSE port
CMD ["executable", "param"]
```

### Solution 2: Fix line continuations
```dockerfile
# Wrong
RUN apt-get update &&
    apt-get install -y package

# Correct
RUN apt-get update && \
    apt-get install -y package
```

### Solution 3: Use correct quotes
```dockerfile
# CMD uses JSON array format (double quotes)
CMD ["python", "app.py"]

# Not single quotes
CMD ['python', 'app.py']  # Wrong!
```

---

## Error: layer does not exist

**Error Message:**
```
layer does not exist
```

**What It Means:**
Docker is trying to use a cached layer that no longer exists.

**Solutions:**

### Solution 1: Build without cache
```bash
docker build --no-cache -t myapp .
```

### Solution 2: Pull fresh base image
```bash
docker pull python:3.11-slim
docker build -t myapp .
```

---

## Error: EXPOSE requires exactly one argument

**Error Message:**
```
EXPOSE requires exactly one argument
```

**Solutions:**

### Solution 1: Use correct EXPOSE syntax
```dockerfile
# Multiple ports - one per line
EXPOSE 80
EXPOSE 443

# Or on same line with spaces (not commas)
EXPOSE 80 443 8080
```

---

## Best Practices to Avoid Build Errors

### 1. Structure your Dockerfile properly
```dockerfile
# Start with base image
FROM python:3.11-slim

# Set working directory early
WORKDIR /app

# Copy dependency files first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]
```

### 2. Use .dockerignore
```plaintext
# .dockerignore
.git
.gitignore
__pycache__
*.pyc
.env
.venv
node_modules
*.md
Dockerfile
docker-compose.yml
.dockerignore
```

### 3. Minimize layers
```dockerfile
# Combine RUN commands
RUN apt-get update && \
    apt-get install -y \
        package1 \
        package2 \
    && rm -rf /var/lib/apt/lists/*
```

### 4. Use specific image tags
```dockerfile
# Bad - can break unexpectedly
FROM python:latest

# Good - predictable builds
FROM python:3.11.4-slim
```

---

## Quick Reference: Debug Commands

```bash
# Build with verbose output
docker build --progress=plain -t myapp .

# Build specific stage
docker build --target builder -t myapp:builder .

# Inspect image layers
docker history myapp

# View image contents
docker run --rm -it myapp /bin/sh

# Check build context size
du -sh .

# List files Docker will see
docker build -t test . && docker run --rm test ls -la /app
```
