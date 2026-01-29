# GitHub Actions Errors and Solutions

## Error: Workflow Syntax Error

**Error Message:**
```
.github/workflows/main.yml (Line 15, Col 3): Unexpected value 'steps'
```

**What It Means:**
Your workflow YAML file has a syntax error.

**Common Causes:**
1. Incorrect indentation (YAML is sensitive to spaces)
2. Missing required fields
3. Using tabs instead of spaces
4. Invalid key names

**Solutions:**

### Solution 1: Fix indentation (use 2 spaces)
```yaml
# Correct structure
name: CI Pipeline

on:
  push:
    branches: [main]

jobs:
  build:                    # 0 spaces (job level)
    runs-on: ubuntu-latest  # 2 spaces
    steps:                  # 2 spaces
      - uses: actions/checkout@v4  # 4 spaces
      - name: Run tests            # 4 spaces
        run: npm test              # 6 spaces
```

### Solution 2: Validate YAML locally
```bash
# Install yamllint
pip install yamllint

# Check your workflow
yamllint .github/workflows/main.yml

# Or use online validators
```

### Solution 3: Use workflow validator
```bash
# GitHub CLI validation
gh workflow view main.yml

# Check Actions tab in GitHub for error details
```

---

## Error: Resource not accessible by integration

**Error Message:**
```
Resource not accessible by integration
```

**What It Means:**
The GitHub token doesn't have permission for the requested action.

**Common Causes:**
1. Missing permissions in workflow
2. Action requires write access
3. Fork workflow restrictions
4. Organization security settings

**Solutions:**

### Solution 1: Add explicit permissions
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      pull-requests: write
      issues: write
    steps:
      - uses: actions/checkout@v4
```

### Solution 2: For creating releases
```yaml
permissions:
  contents: write  # Required for creating releases
```

### Solution 3: For commenting on PRs
```yaml
permissions:
  pull-requests: write
```

---

## Error: Context access might be invalid

**Error Message:**
```
Context access might be invalid: secrets.MY_SECRET
```

**What It Means:**
The secret you're referencing might not exist or is misspelled.

**Solutions:**

### Solution 1: Verify secret exists
1. Go to repository Settings → Secrets and variables → Actions
2. Check if the secret name matches exactly (case-sensitive)

### Solution 2: Use correct syntax
```yaml
- name: Deploy
  env:
    API_KEY: ${{ secrets.API_KEY }}
  run: ./deploy.sh

# For secrets in run commands directly
- name: Login
  run: echo "${{ secrets.PASSWORD }}" | docker login -u user --password-stdin
```

### Solution 3: Set default value for optional secrets
```yaml
- name: Optional feature
  if: ${{ secrets.OPTIONAL_SECRET != '' }}
  run: ./optional-feature.sh
  env:
    SECRET: ${{ secrets.OPTIONAL_SECRET }}
```

---

## Error: Process completed with exit code 1

**Error Message:**
```
Error: Process completed with exit code 1
```

**What It Means:**
A command in your workflow failed.

**Diagnosis:**
Look at the step output above the error for the actual failure reason.

**Solutions:**

### Solution 1: Debug with more output
```yaml
- name: Debug info
  run: |
    echo "Current directory: $(pwd)"
    echo "Files:"
    ls -la
    echo "Environment:"
    env | sort
```

### Solution 2: Continue on error for debugging
```yaml
- name: Potentially failing step
  continue-on-error: true
  run: npm test

- name: Debug after failure
  if: failure()
  run: |
    cat test-output.log
    cat npm-debug.log 2>/dev/null || true
```

### Solution 3: Use fail-fast false for matrix jobs
```yaml
strategy:
  fail-fast: false
  matrix:
    node: [16, 18, 20]
```

---

## Error: Unable to resolve action

**Error Message:**
```
Unable to resolve action `owner/action@v1`
```

**What It Means:**
GitHub cannot find the action you're trying to use.

**Common Causes:**
1. Action name is misspelled
2. Version/tag doesn't exist
3. Action repository is private
4. Action was deleted

**Solutions:**

### Solution 1: Verify action name and version
```yaml
# Check the action's GitHub page for correct name
- uses: actions/checkout@v4      # Correct
- uses: action/checkout@v4       # Wrong (typo)
- uses: actions/checkout@v99     # Wrong (version doesn't exist)
```

### Solution 2: Use specific commit SHA (more secure)
```yaml
- uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11
```

### Solution 3: For private actions
```yaml
- uses: ./.github/actions/my-action  # Local action
```

---

## Error: Artifacts not found

**Error Message:**
```
Unable to download artifact(s). Artifact not found.
```

**What It Means:**
The artifact you're trying to download doesn't exist or expired.

**Solutions:**

### Solution 1: Ensure upload happens first
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: my-artifact
          path: build/
          retention-days: 5

  deploy:
    needs: build  # Important! Wait for build to complete
    runs-on: ubuntu-latest
    steps:
      - name: Download artifact
        uses: actions/download-artifact@v4
        with:
          name: my-artifact
```

### Solution 2: Check artifact name matches exactly
```yaml
# Upload
with:
  name: build-output  # Must match

# Download
with:
  name: build-output  # Must match exactly
```

### Solution 3: Use correct path
```yaml
- uses: actions/download-artifact@v4
  with:
    name: my-artifact
    path: downloaded-files  # Files go here

- run: ls -la downloaded-files/  # Verify download
```

---

## Error: Timeout

**Error Message:**
```
The job running on runner GitHub Actions X has exceeded the maximum execution time of 360 minutes.
```

**Solutions:**

### Solution 1: Set explicit timeout
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - name: Long running step
        timeout-minutes: 10
        run: ./long-script.sh
```

### Solution 2: Cancel workflow on timeout
```yaml
- name: Setup timeout
  uses: actions/github-script@v7
  with:
    script: |
      setTimeout(() => {
        core.setFailed('Custom timeout reached');
        process.exit(1);
      }, 600000);  // 10 minutes
```

---

## Error: Cache not found

**Error Message:**
```
Cache not found for input keys: npm-cache-xxx
```

**What It Means:**
No cache exists matching your cache key. This is normal on first run.

**Solutions:**

### Solution 1: Proper cache setup
```yaml
- uses: actions/cache@v4
  with:
    path: ~/.npm
    key: npm-${{ runner.os }}-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      npm-${{ runner.os }}-
```

### Solution 2: Use setup actions with built-in caching
```yaml
- uses: actions/setup-node@v4
  with:
    node-version: '18'
    cache: 'npm'  # Built-in caching
```

### Solution 3: For Python
```yaml
- uses: actions/setup-python@v5
  with:
    python-version: '3.11'
    cache: 'pip'  # Built-in pip caching
```

---

## Error: No hosted runner matching the labels

**Error Message:**
```
Waiting for a runner to pick up this job...
No hosted runner matching the requested labels was found
```

**Solutions:**

### Solution 1: Use correct runner label
```yaml
# Available GitHub-hosted runners
runs-on: ubuntu-latest
runs-on: ubuntu-22.04
runs-on: windows-latest
runs-on: macos-latest
```

### Solution 2: For self-hosted runners
```yaml
runs-on: self-hosted
# Or with labels
runs-on: [self-hosted, linux, x64]
```

---

## Common Workflow Patterns

### CI/CD for Node.js
```yaml
name: CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'

      - run: npm ci
      - run: npm test
      - run: npm run build

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy
        run: echo "Deploying..."
```

### CI/CD for Python
```yaml
name: Python CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - run: pip install -r requirements.txt
      - run: pytest
```

### Docker Build and Push
```yaml
name: Docker

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
```

---

## Quick Reference: Useful Expressions

```yaml
# Conditional execution
if: github.ref == 'refs/heads/main'
if: github.event_name == 'pull_request'
if: contains(github.event.head_commit.message, '[skip ci]') == false
if: success()
if: failure()
if: always()

# Matrix values
${{ matrix.node-version }}

# Environment context
${{ env.MY_VAR }}
${{ runner.os }}
${{ github.sha }}
${{ github.repository }}

# Secrets
${{ secrets.MY_SECRET }}

# Expressions
${{ hashFiles('**/package-lock.json') }}
${{ toJSON(github.event) }}
```

---

## Debugging Tips

### Enable debug logging
Add these secrets to your repository:
- `ACTIONS_RUNNER_DEBUG`: `true`
- `ACTIONS_STEP_DEBUG`: `true`

### Add debug output
```yaml
- name: Debug
  run: |
    echo "Event: ${{ github.event_name }}"
    echo "Ref: ${{ github.ref }}"
    echo "SHA: ${{ github.sha }}"
    cat $GITHUB_EVENT_PATH
```

### Use act for local testing
```bash
# Install act
brew install act

# Run workflow locally
act push
act pull_request
```
