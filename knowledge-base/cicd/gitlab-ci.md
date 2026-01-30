# GitLab CI Troubleshooting

## Common Errors

### Error: `This job is stuck because you don't have any active runners`
**Cause:** No GitLab Runner registered or runners are offline.
**Fix:**
```bash
# Check runner status
gitlab-runner list

# Register a new runner
gitlab-runner register \
  --url https://gitlab.com/ \
  --registration-token YOUR_TOKEN

# Start the runner
gitlab-runner run
```

### Error: `yaml invalid`
**Cause:** Syntax error in `.gitlab-ci.yml`.
**Fix:** Use the CI/CD lint tool at `CI/CD → Editor → Validate`

### Error: `Job failed: exit code 1`
**Cause:** A command in the script returned an error.
**Fix:** Check job logs, fix the failing command.

## .gitlab-ci.yml Basics

```yaml
stages:
  - build
  - test
  - deploy

variables:
  APP_NAME: "my-app"

build:
  stage: build
  image: node:18
  script:
    - npm install
    - npm run build
  artifacts:
    paths:
      - dist/

test:
  stage: test
  image: node:18
  script:
    - npm test
  coverage: '/Statements\s*:\s*(\d+\.?\d*)%/'

deploy:
  stage: deploy
  image: bitnami/kubectl
  script:
    - kubectl apply -f k8s/
  only:
    - main
  environment:
    name: production
```

## Caching Issues

### Slow Builds Due to No Cache
```yaml
build:
  cache:
    key: ${CI_COMMIT_REF_SLUG}
    paths:
      - node_modules/
  script:
    - npm ci
    - npm run build
```

### Stale Cache Causing Failures
```bash
# Clear cache in CI/CD → Pipelines → Clear Runner Caches
```

## Docker-in-Docker Issues

### Error: `Cannot connect to the Docker daemon`
```yaml
build-image:
  image: docker:24
  services:
    - docker:24-dind   # Docker-in-Docker service
  variables:
    DOCKER_TLS_CERTDIR: "/certs"
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

## Artifacts and Dependencies

### Passing Files Between Stages
```yaml
build:
  stage: build
  script:
    - npm run build
  artifacts:
    paths:
      - dist/
    expire_in: 1 hour

deploy:
  stage: deploy
  dependencies:
    - build            # Downloads artifacts from build job
  script:
    - ls dist/         # Files from build stage available here
    - deploy.sh
```

## Environment Variables and Secrets

### Using CI/CD Variables
Go to Settings → CI/CD → Variables to add secrets.

```yaml
deploy:
  script:
    - echo $KUBE_CONFIG | base64 -d > ~/.kube/config
    - kubectl apply -f k8s/
  variables:
    KUBE_NAMESPACE: "production"
```
