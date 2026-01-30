# Docker Compose Troubleshooting

## Common Docker Compose Errors

### Error: `service "web" depends on undefined service "db"`
**Cause:** Service name typo in `depends_on`.
**Fix:** Check service names match exactly.
```yaml
services:
  web:
    depends_on:
      - db    # Must match a service name below
  db:         # This name must match
    image: postgres
```

### Error: `port is already allocated`
**Cause:** Another container or host process uses the port.
```bash
# Find what's using port 5432
docker compose ps
lsof -i :5432

# Fix: Change the host port mapping
ports:
  - "5433:5432"  # Map to different host port
```

### Error: `no configuration file provided: not found`
**Cause:** No `docker-compose.yml` or `compose.yaml` in current directory.
```bash
# Check you're in the right directory
ls docker-compose.yml

# Or specify the file
docker compose -f path/to/docker-compose.yml up
```

## Volume Issues

### Data Lost After `docker compose down`
**Problem:** Using anonymous volumes instead of named volumes.
```yaml
# Wrong: anonymous volume (destroyed on down)
volumes:
  - /var/lib/postgresql/data

# Right: named volume (persists across restarts)
volumes:
  - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:  # Declare named volume
```

### Warning: `docker compose down -v` Deletes Volumes!
The `-v` flag removes named volumes. Use carefully!
```bash
docker compose down     # Keeps volumes
docker compose down -v  # DELETES volumes and data!
```

## Networking Between Services

### Service Can't Connect to Database
```yaml
services:
  web:
    environment:
      # Use the SERVICE NAME as hostname, not localhost!
      DATABASE_URL: postgresql://user:pass@db:5432/mydb
  db:
    image: postgres
```

### Accessing Services from Host
```yaml
services:
  web:
    ports:
      - "3000:3000"  # host:container - accessible at localhost:3000
  api:
    expose:
      - "8080"       # Only accessible from other containers, not host!
```

## Build Issues

### Changes Not Reflected After Rebuild
```bash
# Force rebuild without cache
docker compose build --no-cache

# Rebuild and restart
docker compose up --build

# Remove old images first
docker compose down --rmi all
docker compose up --build
```

## Useful Commands

```bash
# Start in background
docker compose up -d

# View logs
docker compose logs -f

# View logs for specific service
docker compose logs -f web

# Restart a single service
docker compose restart web

# Scale a service
docker compose up -d --scale worker=3

# Check status
docker compose ps
```
