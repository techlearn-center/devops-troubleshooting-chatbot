# Terraform State Management

## What is Terraform State?

Terraform state is a JSON file (`terraform.tfstate`) that maps your configuration to real-world cloud resources. It tracks what Terraform manages so it knows what to create, update, or delete.

## State File Location

By default, state is stored locally in `terraform.tfstate`. For teams, use remote state.

## Remote State Backends

### AWS S3 + DynamoDB (Locking)
```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
```

### Azure Blob Storage
```hcl
terraform {
  backend "azurerm" {
    resource_group_name  = "tfstate"
    storage_account_name = "tfstate12345"
    container_name       = "tfstate"
    key                  = "prod.terraform.tfstate"
  }
}
```

### Google Cloud Storage
```hcl
terraform {
  backend "gcs" {
    bucket = "my-terraform-state"
    prefix = "prod"
  }
}
```

## Common State Issues

### State Lock Error
**Error:** `Error acquiring the state lock`
**Cause:** Another process is running or a previous run crashed.
**Fix:**
```bash
terraform force-unlock <LOCK_ID>
```

### State Drift
**Problem:** Real infrastructure differs from state file.
**Fix:**
```bash
terraform refresh   # Update state to match real infrastructure
terraform plan      # See what changed
```

### Corrupted State
**Fix:**
```bash
# Restore from backup
cp terraform.tfstate.backup terraform.tfstate

# Or pull from remote
terraform state pull > terraform.tfstate
```

## State Best Practices

1. **Always use remote state** for team projects
2. **Enable state locking** to prevent concurrent modifications
3. **Enable encryption** for sensitive data in state
4. **Never edit state manually** - use `terraform state` commands
5. **Back up state** before destructive operations
