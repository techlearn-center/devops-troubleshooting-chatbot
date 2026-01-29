# Terraform Common Errors and Solutions

## Error: Resource Already Exists

**Error Message:**
```
Error: A resource with the ID "xxx" already exists -
to be managed via Terraform this resource needs to be imported
```

**What It Means:**
Terraform is trying to create a resource that already exists in your cloud provider, but Terraform doesn't know about it (it's not in the state file).

**Common Causes:**
1. Resource was created manually via AWS/GCP/Azure console
2. State file was deleted or corrupted
3. Terraform import was incomplete
4. Another Terraform workspace created the resource
5. Resource name conflicts between modules

**Solutions:**

### Solution 1: Import the existing resource
```bash
# Find the resource ID in your cloud provider console
# Then import it
terraform import aws_instance.example i-1234567890abcdef0

# For S3 bucket
terraform import aws_s3_bucket.my_bucket my-bucket-name

# For security group
terraform import aws_security_group.web sg-0123456789abcdef0
```

### Solution 2: Remove from state (if duplicate)
```bash
# If the resource in state is a duplicate, remove it
terraform state rm aws_instance.example

# Then apply to create fresh
terraform apply
```

### Solution 3: Use data source instead
If you just need to reference the resource, not manage it:
```hcl
data "aws_instance" "existing" {
  instance_id = "i-1234567890abcdef0"
}

# Reference it
resource "aws_eip" "ip" {
  instance = data.aws_instance.existing.id
}
```

**Prevention:**
- Always use remote state with locking (S3 + DynamoDB)
- Never modify infrastructure outside Terraform
- Use workspaces carefully
- Document manually-created resources

---

## Error: State Lock

**Error Message:**
```
Error: Error acquiring the state lock
Lock Info:
  ID:        xxx
  Path:      terraform.tfstate
  Operation: OperationTypeApply
```

**What It Means:**
Another Terraform process has locked the state file to prevent concurrent modifications.

**Common Causes:**
1. Previous terraform apply/plan was interrupted (Ctrl+C)
2. Another team member is running Terraform
3. CI/CD pipeline is running
4. Crashed Terraform process left stale lock

**Solutions:**

### Solution 1: Wait and retry
```bash
# Simply wait for the other process to complete
# Then retry your command
terraform apply
```

### Solution 2: Force unlock (use with caution!)
```bash
# Get the lock ID from the error message
terraform force-unlock LOCK_ID

# Example
terraform force-unlock 12345678-1234-1234-1234-123456789012
```

**Warning:** Only force-unlock if you're certain no other process is running!

### Solution 3: Check what's holding the lock
```bash
# For S3 backend, check DynamoDB
aws dynamodb scan --table-name terraform-locks

# Delete stale lock
aws dynamodb delete-item \
  --table-name terraform-locks \
  --key '{"LockID": {"S": "your-bucket/path/terraform.tfstate"}}'
```

**Prevention:**
- Always let Terraform commands complete
- Use proper state locking with DynamoDB
- Set up lock timeouts in CI/CD

---

## Error: Provider Configuration Not Found

**Error Message:**
```
Error: Provider configuration not present
```

**What It Means:**
Terraform can't find a provider configuration for a resource that exists in state.

**Common Causes:**
1. Provider was removed from configuration
2. Provider alias mismatch
3. State contains resources from deleted modules

**Solutions:**

### Solution 1: Add missing provider
```hcl
# Add the provider back
provider "aws" {
  region = "us-east-1"
}

# If using aliases
provider "aws" {
  alias  = "west"
  region = "us-west-2"
}
```

### Solution 2: Remove orphaned resources
```bash
# List resources with missing provider
terraform state list

# Remove orphaned resources
terraform state rm module.deleted_module.aws_instance.example
```

---

## Error: Cycle Detected

**Error Message:**
```
Error: Cycle: aws_security_group.a, aws_security_group.b
```

**What It Means:**
Two or more resources depend on each other in a circular way.

**Common Causes:**
1. Security groups referencing each other
2. IAM roles and policies with circular dependencies
3. Incorrect use of depends_on

**Solutions:**

### Solution 1: Break the cycle with aws_security_group_rule
```hcl
# Instead of inline rules, use separate rule resources
resource "aws_security_group" "a" {
  name = "sg-a"
}

resource "aws_security_group" "b" {
  name = "sg-b"
}

resource "aws_security_group_rule" "a_to_b" {
  type                     = "ingress"
  from_port                = 443
  to_port                  = 443
  protocol                 = "tcp"
  security_group_id        = aws_security_group.b.id
  source_security_group_id = aws_security_group.a.id
}
```

### Solution 2: Use self-referencing rules
```hcl
resource "aws_security_group" "cluster" {
  name = "cluster-sg"

  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true  # Allows traffic from same SG
  }
}
```
