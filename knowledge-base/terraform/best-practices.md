# Terraform Best Practices

## Project Structure

```
project/
├── main.tf           # Main resources
├── variables.tf      # Variable declarations
├── outputs.tf        # Output values
├── providers.tf      # Provider configuration
├── terraform.tfvars  # Variable values (DON'T commit secrets!)
└── modules/
    ├── networking/
    ├── compute/
    └── database/
```

## Code Organization

### Use Variables for Reusability
```hcl
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

resource "aws_instance" "web" {
  ami           = var.ami_id
  instance_type = var.environment == "prod" ? "t3.large" : "t3.small"

  tags = {
    Environment = var.environment
  }
}
```

### Use Modules for Reuse
```hcl
module "vpc" {
  source      = "./modules/networking"
  cidr_block  = "10.0.0.0/16"
  environment = var.environment
}
```

## Security Best Practices

1. **Never hardcode secrets** in `.tf` files
2. **Use environment variables** or secret managers for credentials
3. **Enable state encryption** for remote backends
4. **Use least privilege IAM roles** for Terraform execution
5. **Pin provider versions** to avoid breaking changes

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"  # Pin to major version
    }
  }
}
```

## Workflow Best Practices

1. **Always run `terraform plan`** before `apply`
2. **Review plans carefully** in CI/CD pipelines
3. **Use workspaces** for environment separation
4. **Tag all resources** for cost tracking
5. **Use `terraform fmt`** to keep code consistent
