variable "deployment_tier" {
  description = "Deployment tier: tier0 (MVP) or tier1 (Production)"
  type        = string
  default     = "tier0"
  validation {
    condition     = contains(["tier0", "tier1"], var.deployment_tier)
    error_message = "Must be tier0 or tier1."
  }
}
