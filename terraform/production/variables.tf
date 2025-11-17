# Variables for Production Kubernetes Cluster

variable "project_id" {
  description = "GCP Project ID"
  type        = string
  # Set via: export TF_VAR_project_id="your-project-id"
}

variable "region" {
  description = "GCP region for cluster"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "subnet_cidr" {
  description = "CIDR range for the subnet"
  type        = string
  default     = "10.0.0.0/16"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "master_authorized_networks" {
  description = "List of CIDR blocks that can access the Kubernetes master"
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = [
    {
      cidr_block   = "0.0.0.0/0"
      display_name = "All networks (CHANGE IN PRODUCTION)"
    }
  ]
}

variable "api_node_machine_type" {
  description = "Machine type for API node pool"
  type        = string
  default     = "n2-standard-4" # 4 vCPU, 16GB RAM
}

variable "consumer_node_machine_type" {
  description = "Machine type for consumer node pool"
  type        = string
  default     = "n2-standard-8" # 8 vCPU, 32GB RAM
}

variable "data_node_machine_type" {
  description = "Machine type for data node pool"
  type        = string
  default     = "n2-standard-4" # 4 vCPU, 16GB RAM
}

variable "enable_monitoring" {
  description = "Enable Google Cloud Monitoring"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable Google Cloud Logging"
  type        = bool
  default     = true
}
