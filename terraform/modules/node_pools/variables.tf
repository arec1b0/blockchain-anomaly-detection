# Node Pools Module Variables

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
}

variable "node_pool_name" {
  description = "Name of the node pool"
  type        = string
}

variable "machine_type" {
  description = "Machine type for nodes"
  type        = string
}

variable "disk_size_gb" {
  description = "Disk size in GB"
  type        = number
}

variable "disk_type" {
  description = "Disk type (pd-standard, pd-ssd, pd-balanced)"
  type        = string
  default     = "pd-ssd"
}

variable "min_node_count" {
  description = "Minimum number of nodes"
  type        = number
}

variable "max_node_count" {
  description = "Maximum number of nodes"
  type        = number
}

variable "node_labels" {
  description = "Labels to apply to nodes"
  type        = map(string)
  default     = {}
}

variable "node_taints" {
  description = "Taints to apply to nodes"
  type = list(object({
    key    = string
    value  = string
    effect = string
  }))
  default = []
}

variable "preemptible" {
  description = "Use preemptible nodes"
  type        = bool
  default     = false
}
