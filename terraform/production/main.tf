# Terraform configuration for Production Kubernetes Cluster
# Supports: GKE (Google), EKS (AWS), AKS (Azure)
# Default: GKE for this example

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  # Backend configuration for state management
  # Uncomment and configure for production
  # backend "gcs" {
  #   bucket = "blockchain-anomaly-terraform-state"
  #   prefix = "production/terraform.tfstate"
  # }
}

# Configure the Google Cloud Provider
provider "google" {
  project = var.project_id
  region  = var.region
}

# Configure Kubernetes provider (will be set after cluster creation)
provider "kubernetes" {
  host                   = "https://${module.gke_cluster.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(module.gke_cluster.ca_certificate)
}

# Configure Helm provider
provider "helm" {
  kubernetes {
    host                   = "https://${module.gke_cluster.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(module.gke_cluster.ca_certificate)
  }
}

# Get current Google client config
data "google_client_config" "default" {}

# VPC Network
module "networking" {
  source = "../modules/networking"

  project_id   = var.project_id
  region       = var.region
  network_name = "${var.environment}-network"
  subnet_name  = "${var.environment}-subnet"
  subnet_cidr  = var.subnet_cidr
}

# GKE Cluster
module "gke_cluster" {
  source = "../modules/cluster"

  project_id          = var.project_id
  region              = var.region
  cluster_name        = "${var.environment}-gke-cluster"
  network             = module.networking.network_name
  subnetwork          = module.networking.subnet_name
  kubernetes_version  = var.kubernetes_version

  # Master authorized networks (restrict access to cluster API)
  master_authorized_networks = var.master_authorized_networks

  # Enable features
  enable_vertical_pod_autoscaling = true
  enable_horizontal_pod_autoscaling = true
  enable_network_policy           = true
  enable_binary_authorization     = true
  enable_shielded_nodes          = true
}

# Node Pools
module "api_node_pool" {
  source = "../modules/node_pools"

  project_id   = var.project_id
  region       = var.region
  cluster_name = module.gke_cluster.cluster_name

  node_pool_name = "api-pool"
  machine_type   = var.api_node_machine_type
  disk_size_gb   = 100
  disk_type      = "pd-ssd"

  min_node_count = 3
  max_node_count = 10

  node_labels = {
    workload_type = "api"
    environment   = var.environment
  }

  node_taints = []

  preemptible = false
}

module "consumer_node_pool" {
  source = "../modules/node_pools"

  project_id   = var.project_id
  region       = var.region
  cluster_name = module.gke_cluster.cluster_name

  node_pool_name = "consumer-pool"
  machine_type   = var.consumer_node_machine_type
  disk_size_gb   = 200
  disk_type      = "pd-ssd"

  min_node_count = 2
  max_node_count = 8

  node_labels = {
    workload_type = "consumer"
    environment   = var.environment
  }

  node_taints = []

  preemptible = false
}

module "data_node_pool" {
  source = "../modules/node_pools"

  project_id   = var.project_id
  region       = var.region
  cluster_name = module.gke_cluster.cluster_name

  node_pool_name = "data-pool"
  machine_type   = var.data_node_machine_type
  disk_size_gb   = 500
  disk_type      = "pd-ssd"

  min_node_count = 2
  max_node_count = 5

  node_labels = {
    workload_type = "data"
    environment   = var.environment
  }

  node_taints = []

  preemptible = false
}

# Create Kubernetes namespaces
resource "kubernetes_namespace" "blockchain_anomaly_prod" {
  metadata {
    name = "blockchain-anomaly-prod"
    labels = {
      environment = "production"
      managed_by  = "terraform"
    }
  }

  depends_on = [module.gke_cluster]
}

resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = "monitoring"
    labels = {
      environment = "production"
      managed_by  = "terraform"
    }
  }

  depends_on = [module.gke_cluster]
}

resource "kubernetes_namespace" "logging" {
  metadata {
    name = "logging"
    labels = {
      environment = "production"
      managed_by  = "terraform"
    }
  }

  depends_on = [module.gke_cluster]
}

# Output important values
output "cluster_name" {
  description = "GKE cluster name"
  value       = module.gke_cluster.cluster_name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = module.gke_cluster.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = module.gke_cluster.ca_certificate
  sensitive   = true
}

output "region" {
  description = "GCP region"
  value       = var.region
}

output "kubeconfig_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${module.gke_cluster.cluster_name} --region ${var.region} --project ${var.project_id}"
}
