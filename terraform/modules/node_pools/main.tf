# Node Pools Module

data "google_container_cluster" "cluster" {
  name     = var.cluster_name
  location = var.region
  project  = var.project_id
}

resource "google_container_node_pool" "pool" {
  name       = var.node_pool_name
  location   = var.region
  cluster    = data.google_container_cluster.cluster.name
  project    = var.project_id

  # Node count and autoscaling
  initial_node_count = var.min_node_count

  autoscaling {
    min_node_count = var.min_node_count
    max_node_count = var.max_node_count
  }

  # Node management
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # Node configuration
  node_config {
    machine_type = var.machine_type
    disk_size_gb = var.disk_size_gb
    disk_type    = var.disk_type

    # Use preemptible nodes for cost savings (not recommended for production)
    preemptible = var.preemptible

    # OAuth scopes
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only",
    ]

    # Labels
    labels = merge(
      var.node_labels,
      {
        node_pool = var.node_pool_name
      }
    )

    # Taints
    dynamic "taint" {
      for_each = var.node_taints
      content {
        key    = taint.value.key
        value  = taint.value.value
        effect = taint.value.effect
      }
    }

    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }

    # Shielded instance config
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    # Workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # Tags for network firewall rules
    tags = [var.node_pool_name, "gke-node"]
  }

  # Upgrade settings
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }

  lifecycle {
    ignore_changes = [
      initial_node_count,
    ]
  }
}

# Outputs
output "node_pool_name" {
  description = "Node pool name"
  value       = google_container_node_pool.pool.name
}

output "node_pool_id" {
  description = "Node pool ID"
  value       = google_container_node_pool.pool.id
}
