# QUANTEX Terraform — Hetzner Cloud Infrastructure
terraform {
  required_providers {
    hcloud = {
      source = "hetznercloud/hcloud"
      version = "~> 1.45"
    }
  }
}

variable "hcloud_token" {
  description = "Hetzner Cloud API Token"
  sensitive   = true
}

variable "ssh_key_name" {
  description = "Hetzner SSH key name"
  default     = "quantex-deploy"
}

provider "hcloud" {
  token = var.hcloud_token
}

# ── Tier 0: MVP ($4-5/month) ──────────────────────────────
resource "hcloud_server" "quantex_tier0" {
  count       = var.deployment_tier == "tier0" ? 1 : 0
  name        = "quantex-trading"
  server_type = "cx21"    # 2 vCPU, 4GB RAM
  image       = "ubuntu-24.04"
  location    = "hel1"
  ssh_keys    = [var.ssh_key_name]

  labels = {
    app     = "quantex"
    tier    = "mvp"
    cost    = "budget"
  }

  provisioner "remote-exec" {
    inline = [
      "apt-get update",
      "apt-get install -y docker.io docker-compose-v2",
      "systemctl enable docker",
      "systemctl start docker",
    ]
  }
}

# ── Tier 1: Production ($20-40/month) ──────────────────────
resource "hcloud_server" "quantex_tier1" {
  count       = var.deployment_tier == "tier1" ? 1 : 0
  name        = "quantex-production"
  server_type = "cx41"    # 4 vCPU, 16GB RAM
  image       = "ubuntu-24.04"
  location    = "hel1"
  ssh_keys    = [var.ssh_key_name]

  labels = {
    app     = "quantex"
    tier    = "production"
    cost    = "standard"
  }

  provisioner "remote-exec" {
    inline = [
      "apt-get update",
      "apt-get install -y docker.io docker-compose-v2 prometheus grafana",
      "systemctl enable docker prometheus grafana-server",
      "systemctl start docker prometheus grafana-server",
    ]
  }
}

# ── Firewall ───────────────────────────────────────────────
resource "hcloud_firewall" "quantex_fw" {
  name = "quantex-firewall"

  rule {
    direction = "in"
    protocol  = "tcp"
    port      = "22"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction = "in"
    protocol  = "tcp"
    port      = "80"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction = "in"
    protocol  = "tcp"
    port      = "443"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction = "in"
    protocol  = "tcp"
    port      = "3000"
    source_ips = ["0.0.0.0/0", "::/0"]
    description = "Frontend"
  }

  rule {
    direction = "in"
    protocol  = "tcp"
    port      = "8001"
    source_ips = ["0.0.0.0/0", "::/0"]
    description = "Orchestrator API"
  }

  rule {
    direction = "in"
    protocol  = "tcp"
    port      = "9090"
    source_ips = ["0.0.0.0/0", "::/0"]
    description = "Prometheus"
  }

  rule {
    direction = "in"
    protocol  = "tcp"
    port      = "3001"
    source_ips = ["0.0.0.0/0", "::/0"]
    description = "Grafana"
  }

  # Internal services (restricted)
  rule {
    direction = "in"
    protocol  = "tcp"
    port      = "4222"
    source_ips = ["10.0.0.0/8", "172.16.0.0/12"]
    description = "NATS internal"
  }
}

resource "hcloud_firewall_attachment" "quantex_fw_attach" {
  firewall_id = hcloud_firewall.quantex_fw.id
  server_ids  = concat(
    hcloud_server.quantex_tier0[*].id,
    hcloud_server.quantex_tier1[*].id,
  )
}

# ── Outputs ────────────────────────────────────────────────
output "server_ip" {
  value = var.deployment_tier == "tier0" ?
    hcloud_server.quantex_tier0[0].ipv4_address :
    hcloud_server.quantex_tier1[0].ipv4_address
  description = "Server IPv4 address"
}
