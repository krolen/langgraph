#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import urllib.request
import urllib.error

# Basic env loading if .env exists
def load_env_file(file_path=".env"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip())
                    except ValueError:
                        continue

load_env_file()

DEFAULT_AEGRA_PATH = "/mnt/truenas/nfs/aegra"
DEFAULT_AEGRA_URL = os.environ.get("AEGRA_URL", "http://localhost:8000")

def get_manifest(agent_dir):
    """Load and validate the manifest.json for the agent."""
    manifest_path = Path("src/agents") / agent_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            return manifest
    except Exception as e:
        print(f"⚠️ Warning: Could not read manifest at {manifest_path}: {e}")
        return None

def sync_code(agent_dir, target_path):
    """Sync agent code to the target path."""
    src_dir = Path("src/agents") / agent_dir
    if not src_dir.exists():
        print(f"❌ Error: Agent directory {src_dir} does not exist.")
        sys.exit(1)

    dest_agents_root = Path(target_path) / "src/agents"
    print(f"📦 Syncing src/agents to {dest_agents_root}...")

    dest_agents_root.mkdir(parents=True, exist_ok=True)

    def ignore_files(dir, files):
        return [f for f in files if f == "__pycache__" or f.startswith(".")]

    for item in os.listdir("src/agents"):
        s = Path("src/agents") / item
        d = dest_agents_root / item
        
        # Sync specific agent, shared modules, and init files
        if item == agent_dir or item in ["__init__.py", "config.py", "llm_wrapper.py"]:
            if s.is_dir():
                if d.exists():
                    shutil.rmtree(d)
                shutil.copytree(s, d, ignore=ignore_files)
            else:
                shutil.copy2(s, d)
    
    print("✅ Code sync complete.")

def update_config(agent_dir, target_path):
    """Update aegra.json in the target path."""
    config_path = Path(target_path) / "aegra.json"
    graph_id = agent_dir.replace("_", "-") + "-agent"
    entry_point = f"src.agents.{agent_dir}.graph"

    print(f"⚙️ Updating {config_path} for {graph_id}...")

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"⚠️ Warning: Could not parse {config_path}. Creating new config.")
            data = {"agents": []}
    else:
        data = {"agents": []}

    if "agents" not in data:
        data["agents"] = []

    new_agent = {
        "graph_id": graph_id,
        "entry_point": entry_point
    }

    updated = False
    for agent in data["agents"]:
        if agent.get("graph_id") == graph_id:
            agent.update(new_agent)
            updated = True
            break
    
    if not updated:
        data["agents"].append(new_agent)

    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Config updated with {graph_id}.")
    return graph_id

def register_agent(graph_id, agent_dir, aegra_url, api_key):
    """Register agent with Aegra Control Plane using manifest.json."""
    if not api_key:
        print("⚠️ Warning: AEGRA_API_KEY not found. Skipping registration.")
        return

    manifest = get_manifest(agent_dir)
    
    if not manifest:
        print(f"❌ Error: manifest.json is required for registration but was not found in src/agents/{agent_dir}/")
        sys.exit(1)
    
    required_fields = ["name", "description"]
    missing = [f for f in required_fields if f not in manifest]
    if missing:
        print(f"❌ Error: manifest.json for {agent_dir} is missing required fields: {', '.join(missing)}")
        sys.exit(1)

    payload = {
        "graph_id": graph_id,
        "assistant_name": manifest["name"],
        "assistant_version": manifest.get("version", "0.1.0"),
        "assistant_description": manifest["description"]
    }

    print(f"📝 Registering {graph_id} ('{manifest['name']}') with Aegra Control Plane at {aegra_url}...")
    
    url = f"{aegra_url}/api/v1/assistants/register"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            print(f"✅ Registration successful: {res_data}")
    except urllib.error.HTTPError as e:
        print(f"❌ Registration failed: {e.code} {e.reason}")
        try:
            print(f"Response: {e.read().decode('utf-8')}")
        except:
            pass
    except Exception as e:
        print(f"❌ Registration failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Deploy Aegra agents to a mounted directory.")
    parser.add_argument("agent", help="Name of the agent directory in src/agents/")
    parser.add_argument("--path", help="Path to the mounted Aegra directory", default=os.environ.get("AEGRA_PATH", DEFAULT_AEGRA_PATH))
    parser.add_argument("--register", action="store_true", help="Register the agent with the Control Plane")
    parser.add_argument("--url", help="Aegra Control Plane URL", default=DEFAULT_AEGRA_URL)
    
    args = parser.parse_args()

    api_key = os.environ.get("AEGRA_API_KEY")

    sync_code(args.agent, args.path)
    graph_id = update_config(args.agent, args.path)

    if args.register:
        register_agent(graph_id, args.agent, args.url, api_key)

    print(f"🚀 Deployment of {args.agent} completed!")

if __name__ == "__main__":
    main()
