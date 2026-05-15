#!/bin/bash

# Deploy LangGraph agents to remote Aegra server
# Usage: ./bin/deploy_agents.sh <agent_dir_name> [--register]

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <agent_dir_name> [--register]"
    echo "Example: $0 deep_research"
    echo "Example (with registration): $0 deep_research --register"
    exit 1
fi

AGENT_DIR=$1
REGISTER=false

if [ "$2" == "--register" ]; then
    REGISTER=true
fi

# Convert deep_research -> deep-research-agent
GRAPH_ID=$(echo $AGENT_DIR | sed 's/_/-/g')-agent

# Load environment variables
if [ -f .env ]; then
    # Load only variables that don't start with a hash
    export $(grep -v '^#' .env | xargs)
fi

# Fallback for AEGRA_HOST if not in .env
REMOTE_HOST=${AEGRA_HOST:-192.168.0.100}
REMOTE_PATH="/mnt/truenas/nfs/aegra"
echo "🚀 Deploying agent: $AGENT_DIR to $REMOTE_HOST..."

# 1. Upload source code
echo "📦 Uploading source files..."
# Sync the agents directory to the remote path
rsync -avz --exclude '.git*' src/agents/ $REMOTE_HOST:$REMOTE_PATH/src/agents/

# 2. Update aegra.json on remote
echo "⚙️ Updating aegra.json..."
# We use a python one-liner via ssh to safely merge JSON config and preserve existing agents
ssh $REMOTE_HOST "python3 -c \"
import json, os

config_path = '$REMOTE_PATH/aegra.json'
new_agent = {
    'graph_id': '$GRAPH_ID',
    'entry_point': 'src.agents.$AGENT_DIR.graph'
}

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {'agents': []}
else:
    data = {'agents': []}

if 'agents' not in data:
    data['agents'] = []

# Update if exists, otherwise add
updated = False
for agent in data['agents']:
    if agent.get('graph_id') == new_agent['graph_id']:
        agent.update(new_agent)
        updated = True
        break

if not updated:
    data['agents'].append(new_agent)

with open(config_path, 'w') as f:
    json.dump(data, f, indent=2)
\""

# 4. Register the agent with Aegra Control Plane
if [ "$REGISTER" = true ]; then
    echo "📝 Registering agent with Aegra Control Plane..."
    # We assume the registration script is in src/agents/<agent_dir>/register.py
    if [ -f "src/agents/$AGENT_DIR/register.py" ]; then
        python3 src/agents/$AGENT_DIR/register.py
    else
        echo "⚠️ Registration script not found at src/agents/$AGENT_DIR/register.py"
    fi
else
    echo "⏭️ Skipping registration (use --register to register agent)"
fi

echo "✅ Deployment of $AGENT_DIR completed successfully!"
