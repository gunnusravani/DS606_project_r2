#!/bin/bash
# Wrapper script to load HF_TOKEN and run token identification

# Try to load .env from common locations
ENV_FILE=""

# Check various possible locations
for location in \
    "/Users/sravani/.env" \
    "/Users/sravani/Documents/.env" \
    "/Users/sravani/Documents/VSCode_projects/.env" \
    "/Users/sravani/Documents/VSCode_projects/DS606_project_r2/.env" \
    "$HOME/.env" \
    ".env"
do
    if [ -f "$location" ]; then
        ENV_FILE="$location"
        break
    fi
done

if [ -n "$ENV_FILE" ]; then
    echo "📁 Found .env file at: $ENV_FILE"
    echo "📥 Loading HF_TOKEN from environment..."
    export $(cat "$ENV_FILE" | grep HF_TOKEN | xargs)
    echo "✅ HF_TOKEN loaded"
else
    echo "⚠️  .env file not found"
    echo "   Please provide the path to your .env file:"
    read -p "   Enter full path to .env: " ENV_FILE
    
    if [ -f "$ENV_FILE" ]; then
        export $(cat "$ENV_FILE" | grep HF_TOKEN | xargs)
        echo "✅ HF_TOKEN loaded from $ENV_FILE"
    else
        echo "❌ File not found: $ENV_FILE"
        exit 1
    fi
fi

# Verify HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN not set in environment variables"
    exit 1
fi

echo ""
echo "🚀 Starting token identification..."
echo "This will take 5-10 minutes on GPU"
echo ""

# Run the identification script
python scripts/identify_refusal_tokens.py
