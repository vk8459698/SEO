#!/bin/bash

echo " Setting up LinkedIn Profile Search Engine..."
echo "================================================"

# Install required Python packages
echo " Installing Python packages..."
pip install opensearch-py sentence-transformers nltk spacy

# Download spaCy English model
echo " Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Install and start OpenSearch (if not already running)
echo " Setting up OpenSearch..."

# Check if OpenSearch is already running
if curl -s http://localhost:9200 > /dev/null 2>&1; then
    echo " OpenSearch is already running on localhost:9200"
else
    echo " Starting OpenSearch..."
    
    # For Ubuntu/Debian systems
    if command -v apt-get &> /dev/null; then
        echo " Installing OpenSearch on Ubuntu/Debian..."
        
        # Download and install OpenSearch
        wget https://artifacts.opensearch.org/releases/bundle/opensearch/2.11.0/opensearch-2.11.0-linux-x64.tar.gz
        tar -xzf opensearch-2.11.0-linux-x64.tar.gz
        cd opensearch-2.11.0
        
        # Configure OpenSearch for single-node cluster
        echo "discovery.type: single-node" >> config/opensearch.yml
        echo "plugins.security.disabled: true" >> config/opensearch.yml
        
        # Start OpenSearch in background
        ./bin/opensearch &
        
        echo " Waiting for OpenSearch to start..."
        sleep 30
        
    # For macOS with Homebrew
    elif command -v brew &> /dev/null; then
        echo " Installing OpenSearch on macOS..."
        brew install opensearch
        brew services start opensearch
        
        echo " Waiting for OpenSearch to start..."
        sleep 30
        
    # For systems with Docker
    elif command -v docker &> /dev/null; then
        echo " Starting OpenSearch with Docker..."
        docker run -d \
            --name opensearch-node \
            -p 9200:9200 -p 9600:9600 \
            -e "discovery.type=single-node" \
            -e "plugins.security.disabled=true" \
            -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=admin123" \
            opensearchproject/opensearch:2.11.0
        
        echo " Waiting for OpenSearch to start..."
        sleep 45
        
    else
        echo " Could not automatically install OpenSearch."
        echo "Please install OpenSearch manually or use Docker:"
        echo ""
        echo "Using Docker:"
        echo "docker run -d --name opensearch-node -p 9200:9200 -p 9600:9600 -e \"discovery.type=single-node\" -e \"plugins.security.disabled=true\" opensearchproject/opensearch:2.11.0"
        echo ""
        echo "Or download from: https://opensearch.org/downloads.html"
        exit 1
    fi
fi

# Verify OpenSearch is running
echo " Verifying OpenSearch connection..."
for i in {1..10}; do
    if curl -s http://localhost:9200 > /dev/null 2>&1; then
        echo " OpenSearch is running successfully!"
        curl -s http://localhost:9200 | python -m json.tool
        break
    else
        echo " Waiting for OpenSearch... (attempt $i/10)"
        sleep 5
    fi
    
    if [ $i -eq 10 ]; then
        echo " OpenSearch failed to start. Please check the logs and try manually."
        exit 1
    fi
done

echo ""
echo " Setup complete! You can now run your LinkedIn Profile Search Engine."
echo ""
echo "To test the setup:"
echo "python -c \"import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model loaded successfully!')\""
echo ""
echo "To run the application:"
echo "python app.py"
