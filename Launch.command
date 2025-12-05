#!/bin/bash

# Ultra Fast Image Generation - Mac Launcher
# Double-click this file to start the app!

cd "$(dirname "$0")"

echo "============================================"
echo "  Ultra Fast Image Generation for Mac"
echo "============================================"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is required but not installed."
    echo ""

    # Check if curl is available for installation
    if command -v curl &> /dev/null; then
        echo "Would you like to install uv? (y/n)"
        read -p "> " install_uv
        if [ "$install_uv" = "y" ] || [ "$install_uv" = "Y" ]; then
            echo ""
            echo "Installing uv..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.cargo/bin:$PATH"
            echo ""
            echo "uv installed successfully!"
        else
            echo "Please install uv manually and try again."
            echo "Installation instructions: https://docs.astral.sh/uv/getting-started/installation/"
            read -p "Press Enter to exit..."
            exit 1
        fi
    else
        echo "Please install uv manually and try again."
        echo "Installation instructions: https://docs.astral.sh/uv/getting-started/installation/"
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

echo "Using: $(uv --version)"

# Check if proxy settings should be applied
if [ -f "proxy.env" ]; then
    echo ""
    echo "Loading proxy settings from proxy.env..."
    source proxy.env
    echo "Proxy settings loaded:"
    [ ! -z "$https_proxy" ] && echo "  https_proxy=$https_proxy"
    [ ! -z "$http_proxy" ] && echo "  http_proxy=$http_proxy"
    [ ! -z "$all_proxy" ] && echo "  all_proxy=$all_proxy"
    echo ""
fi

# Install dependencies if needed
if [ ! -f ".venv/pyvenv.cfg" ]; then
    echo ""
    echo "First time setup - installing dependencies (this may take a few minutes)..."
    uv sync
    echo ""
    echo "Installation complete!"
fi

echo ""
echo "Starting Gradio UI..."
echo "Opening browser to http://localhost:7860"
echo ""
echo "(Press Ctrl+C to stop the server)"
echo ""

# Open browser after a short delay
(sleep 3 && open http://localhost:7860) &

# Run the app
uv run app.py
