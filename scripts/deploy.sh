#!/bin/bash

# Codebase Improvement System Deployment Script
# This script deploys the automated improvement system

set -euo pipefail

REPO_DIR="/home/lokrain/map/workspace"
SCRIPT_DIR="$REPO_DIR/scripts"
LOG_DIR="$REPO_DIR/logs"

echo "🚀 Deploying Codebase Improvement System..."

# Create required directories
mkdir -p "$LOG_DIR"

# Function to deploy systemd service
deploy_systemd() {
    echo "📦 Deploying systemd service..."
    
    # Copy service files to systemd user directory
    SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
    mkdir -p "$SYSTEMD_USER_DIR"
    
    cp "$SCRIPT_DIR/improvement-cycle.service" "$SYSTEMD_USER_DIR/"
    cp "$SCRIPT_DIR/improvement-cycle.timer" "$SYSTEMD_USER_DIR/"
    
    # Reload systemd and enable timer
    systemctl --user daemon-reload
    systemctl --user enable improvement-cycle.timer
    systemctl --user start improvement-cycle.timer
    
    echo "✅ Systemd service deployed and started"
    echo "ℹ️  Check status with: systemctl --user status improvement-cycle.timer"
}

# Function to setup GitHub Actions
setup_github_actions() {
    echo "🐙 Setting up GitHub Actions..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo "⚠️  Not in a git repository. Initializing..."
        git init
        git add .
        git commit -m "Initial commit"
    fi
    
    # Check if GitHub Actions workflow exists
    if [[ -f "$REPO_DIR/.github/workflows/improvement-cycle.yml" ]]; then
        echo "✅ GitHub Actions workflow already exists"
    else
        echo "❌ GitHub Actions workflow not found"
        return 1
    fi
    
    echo "ℹ️  To enable GitHub Actions:"
    echo "   1. Push this repository to GitHub"
    echo "   2. The workflow will run automatically every hour"
    echo "   3. Set up branch protection rules if desired"
}

# Function to test the system
test_system() {
    echo "🧪 Testing improvement system..."
    
    # Test basic script functionality
    echo "Testing basic improvement script..."
    if "$SCRIPT_DIR/improvement-cycle.sh" test; then
        echo "✅ Basic script test passed"
    else
        echo "❌ Basic script test failed"
        return 1
    fi
    
    # Test Python manager if available
    if command -v python3 &> /dev/null; then
        echo "Testing Python improvement manager..."
        if python3 "$SCRIPT_DIR/improvement_manager.py" analyze > /dev/null; then
            echo "✅ Python manager test passed"
        else
            echo "❌ Python manager test failed"
            return 1
        fi
    else
        echo "⚠️  Python3 not available, skipping Python manager test"
    fi
}

# Function to show deployment options
show_deployment_options() {
    echo ""
    echo "🎯 Deployment Options:"
    echo ""
    echo "1. 🖥️  Local Systemd Service (recommended for development)"
    echo "   - Runs automatically every hour as a user service"
    echo "   - Uses systemd timer for reliable scheduling"
    echo "   - Logs to $LOG_DIR/"
    echo ""
    echo "2. ☁️  GitHub Actions (recommended for CI/CD)"
    echo "   - Runs in GitHub's cloud infrastructure"
    echo "   - Automatic on push and scheduled hourly"
    echo "   - Creates pull requests for improvements"
    echo ""
    echo "3. 🔧 Manual Execution"
    echo "   - Run scripts manually when needed"
    echo "   - Good for testing and one-off improvements"
    echo ""
    echo "4. 🐍 Python Manager"
    echo "   - Advanced analytics and reporting"
    echo "   - Continuous or single-run modes"
    echo "   - Detailed metrics tracking"
    echo ""
}

# Function to start manual improvement cycle
start_manual_cycle() {
    echo "🔄 Starting manual improvement cycle..."
    
    if "$SCRIPT_DIR/improvement-cycle.sh" single; then
        echo "✅ Manual improvement cycle completed successfully"
    else
        echo "❌ Manual improvement cycle failed"
        return 1
    fi
}

# Function to start continuous cycles
start_continuous() {
    echo "🔄 Starting continuous improvement cycles..."
    echo "⚠️  This will run indefinitely. Press Ctrl+C to stop."
    
    exec "$SCRIPT_DIR/improvement-cycle.sh" continuous
}

# Function to show status
show_status() {
    echo "📊 Improvement System Status:"
    echo ""
    
    # Check systemd status
    if systemctl --user is-active improvement-cycle.timer &> /dev/null; then
        echo "✅ Systemd timer: Active"
        echo "   Next run: $(systemctl --user list-timers improvement-cycle.timer --no-legend | awk '{print $1, $2, $3}')"
    else
        echo "❌ Systemd timer: Inactive"
    fi
    
    # Check recent logs
    if [[ -d "$LOG_DIR" ]] && [[ -n "$(find "$LOG_DIR" -name "*.log" -type f 2>/dev/null)" ]]; then
        echo "📝 Recent log files:"
        find "$LOG_DIR" -name "*.log" -type f -mtime -1 -exec ls -la {} \;
    else
        echo "📝 No recent log files found"
    fi
    
    # Check git status
    if git status --porcelain &> /dev/null; then
        if [[ -n "$(git status --porcelain)" ]]; then
            echo "🔄 Git status: Uncommitted changes"
        else
            echo "✅ Git status: Clean working directory"
        fi
    else
        echo "⚠️  Git status: Not a git repository"
    fi
}

# Main menu
main_menu() {
    echo ""
    echo "Select deployment option:"
    echo "1) Deploy systemd service (hourly automation)"
    echo "2) Setup GitHub Actions"
    echo "3) Test system"
    echo "4) Run single improvement cycle"
    echo "5) Start continuous cycles"
    echo "6) Show system status"
    echo "7) Show all options"
    echo "0) Exit"
    echo ""
    read -p "Enter your choice (0-7): " choice
    
    case $choice in
        1) deploy_systemd ;;
        2) setup_github_actions ;;
        3) test_system ;;
        4) start_manual_cycle ;;
        5) start_continuous ;;
        6) show_status ;;
        7) show_deployment_options ;;
        0) echo "👋 Goodbye!"; exit 0 ;;
        *) echo "❌ Invalid option"; main_menu ;;
    esac
}

# Parse command line arguments
case "${1:-menu}" in
    "systemd") deploy_systemd ;;
    "github") setup_github_actions ;;
    "test") test_system ;;
    "single") start_manual_cycle ;;
    "continuous") start_continuous ;;
    "status") show_status ;;
    "options") show_deployment_options ;;
    "menu") show_deployment_options; main_menu ;;
    *) 
        echo "Usage: $0 [systemd|github|test|single|continuous|status|options|menu]"
        exit 1
        ;;
esac

echo ""
echo "🎉 Deployment operation completed!"
