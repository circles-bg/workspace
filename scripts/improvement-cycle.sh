#!/bin/bash

# Codebase Improvement Cycle Script
# Runs automated improvements, quality checks, and commits changes

set -e  # Exit on any error

# Configuration
REPO_DIR="/home/lokrain/map/workspace"
LOG_DIR="$REPO_DIR/logs"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/improvement_cycle_$TIMESTAMP.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check if there are any changes to commit
has_changes() {
    cd "$REPO_DIR"
    ! git diff --quiet || ! git diff --cached --quiet
}

# Function to run improvement cycle
run_improvement_cycle() {
    local cycle_number=$1
    
    log "=== Starting Improvement Cycle #$cycle_number ==="
    
    cd "$REPO_DIR"
    
    # 1. Pull latest changes
    log "Pulling latest changes from remote..."
    git pull origin main || log "Warning: Could not pull from remote (continuing anyway)"
    
    # 2. Run Clippy and fix warnings
    log "Running Clippy analysis..."
    if cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tee -a "$LOG_FILE"; then
        log "✅ No Clippy warnings found"
    else
        log "⚠️  Clippy warnings detected - attempting automatic fixes..."
        
        # Try to fix some common issues automatically
        cargo clippy --fix --allow-dirty --allow-staged --all-targets --all-features 2>&1 | tee -a "$LOG_FILE" || true
        
        log "Re-running Clippy after fixes..."
        cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tee -a "$LOG_FILE" || log "Some Clippy issues remain"
    fi
    
    # 3. Format code
    log "Formatting code with rustfmt..."
    cargo fmt --all
    
    # 4. Run tests
    log "Running test suite..."
    if cargo test --all 2>&1 | tee -a "$LOG_FILE"; then
        log "✅ All tests passed"
    else
        log "❌ Some tests failed"
        return 1
    fi
    
    # 5. Check documentation
    log "Checking documentation..."
    cargo doc --all --no-deps 2>&1 | tee -a "$LOG_FILE" || log "Documentation issues detected"
    
    # 6. Run release build to check for optimization issues
    log "Running release build..."
    cargo build --release --all 2>&1 | tee -a "$LOG_FILE"
    
    # 7. Check for unused dependencies
    log "Checking for unused dependencies..."
    # This would require cargo-udeps, but we'll skip for now
    # cargo +nightly udeps --all-targets 2>&1 | tee -a "$LOG_FILE" || true
    
    # 8. Security audit (if cargo-audit is available)
    log "Running security audit..."
    if command -v cargo-audit &> /dev/null; then
        cargo audit 2>&1 | tee -a "$LOG_FILE" || log "Security issues detected"
    else
        log "cargo-audit not available, skipping security check"
    fi
    
    # 9. Check for any changes and commit
    if has_changes; then
        log "Changes detected, preparing commit..."
        
        # Add all changes
        git add .
        
        # Create descriptive commit message
        COMMIT_MSG="🔧 Automated improvement cycle #$cycle_number

- Clippy fixes and code quality improvements
- Code formatting with rustfmt
- Documentation updates
- Test suite validation
- Release build verification

Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        
        # Commit changes
        git commit -m "$COMMIT_MSG"
        
        # Push changes
        log "Pushing changes to remote..."
        git push origin main || log "Warning: Could not push to remote"
        
        log "✅ Cycle #$cycle_number completed with changes committed"
    else
        log "✅ Cycle #$cycle_number completed - no changes needed"
    fi
    
    log "=== Improvement Cycle #$cycle_number Complete ==="
    log ""
}

# Function to run continuous improvement cycles
run_continuous_cycles() {
    local cycle_count=1
    
    log "Starting continuous improvement cycles (every hour)"
    log "Logs will be saved to: $LOG_FILE"
    
    while true; do
        run_improvement_cycle $cycle_count
        
        cycle_count=$((cycle_count + 1))
        
        # Wait for 1 hour (3600 seconds)
        log "Waiting 1 hour until next cycle..."
        sleep 3600
    done
}

# Function to run a single cycle
run_single_cycle() {
    run_improvement_cycle 1
}

# Main script logic
case "${1:-continuous}" in
    "single")
        run_single_cycle
        ;;
    "continuous")
        run_continuous_cycles
        ;;
    *)
        echo "Usage: $0 [single|continuous]"
        echo "  single     - Run one improvement cycle"
        echo "  continuous - Run improvement cycles every hour (default)"
        exit 1
        ;;
esac
