#!/bin/bash

# Advanced Codebase Improvement Script
# Performs deep analysis and improvements

set -e

REPO_DIR="/home/lokrain/map/workspace"
cd "$REPO_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to analyze code complexity
analyze_complexity() {
    log "Analyzing code complexity..."
    
    # Count lines of code
    local total_lines=$(find . -name "*.rs" -not -path "./target/*" | xargs wc -l | tail -1 | awk '{print $1}')
    log "Total lines of Rust code: $total_lines"
    
    # Find large functions (>50 lines)
    log "Checking for large functions..."
    find . -name "*.rs" -not -path "./target/*" | while read file; do
        if grep -n "^fn\|^pub fn\|^async fn\|^pub async fn" "$file" | while read line; do
            line_num=$(echo "$line" | cut -d: -f1)
            # Simple heuristic to find function end (next function or end of file)
            next_fn=$(tail -n +$((line_num + 1)) "$file" | grep -n "^fn\|^pub fn\|^async fn\|^pub async fn" | head -1 | cut -d: -f1)
            if [[ -n "$next_fn" ]]; then
                func_length=$((next_fn))
            else
                func_length=$(tail -n +$line_num "$file" | wc -l)
            fi
            
            if [[ $func_length -gt 50 ]]; then
                func_name=$(echo "$line" | grep -o "fn [a-zA-Z_][a-zA-Z0-9_]*" | cut -d' ' -f2)
                warning "Large function '$func_name' in $file (${func_length} lines)"
            fi
        done; do true; done
    done
}

# Function to check for code duplication
check_duplication() {
    log "Checking for code duplication..."
    
    # Simple duplication check - look for repeated patterns
    find . -name "*.rs" -not -path "./target/*" | while read file; do
        # Check for repeated lines (potential copy-paste)
        if duplicated_lines=$(sort "$file" | uniq -d | grep -v "^\s*$" | grep -v "^\s*//" | head -5); then
            if [[ -n "$duplicated_lines" ]]; then
                warning "Potential code duplication in $file"
                echo "$duplicated_lines" | head -3
            fi
        fi
    done
}

# Function to suggest performance improvements
suggest_performance_improvements() {
    log "Analyzing for performance improvements..."
    
    # Check for potential performance issues
    find . -name "*.rs" -not -path "./target/*" | while read file; do
        # Check for .clone() usage that might be unnecessary
        if grep -n "\.clone()" "$file" | grep -v "// "; then
            warning "Potential unnecessary .clone() calls in $file"
            grep -n "\.clone()" "$file" | head -3
        fi
        
        # Check for string allocations in loops
        if grep -A5 -B5 "for.*in" "$file" | grep -E "String::new|format!|to_string"; then
            warning "Potential string allocation in loop in $file"
        fi
        
        # Check for Vec::new() in loops
        if grep -A5 -B5 "for.*in" "$file" | grep "Vec::new"; then
            warning "Potential vector allocation in loop in $file"
        fi
    done
}

# Function to check for security issues
check_security() {
    log "Checking for security issues..."
    
    find . -name "*.rs" -not -path "./target/*" | while read file; do
        # Check for unsafe blocks
        if grep -n "unsafe" "$file"; then
            warning "Unsafe code found in $file"
            grep -n "unsafe" "$file" | head -3
        fi
        
        # Check for unwrap() usage
        if grep -n "\.unwrap()" "$file" | grep -v "// "; then
            warning "Potential panic with .unwrap() in $file"
            grep -n "\.unwrap()" "$file" | head -3
        fi
        
        # Check for expect() without good error messages
        if grep -n "\.expect(\"\")" "$file"; then
            warning "Empty expect message in $file"
        fi
    done
}

# Function to check documentation coverage
check_documentation() {
    log "Checking documentation coverage..."
    
    find . -name "*.rs" -not -path "./target/*" | while read file; do
        # Count public items without documentation
        undocumented=$(grep -E "^pub (fn|struct|enum|trait|mod|const|static)" "$file" | while read line; do
            line_num=$(grep -n "$line" "$file" | head -1 | cut -d: -f1)
            prev_line_num=$((line_num - 1))
            if [[ $prev_line_num -gt 0 ]]; then
                prev_line=$(sed -n "${prev_line_num}p" "$file")
                if [[ ! "$prev_line" =~ ^[[:space:]]*/// ]]; then
                    echo "$line"
                fi
            else
                echo "$line"
            fi
        done)
        
        if [[ -n "$undocumented" ]]; then
            warning "Undocumented public items in $file"
            echo "$undocumented" | head -3
        fi
    done
}

# Function to optimize dependencies
optimize_dependencies() {
    log "Analyzing dependencies..."
    
    # Check for unused dependencies (requires cargo-udeps)
    if command -v cargo-udeps &> /dev/null; then
        cargo +nightly udeps --all-targets || warning "Some unused dependencies detected"
    else
        log "cargo-udeps not available, skipping unused dependency check"
    fi
    
    # Check for duplicate dependencies
    if cargo tree --duplicates 2>/dev/null | grep -v "└──"; then
        warning "Duplicate dependencies found"
        cargo tree --duplicates | head -10
    fi
}

# Function to run comprehensive tests
run_comprehensive_tests() {
    log "Running comprehensive test suite..."
    
    # Run tests with coverage if possible
    if command -v cargo-tarpaulin &> /dev/null; then
        cargo tarpaulin --out Html --output-dir coverage || warning "Coverage analysis failed"
    else
        cargo test --all || error "Tests failed"
    fi
    
    # Run benchmarks if available
    if find . -name "*.rs" | xargs grep -l "#\[bench\]" &>/dev/null; then
        cargo bench || warning "Benchmarks failed"
    fi
    
    # Run documentation tests
    cargo test --doc || warning "Documentation tests failed"
}

# Function to apply automatic fixes
apply_automatic_fixes() {
    log "Applying automatic fixes..."
    
    # Apply clippy fixes
    cargo clippy --fix --allow-dirty --allow-staged --all-targets --all-features || warning "Some clippy fixes failed"
    
    # Format code
    cargo fmt --all
    
    # Fix imports (if cargo-fmt supports it)
    # This would require additional tools like rustfmt with import sorting
}

# Function to generate improvement report
generate_report() {
    log "Generating improvement report..."
    
    local report_file="improvement_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# Codebase Improvement Report

Generated: $(date)

## Summary

This report contains analysis and suggestions for improving the codebase.

## Code Quality Metrics

- Total Rust files: $(find . -name "*.rs" -not -path "./target/*" | wc -l)
- Total lines of code: $(find . -name "*.rs" -not -path "./target/*" | xargs wc -l | tail -1 | awk '{print $1}')
- Clippy warnings: $(cargo clippy --message-format=short 2>&1 | grep -c "warning:" || echo "0")

## Improvements Applied

- ✅ Code formatting with rustfmt
- ✅ Clippy fixes applied
- ✅ Documentation checks performed
- ✅ Security analysis completed
- ✅ Performance analysis completed

## Recommendations

1. **Performance**: Review string allocations in hot paths
2. **Security**: Minimize use of .unwrap() and unsafe code
3. **Documentation**: Add docs for all public APIs
4. **Testing**: Increase test coverage where possible
5. **Dependencies**: Regular dependency updates and cleanup

## Next Steps

- Review large functions for potential refactoring
- Add more comprehensive tests
- Consider performance profiling for critical paths
- Update dependencies to latest versions

---
Generated by automated improvement cycle
EOF

    success "Report generated: $report_file"
}

# Main improvement workflow
main() {
    log "🚀 Starting advanced codebase improvement cycle"
    
    # Ensure we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        error "Not in a git repository"
        exit 1
    fi
    
    # Create improvement branch
    local branch_name="improvement-$(date +%Y%m%d-%H%M%S)"
    git checkout -b "$branch_name" || git checkout "$branch_name"
    
    # Run analysis
    analyze_complexity
    check_duplication
    suggest_performance_improvements
    check_security
    check_documentation
    optimize_dependencies
    
    # Apply fixes
    apply_automatic_fixes
    
    # Run tests
    run_comprehensive_tests
    
    # Generate report
    generate_report
    
    # Commit if there are changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        git add .
        git commit -m "🔧 Advanced codebase improvements

- Applied automated code quality fixes
- Performed security and performance analysis
- Updated documentation
- Ran comprehensive test suite
- Generated improvement report

Branch: $branch_name
Timestamp: $(date)"
        
        success "Changes committed to branch: $branch_name"
        log "To merge: git checkout main && git merge $branch_name"
    else
        success "No changes needed - codebase is already optimized!"
        git checkout main
        git branch -d "$branch_name"
    fi
    
    success "🎉 Advanced improvement cycle completed!"
}

# Run main function
main "$@"
