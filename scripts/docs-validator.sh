#!/bin/bash

# XORB Platform Documentation Validator
# Validates documentation quality, links, and formatting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCS_DIR="docs"
REPORT_DIR="reports/docs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="$REPORT_DIR/validation_report_$TIMESTAMP.md"

# Counters
TOTAL_FILES=0
VALID_FILES=0
ERROR_COUNT=0
WARNING_COUNT=0

# Create report directory
mkdir -p "$REPORT_DIR"

echo -e "${BLUE}🔍 XORB Documentation Validator${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# Initialize report
cat > "$REPORT_FILE" << EOF
# Documentation Validation Report

**Generated**: $(date)  
**Validator Version**: 1.0.0  
**Repository**: XORB Platform  

## Summary

EOF

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARNING_COUNT++))
    echo "- ⚠️ $1" >> "$REPORT_FILE"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    ((ERROR_COUNT++))
    echo "- ❌ $1" >> "$REPORT_FILE"
}

# Check if required tools are installed
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    
    if ! command -v markdown-link-check &> /dev/null; then
        missing_deps+=("markdown-link-check")
    fi
    
    if ! command -v markdownlint &> /dev/null; then
        missing_deps+=("markdownlint")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Install with: npm install -g markdown-link-check markdownlint-cli"
        exit 1
    fi
    
    log_success "All dependencies found"
}

# Validate frontmatter
validate_frontmatter() {
    local file="$1"
    local has_frontmatter=false
    local required_fields=("title" "description" "category" "last_updated")
    local missing_fields=()
    
    # Check if file has frontmatter
    if head -n 1 "$file" | grep -q "^---$"; then
        has_frontmatter=true
        
        # Extract frontmatter
        local frontmatter=$(sed -n '2,/^---$/p' "$file" | sed '$d')
        
        # Check required fields
        for field in "${required_fields[@]}"; do
            if ! echo "$frontmatter" | grep -q "^$field:"; then
                missing_fields+=("$field")
            fi
        done
        
        if [ ${#missing_fields[@]} -gt 0 ]; then
            log_warning "$file: Missing frontmatter fields: ${missing_fields[*]}"
        fi
    else
        log_warning "$file: No frontmatter found"
    fi
}

# Validate markdown structure
validate_structure() {
    local file="$1"
    local filename=$(basename "$file")
    
    # Check for single H1
    local h1_count=$(grep -c "^# " "$file" || true)
    if [ "$h1_count" -gt 1 ]; then
        log_warning "$filename: Multiple H1 headers found ($h1_count)"
    elif [ "$h1_count" -eq 0 ] && [ "$filename" != "README.md" ]; then
        log_warning "$filename: No H1 header found"
    fi
    
    # Check for proper header hierarchy
    local prev_level=0
    while IFS= read -r line; do
        if [[ $line =~ ^#{1,6}[[:space:]] ]]; then
            local current_level=${#line}
            current_level=$((current_level - ${#line##*#}))
            
            if [ $current_level -gt $((prev_level + 1)) ] && [ $prev_level -gt 0 ]; then
                log_warning "$filename: Header hierarchy skip detected (H$prev_level to H$current_level)"
            fi
            prev_level=$current_level
        fi
    done < "$file"
    
    # Check for empty sections
    local empty_sections=$(grep -n "^## " "$file" | while read -r line; do
        local line_num=$(echo "$line" | cut -d: -f1)
        local next_line_num=$((line_num + 1))
        local next_content=$(sed -n "${next_line_num}p" "$file")
        
        if [[ "$next_content" =~ ^##[[:space:]] ]] || [ -z "$next_content" ]; then
            echo "Line $line_num: Empty section"
        fi
    done)
    
    if [ -n "$empty_sections" ]; then
        log_warning "$filename: Empty sections detected"
    fi
}

# Validate code blocks
validate_code_blocks() {
    local file="$1"
    local filename=$(basename "$file")
    
    # Check for code blocks without language specification
    local code_blocks=$(grep -n "^```$" "$file" || true)
    if [ -n "$code_blocks" ]; then
        log_warning "$filename: Code blocks without language specification found"
    fi
    
    # Check for unclosed code blocks
    local opening_blocks=$(grep -c "^```" "$file" || true)
    local closing_blocks=$(grep -c "^```$" "$file" || true)
    local lang_blocks=$(grep -c "^```[a-zA-Z]" "$file" || true)
    
    if [ $((closing_blocks + lang_blocks)) -ne $opening_blocks ]; then
        log_error "$filename: Mismatched code block delimiters"
    fi
}

# Validate links
validate_links() {
    local file="$1"
    local filename=$(basename "$file")
    
    log_info "Checking links in $filename..."
    
    # Use markdown-link-check with custom config
    local link_config=$(cat << 'EOF'
{
  "ignorePatterns": [
    {
      "pattern": "^http://localhost"
    },
    {
      "pattern": "^https://localhost"
    }
  ],
  "httpHeaders": [
    {
      "urls": ["https://github.com"],
      "headers": {
        "Accept": "text/html"
      }
    }
  ],
  "timeout": "10s",
  "retryOn429": true,
  "retryCount": 3
}
EOF
)
    
    echo "$link_config" > /tmp/link-check-config.json
    
    if ! markdown-link-check "$file" --config /tmp/link-check-config.json --quiet; then
        log_error "$filename: Broken links detected"
    fi
    
    rm -f /tmp/link-check-config.json
}

# Validate markdown style
validate_style() {
    local file="$1"
    local filename=$(basename "$file")
    
    # Create markdownlint config
    local lint_config=$(cat << 'EOF'
{
  "MD013": { "line_length": 120 },
  "MD024": { "allow_different_nesting": true },
  "MD033": false,
  "MD041": false
}
EOF
)
    
    echo "$lint_config" > /tmp/markdownlint-config.json
    
    if ! markdownlint "$file" --config /tmp/markdownlint-config.json; then
        log_warning "$filename: Style violations detected"
    fi
    
    rm -f /tmp/markdownlint-config.json
}

# Check for TODO/FIXME comments
check_todos() {
    local file="$1"
    local filename=$(basename "$file")
    
    local todos=$(grep -n -i "TODO\|FIXME\|XXX" "$file" || true)
    if [ -n "$todos" ]; then
        log_warning "$filename: TODO/FIXME comments found"
        echo "$todos" | while read -r todo; do
            log_info "  $todo"
        done
    fi
}

# Check image references
check_images() {
    local file="$1"
    local filename=$(basename "$file")
    local file_dir=$(dirname "$file")
    
    # Find image references
    local images=$(grep -o '!\[.*\](.*\.(png\|jpg\|jpeg\|gif\|svg))' "$file" || true)
    
    if [ -n "$images" ]; then
        echo "$images" | while read -r img; do
            local img_path=$(echo "$img" | sed 's/.*](\([^)]*\)).*/\1/')
            
            # Check if image file exists
            if [ ! -f "$file_dir/$img_path" ] && [ ! -f "$img_path" ]; then
                log_warning "$filename: Missing image: $img_path"
            fi
        done
    fi
}

# Validate table formatting
validate_tables() {
    local file="$1"
    local filename=$(basename "$file")
    
    # Check for malformed tables
    local in_table=false
    local line_num=0
    
    while IFS= read -r line; do
        ((line_num++))
        
        if [[ $line =~ ^\|.*\|$ ]]; then
            if [ "$in_table" = false ]; then
                in_table=true
                local col_count=$(echo "$line" | tr -cd '|' | wc -c)
            else
                local current_cols=$(echo "$line" | tr -cd '|' | wc -c)
                if [ "$current_cols" -ne "$col_count" ]; then
                    log_warning "$filename: Inconsistent table columns at line $line_num"
                fi
            fi
        else
            in_table=false
        fi
    done < "$file"
}

# Main validation function
validate_file() {
    local file="$1"
    local filename=$(basename "$file")
    
    log_info "Validating $filename..."
    ((TOTAL_FILES++))
    
    local file_errors=0
    local initial_error_count=$ERROR_COUNT
    
    # Run all validations
    validate_frontmatter "$file"
    validate_structure "$file"
    validate_code_blocks "$file"
    validate_style "$file"
    validate_tables "$file"
    check_todos "$file"
    check_images "$file"
    
    # Check if this file is being linted by external tools
    if [ "${SKIP_LINK_CHECK:-false}" != "true" ]; then
        validate_links "$file"
    fi
    
    # Determine if file passed validation
    if [ $ERROR_COUNT -eq $initial_error_count ]; then
        ((VALID_FILES++))
        log_success "$filename validated successfully"
    else
        log_error "$filename failed validation"
    fi
}

# Generate summary report
generate_report() {
    cat >> "$REPORT_FILE" << EOF

## Statistics

- **Total Files**: $TOTAL_FILES
- **Valid Files**: $VALID_FILES
- **Files with Issues**: $((TOTAL_FILES - VALID_FILES))
- **Total Errors**: $ERROR_COUNT
- **Total Warnings**: $WARNING_COUNT

## Validation Details

### Checks Performed

- ✅ Frontmatter validation
- ✅ Document structure analysis
- ✅ Code block validation
- ✅ Markdown style checking
- ✅ Table formatting validation
- ✅ TODO/FIXME detection
- ✅ Image reference validation
$([ "${SKIP_LINK_CHECK:-false}" != "true" ] && echo "- ✅ Link validation" || echo "- ⏭️ Link validation (skipped)")

### Quality Score

**Overall Score**: $((100 * VALID_FILES / TOTAL_FILES))%

### Recommendations

EOF

    if [ $ERROR_COUNT -gt 0 ]; then
        echo "- 🔴 **High Priority**: Fix $ERROR_COUNT errors before deploying" >> "$REPORT_FILE"
    fi
    
    if [ $WARNING_COUNT -gt 0 ]; then
        echo "- 🟡 **Medium Priority**: Address $WARNING_COUNT warnings to improve quality" >> "$REPORT_FILE"
    fi
    
    if [ $ERROR_COUNT -eq 0 ] && [ $WARNING_COUNT -eq 0 ]; then
        echo "- 🟢 **Excellent**: All documentation meets quality standards!" >> "$REPORT_FILE"
    fi
    
    cat >> "$REPORT_FILE" << EOF

---

**Report Generated**: $(date)  
**Validator**: XORB Documentation Validator v1.0.0  
**Next Scheduled Review**: $(date -d "+1 week")

EOF
}

# Main execution
main() {
    # Parse command line arguments
    local skip_links=false
    local files_to_check=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-links)
                skip_links=true
                export SKIP_LINK_CHECK=true
                shift
                ;;
            --file)
                files_to_check+=("$2")
                shift 2
                ;;
            --help)
                echo "Usage: $0 [--skip-links] [--file FILE] [--help]"
                echo ""
                echo "Options:"
                echo "  --skip-links    Skip link validation (faster)"
                echo "  --file FILE     Validate specific file"
                echo "  --help          Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check dependencies
    check_dependencies
    
    echo ""
    log_info "Starting documentation validation..."
    echo ""
    
    # Determine files to validate
    if [ ${#files_to_check[@]} -gt 0 ]; then
        for file in "${files_to_check[@]}"; do
            if [ -f "$file" ] && [[ "$file" == *.md ]]; then
                validate_file "$file"
            else
                log_error "File not found or not a markdown file: $file"
            fi
        done
    else
        # Find all markdown files
        while IFS= read -r -d '' file; do
            validate_file "$file"
        done < <(find . -name "*.md" -not -path "./venv/*" -not -path "./.venv/*" -not -path "./node_modules/*" -not -path "./services/ptaas/web/node_modules/*" -print0)
    fi
    
    echo ""
    log_info "Generating report..."
    generate_report
    
    # Print summary
    echo ""
    echo -e "${BLUE}📊 Validation Summary${NC}"
    echo -e "${BLUE}===================${NC}"
    echo "Files validated: $TOTAL_FILES"
    echo "Files passed: $VALID_FILES"
    echo "Errors: $ERROR_COUNT"
    echo "Warnings: $WARNING_COUNT"
    echo "Report saved: $REPORT_FILE"
    
    # Set exit code based on errors
    if [ $ERROR_COUNT -gt 0 ]; then
        echo ""
        log_error "Validation failed with $ERROR_COUNT errors"
        exit 1
    elif [ $WARNING_COUNT -gt 0 ]; then
        echo ""
        log_warning "Validation completed with $WARNING_COUNT warnings"
        exit 0
    else
        echo ""
        log_success "All documentation passed validation!"
        exit 0
    fi
}

# Run main function with all arguments
main "$@"