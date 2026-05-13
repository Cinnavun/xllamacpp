#!/bin/bash

# Enable exit on error
set -e

# Function for logging
log_error() {
    echo "ERROR: $1" >&2
}

log_info() {
    echo "INFO: $1"
}

# Get output directory or default to index/whl/cpu
output_dir=${1:-"index/whl/cpu"}

# Get pattern from second arg or default to valid python package version pattern
# Supports both standard versions (v2026.4.8672) and patch versions (v2026.4.8672.1)
pattern=${2:-"^[v]?[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?$"}

# Get the current directory (where the script is run from)
current_dir="$(pwd)"

# Check if all_releases.txt exists
if [ ! -f "$current_dir/all_releases.txt" ]; then
    log_error "all_releases.txt not found in the current directory."
    exit 1
fi

# Create output directory
mkdir -p "$output_dir"

# Create an index html file
cat << EOF > "$output_dir/index.html"
<!DOCTYPE html>
<html>
  <head></head>
  <body>
    <a href="xllamacpp/">xllamacpp</a>
    <br>
  </body>
</html>

EOF

# Create xllamacpp directory
mkdir -p "$output_dir/xllamacpp"

# Create an index html file in xllamacpp directory
cat << EOF > "$output_dir/xllamacpp/index.html"
<!DOCTYPE html>
<html>
  <body>
    <h1>Links for xllamacpp</h1>
EOF

# Temporary aggregation directory for per-version links
tmp_dir="$output_dir/.tmp_xllamacpp_links"
rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"

# Filter releases by pattern
releases=$(grep -E "$pattern" "$current_dir/all_releases.txt")

# Prepare curl headers
headers=('--header' 'Accept: application/vnd.github.v3+json')
if [ -n "$GITHUB_TOKEN" ]; then
    headers+=('--header' "authorization: Bearer $GITHUB_TOKEN")
fi
headers+=('--header' 'content-type: application/json')

# For each release, get all assets
for release in $releases; do
    log_info "Processing release: $release"
    response=$(curl -s "${headers[@]}" \
                    "https://api.github.com/repos/xorbitsai/xllamacpp/releases/tags/$release")
    
    if [ -z "$response" ]; then
        log_error "Empty response from GitHub API for release $release"
        continue
    fi

    if ! echo "$response" | jq -e '.assets' > /dev/null 2>&1; then
        log_error "Invalid or unexpected response from GitHub API for release $release"
        log_error "Response: $response"
        continue
    fi

    # Get release version from release ie v0.1.0-cu121 -> v0.1.0, v2026.4.8672.1 -> v2026.4.8672.1
    release_version=$(echo "$release" | grep -oE "^[v]?[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?")
    # Track first-seen order of versions
    if [ ! -f "$tmp_dir/.${release_version}.seen" ]; then
        echo "$release_version" >> "$tmp_dir/order.txt"
        : > "$tmp_dir/${release_version}.html"
        touch "$tmp_dir/.${release_version}.seen"
    fi
    
    wheel_urls=$(echo "$response" | jq -r '.assets[] | select(.name | endswith(".whl")) | .browser_download_url')
    if [ -z "$wheel_urls" ]; then
        log_error "No wheel files found for release $release"
        continue
    fi

    echo "$wheel_urls" | while read -r asset; do
        echo "    <a href=\"$asset\">$asset</a>" >> "$tmp_dir/${release_version}.html"
        echo "    <br>" >> "$tmp_dir/${release_version}.html"
    done
done

if [ -f "$tmp_dir/order.txt" ]; then
    while IFS= read -r ver; do
        echo "    <h2>$ver</h2>" >> "$output_dir/xllamacpp/index.html"
        cat "$tmp_dir/${ver}.html" >> "$output_dir/xllamacpp/index.html"
    done < "$tmp_dir/order.txt"
fi

# Close HTML and clean up
echo "  </body>" >> "$output_dir/xllamacpp/index.html"
echo "</html>" >> "$output_dir/xllamacpp/index.html"
echo "" >> "$output_dir/xllamacpp/index.html"

rm -rf "$tmp_dir"

log_info "Index generation complete. Output directory: $output_dir"
