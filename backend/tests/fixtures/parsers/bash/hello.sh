source ./utils.sh

greet() {
    local name="$1"
    format_name "$name"
}

format_name() {
    echo "$1" | tr a-z A-Z
}

standalone_helper() {
    greet "world"
}
