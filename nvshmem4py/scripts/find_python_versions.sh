declare -A seen
IFS=':' read -ra paths <<< "$PATH"
for dir in "${paths[@]}"; do
    for p in "$dir"/python3.[0-9]*; do
    [[ -e "$p" ]] || continue  # Skip if no match or broken glob
        # Ignore invalid versions
        if [[ $(basename "$p") =~ ^python3\.[0-9]+(\.[0-9]+)*$ ]]; then
            :
        else
           continue
        fi
        if [ -x "$p" ]; then
            v=$(
                "$p" -c 'import sys; print("%d.%d" % (sys.version_info[0], sys.version_info[1]))' 2>/dev/null
            )
            if [[ $v =~ ^3.[0-9]+$ ]]; then
                major=$(echo $v | cut -d. -f1)
                minor=$(echo $v | cut -d. -f2)
                if (( major > 3 || (major == 3 && minor >= 9) )); then
                    if [[ -z ${seen[$v]} ]]; then
                        seen[$v]=1
                        echo "$v|$p"
                    fi
                fi
            fi
        fi
    done
done