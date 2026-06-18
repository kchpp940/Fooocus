#!/bin/bash

ORIGINALDIR=/content/app
# Use predefined DATADIR if it is defined
[[ x"${DATADIR}" == "x" ]] && DATADIR=/content/data

export FOOOCUS_USER_DATA_DIR="$DATADIR"

function do_preflight() {
    local json_mode=0
    local exit_on_error=1
    for arg in "$@"; do
        case "$arg" in
            --json) json_mode=1 ;;
            --no-exit) exit_on_error=0 ;;
        esac
    done

    cd $ORIGINALDIR
    if [[ $json_mode -eq 1 ]]; then
        python launch.py --preflight-check --json
    else
        python launch.py --preflight-check
    fi
    local rc=$?
    if [[ $exit_on_error -eq 0 ]]; then
        exit 0
    fi
    exit $rc
}

if [[ "$1" == "preflight" ]]; then
    shift
    do_preflight "$@"
fi

cd $ORIGINALDIR

# Make persistent dir from original dir
function mklink () {
	mkdir -p $DATADIR/$1
	ln -s $DATADIR/$1 $ORIGINALDIR 2>/dev/null || true
}

# Copy old files from import dir
function import () {
	(test -d /import/$1 && cd /import/$1 && cp -Rpn . $DATADIR/$1/)
}

# models
mklink models
# Copy original files (only on first run, if models dir empty or newly created)
if [ -z "$(ls -A $ORIGINALDIR/models 2>/dev/null)" ]; then
    (cd $ORIGINALDIR/models.org && cp -Rpn . $ORIGINALDIR/models/)
fi
# Import old files
import models

# outputs
mklink outputs
# Import old files
import outputs

# user presets
mklink user_presets
# Import old files
import user_presets

# Start application
python launch.py $*
