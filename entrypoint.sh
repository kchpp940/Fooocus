#!/bin/bash

ORIGINALDIR=/content/app
# Use predefined DATADIR if it is defined
[[ x"${DATADIR}" == "x" ]] && DATADIR=/content/data

export FOOOCUS_USER_DATA_DIR="$DATADIR"

function do_preflight() {
    # 将后续参数原样透传给 launch.py --preflight-check
    # 示例:
    #   preflight                    -> 严格模式，有 FAIL 非零退出 (默认 strict)
    #   preflight --json             -> strict + JSON 输出 (CI 用)
    #   preflight --mode healthcheck -> Docker HEALTHCHECK 用
    #   preflight --mode report      -> 只出报告，退出码永为 0
    #   preflight --json --mode strict -> CI 严格模式 + JSON
    cd $ORIGINALDIR
    exec python launch.py --preflight-check "$@"
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
