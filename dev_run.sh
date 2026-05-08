#!/bin/bash
# USAGE: ./dev_run.sh [matcher_ip matcher_port]
# 1. Define the image path relative to where you are running this script
#    Since you are in 'borzoi_GAME', and the image is in 'src/', we point there.
CONTAINER_IMG="src/borzoi_human_predictor.sif"

# 2. Check if the image exists before trying to run
if [ ! -f "$CONTAINER_IMG" ]; then
    echo "❌ Error: Could not find container at $CONTAINER_IMG"
    echo "   Make sure you are running this script from the 'borzoi_GAME' directory."
    exit 1
fi

# Using 'awk' on hostname -i ensures we only get one IP if the machine has multiple interfaces.
pred_ip=$(hostname -I | awk '{print $2}')
#    Find a random free port between 49152 and 65535
pred_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# Handle Optional Matcher Arguments
#    If the user runs: ./dev_run.sh <matcher_ip> <matcher_port>
#    We capture them here. If not, these variables remain empty.
matcher_ip=$1
matcher_port=$2

# Construct the argument string for Python
if [[ -n "$matcher_ip" && -n "$matcher_port" ]]; then
    echo "🔗 Matcher configuration detected: $matcher_ip:$matcher_port"
    PY_ARGS="$pred_ip $pred_port $matcher_ip $matcher_port"
else
    echo "No Matcher arguments provided. Running in standalone mode."
    PY_ARGS="$pred_ip $pred_port"
fi

echo "=========================================================="
echo "🧪 STARTING DEV MODE: ${CONTAINER_IMG}"
echo "=========================================================="
echo "   Mapping host './src'  ---> Container '/src'"
echo "   Mapping host '$PWD'   ---> Container '/mnt' (Working Dir)"
echo "   Predictor : http://$pred_ip:$pred_port"
if [[ -n "$matcher_ip" && -n "$matcher_port" ]]; then
    echo "   Matcher   : http://$matcher_ip:$matcher_port"
else
    echo "   Matcher   : (none -- standalone mode)"
fi
echo "----------------------------------------------------------"

# 3. The Apptainer Command
#    - We bind ./src to /src. This means when the script looks for /src/script_and_utils,
#      it finds your LOCAL editable files, not the ones frozen in the SIF.
apptainer exec --nv \
    --bind ./src:/src \
    --bind $PWD:/mnt \
    --pwd /mnt \
    --env PYTHONPATH="/src/borzoi_API_script_and_utils:/src/script_and_utils:$PYTHONPATH" \
    "$CONTAINER_IMG" \
    python3 /src/script_and_utils/borzoi_predictor_API.py $PY_ARGS