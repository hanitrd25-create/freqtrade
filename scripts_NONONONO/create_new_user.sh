#!/bin/bash

# 입력된 파라미터가 있는지 확인
if [ -z "$1" ]; then
    # 파라미터가 없을 경우 (username이 입력되지 않았을 때)
    echo "No username provided. Please provide a username."
    exit 1
else
    # 파라미터가 있을 경우 (username이 입력되었을 때)
    new_user_name=$1
    echo "Username is: $new_user_name"
fi

cd /mnt/d/develop/freqtrade
source .venv/bin/activate


# new_user_name="cp_strategies"

freqtrade create-userdir --userdir ${new_user_name}
freqtrade new-config --config ${new_user_name}/config.json
