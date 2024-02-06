#!/bin/bash

# @meta {desc: 'trains all models', date: '2024-02-05'}
# @meta {doc: 'run from root directory'}


TARGET_DIR=target/trainall


init() {
    MODEL=$1 ; shift
    MODEL_TYPE=$1 ; shift
    CONF=${TARGET_DIR}/${MODEL}.conf
    LOG=${TARGET_DIR}/${MODEL}.log
}

write_conf() {
    mkdir -p $(dirname $CONF)
    cat train-config/parse-${MODEL}.conf > $CONF
    cat <<EOF >> ${CONF}
[amr_prep_manager]
preppers = instance: tuple:
  amr_prep_lp_corp_prepper,
  amr_prep_bio3_corp_prepper,
  amr_prep_rel3_corp_prepper
EOF
}

train_parse() {
    echo "training parse model ${MODEL}..."
    ./amr train --config $CONF > $LOG 2>&1 &
}

main() {
    for model in spring ; do
	init $model parse
	write_conf
	train_parse
    done
}

main
