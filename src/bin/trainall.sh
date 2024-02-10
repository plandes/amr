#!/bin/bash

# @meta {desc: 'trains all models', date: '2024-02-05'}
# @meta {doc: 'run from root directory'}


BACKGROUND=0
TARGET_DIR=target/trainall
MODELS="\
	parse-spring
	generate-t5wtense-base \
	generate-t5wtense-large \
	parse-xfm-base \
	parse-xfm-large"

# TODO
MODELS=generate-t5wtense-large


init() {
    MODEL=$1 ; shift
    CONF=${TARGET_DIR}/${MODEL}.conf
    LOG=${TARGET_DIR}/${MODEL}.log
}

write_conf() {
    mkdir -p $(dirname $CONF)
    cat train-config/${MODEL}.conf > $CONF
    cat <<EOF >> ${CONF}
[amr_prep_manager]
preppers = instance: tuple:
  amr_prep_lp_corp_prepper,
  amr_prep_bio3_corp_prepper,
  amr_prep_rel3_corp_prepper
EOF
}


train_model() {
    echo "training model ${MODEL}..."
    if [ $BACKGROUND -eq 1 ] ; then
	./amr train --config $CONF > $LOG 2>&1 &
    else
	./amr train --config $CONF
    fi
}

main() {
    for model in $MODELS ; do
	init $model
	write_conf
	train_model
    done
}

main
