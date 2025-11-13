#!/usr/bin/env python

from zensols.cli import CliHarness


def init_deeplearn():
    try:
        from zensols.deeplearn import TorchConfig
        TorchConfig.init()
    except ModuleNotFoundError:
        pass


if (__name__ == '__main__'):
    init_deeplearn()
    args: str = ''
    model = {
        0: None,
        1: 'no-conf',
        2: 'parse-xfm-base',
        3: 'parse-t5',
        4: 'parse-spring',
        5: 'generate-t5wtense-base',
    }[0]
    if model is None:
        args = ' -c resources/model/inference.conf'
    elif model == 'no-conf':
        args = ''
    else:
        args = f' -c train-config/{model}.conf'
    harness = CliHarness(
        src_dir_name='src',
        app_factory_class='zensols.amr.ApplicationFactory',
        proto_args='proto' + args,
        proto_factory_kwargs={'reload_pattern': r'^zensols\.amr\.(?!annotate)'}
    )
    harness.run()
