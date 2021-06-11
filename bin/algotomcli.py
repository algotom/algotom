#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. module:: algotom.py
   :platform: Unix
   :synopsis: Provides cli support to algotom

"""

import os
import sys
import time
import argparse

from datetime import datetime

from algotom.util import log
from algotom.util import config

import numpy as np
import algotom.io.converter as conv
import algotom.io.loadersaver as losa

def init(args):
    if not os.path.exists(str(args.config)):
        config.write(str(args.config))
    else:
        raise RuntimeError("{0} already exists".format(args.config))

def status(args):
    config.show_config(args)

def explore(args):
    log.info('file path : %s' % args.file_path)
    log.info('output base : %s' % args.output_base)


def main():
    home = os.path.expanduser("~")
    logs_home = home + '/logs/'

    # make sure logs directory exists
    if not os.path.exists(logs_home):
        os.makedirs(logs_home)

    lfname = logs_home + 'pv_' + datetime.strftime(datetime.now(), "%Y-%m-%d_%H:%M:%S") + '.log'
    log.setup_custom_logger(lfname)
    log.warning('Logs are saved at: %s' % lfname)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', **config.SECTIONS['general']['config'])
    explore_params   = config.EXPLORE_PARAMS

    cmd_parsers = [
        ('init',        init,           (),                              "Create configuration file"),
        ('status',      status,         explore_params,                  "Show the algotom-cli status"),
        ('explore',     explore,        explore_params,                  "Explore a tomographic data in the hdf/nxs format"),
    ]

    subparsers = parser.add_subparsers(title="Commands", metavar='')

    for cmd, func, sections, text in cmd_parsers:
        cmd_params = config.Params(sections=sections)
        cmd_parser = subparsers.add_parser(cmd, help=text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        cmd_parser = cmd_params.add_arguments(cmd_parser)
        cmd_parser.set_defaults(_func=func)

    args = config.parse_known_args(parser, subparser=True)
    args.lfname = lfname
    
    try: 
        # load args from default (config.py) if not changed
        config.log_values(args)
        args._func(args)
        # undate meta5.config file
        sections = config.EXPLORE_PARAMS
        config.write(args.config, args=args, sections=sections)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
