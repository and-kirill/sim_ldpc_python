"""
LDPC simulation and postprocessing workflow
"""

# This file is part of the simulator_awgn_python distribution
# https://github.com/and-kirill/sim_ldpc_python/.
# Copyright (c) 2023 Kirill Andreev.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import json

from simulator_awgn_python.tools import run_all_experiments
from ldpc_experiment import LdpcExperiment


def get_experiment(**kwargs):
    """
    Get default LDPC experiment
    """
    src_dir = kwargs.get('src_dir')
    code_filename = os.path.abspath(os.path.join(kwargs.get('src_dir'), kwargs.get('code')))
    if not os.path.isfile(code_filename):
        raise ValueError('JSON file does not exist')
    with open(code_filename, 'r', encoding='utf-8') as file_desc:
        code_params = json.load(file_desc)

    if 'pcm' not in code_params:
        raise TypeError('LDPC code must have a parity check matrix')
    code_params['pcm'] = os.path.abspath(os.path.join(src_dir, code_params['pcm']))
    if 'generator' in code_params:
        code_params['generator'] = os.path.abspath(os.path.join(src_dir, code_params['generator']))

    return LdpcExperiment(
        modulation=kwargs.get('modulation'),
        n_iterations=kwargs.get('n_iterations'),
        algorithm=kwargs.get('algorithm'),
        llr_scale=kwargs.get('llr_scale'),
        **code_params
    )


if __name__ == '__main__':
    import logging
    from simulator_awgn_python.tools import enable_log
    # Usage: python3 main.py --config=experiment.json
    # Default file is experiment.json
    # To create the proposed code to simulate, run the following command:
    # python3 ldpc_5g.py --k=120 --rate=0.2 --BG=2
    LOGFILE = 'simulator.log'
    enable_log('simulator_awgn_python.simulator', logging.INFO, LOGFILE)
    print(f'Check simulator events in {LOGFILE}')
    run_all_experiments(get_experiment, address='127.0.0.1', start_port=8888, update_ms=5000)
