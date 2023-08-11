"""
5G LDPC codes constructor
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

import json
import os
import argparse

import numpy as np
import galois

from ldpc_soft_py.ldpc import Alist

CODES_DIR = 'codes'


def factor_to_index(lifting_size):
    """
    Convert factor to circulant set index
    Table 5.3.2-1: Sets of LDPC lifting size Z encoded as a function
    """
    if lifting_size in [2, 4, 8, 16, 32, 64, 128, 256]:
        return 1
    if lifting_size in [3, 6, 12, 24, 48, 96, 192, 384]:
        return 2
    if lifting_size in [5, 10, 20, 40, 80, 160, 320]:
        return 3
    if lifting_size in [7, 14, 28, 56, 112, 224]:
        return 4
    if lifting_size in [9, 18, 36, 72, 144, 288, 576]:
        return 5
    if lifting_size in [11, 22, 44, 88, 176, 352]:
        return 6
    if lifting_size in [13, 26, 52, 104, 208]:
        return 7
    if lifting_size in [15, 30, 60, 120, 240]:
        return 8
    raise ValueError('Can not get set index for factor ', lifting_size)


def gen_pcm(block_len, rate, base_graph):
    """
    Generate parity check matrix
    """
    n_inf_bits = int(np.round(block_len * rate))
    pcm_exp, k_base = get_pcm_expander(n_inf_bits, base_graph)
    n_base = round(k_base / rate + 2)
    pcm_exp = pcm_exp[:(n_base - k_base), : n_base]
    factor = round(n_inf_bits / k_base)
    return expand_pcm(factor, pcm_exp).astype(np.uint8), k_base, factor


def get_generator(pcm, k_base, factor):
    """
    Generator matrix construction
    """
    inv = np.array(np.linalg.inv(galois.GF2(
        pcm[:(4 * factor), (k_base * factor): (k_base + 4) * factor]
    ))).astype(np.uint8)
    inf_part_1 = pcm[:4 * factor, :k_base * factor]
    inf_part_2 = pcm[4 * factor:, :(k_base + 4) * factor]
    generator_l = np.hstack([np.eye(k_base * factor), np.mod(inv @ inf_part_1, 2).T])
    generator_r = np.hstack([np.eye((k_base + 4) * factor), inf_part_2.T])
    return np.mod(generator_l @ generator_r, 2).astype(np.uint8)


def get_pcm_expander(n_inf_bits, base_graph):
    """
    Get parity check matrix in the expander form
    """
    # Load base-graph
    if base_graph not in [1, 2]:
        raise ValueError('Unknown base graph value')
    with open(f'ldpc_5g_data/bg{base_graph}.json', 'r', encoding='utf-8') as filedesc:
        bg_data = json.load(filedesc)

    pcm_base = np.array(bg_data['H'])
    kb_max = pcm_base.shape[1] - pcm_base.shape[0]

    if n_inf_bits > 640:
        k_base = kb_max
    elif n_inf_bits > 560:
        k_base = 9
    elif n_inf_bits > 192:
        k_base = 8
    else:
        k_base = 6
    factor = int(np.round(n_inf_bits / k_base))

    print(f'Factor: {factor}, K_b = {k_base}.')
    print(f'K = {n_inf_bits}/{k_base * factor} (intended/actual)')
    lift_index = factor_to_index(factor)
    print(f'Index: {lift_index}.')

    pcm_expander = np.array(bg_data['sets'][str(lift_index)])
    return np.hstack([pcm_expander[:, :k_base], pcm_expander[:, kb_max:]]), k_base


def expand_pcm(factor, pcm_exp):
    """
    Expand the base parity check matrix
    """
    # Only binary codes supported
    n_checks, blocklen_base = pcm_exp.shape
    pcm = []
    for i in range(n_checks):
        layer = []
        for j in range(blocklen_base):
            shift = pcm_exp[i, j]
            if shift == -1:
                layer.append(np.zeros((factor, factor)))
            else:
                layer.append(np.roll(np.eye(factor), shift, axis=1))
        pcm.append(np.hstack(layer))
    return np.vstack(pcm)


def get_filename_template(pcm, factor):
    """
    Generate filename template for LDPC code:
    - generator matrix (np.savetxt(), space-separated) (*_generator.txt)
    - parity check matrix in the ALIST format (*_pcm.alist)
    - JSON file that also keeps (*.json):
        - Information bits indices
        - Punctured indices
    """
    n_checks, block_len = pcm.shape
    n_inf_bits = block_len - n_checks
    block_len -= 2 * factor
    return f'ldpc_5g_k{n_inf_bits}_n{block_len}'


def generate_5g_code(inf_bits_count, coding_rate, base_graph):
    """
    Main function
    """
    # Generate parity check matrix:
    print('Creating the parity check matrix...')
    pcm, k_base, factor = gen_pcm(
        int(inf_bits_count / coding_rate),
        coding_rate,
        base_graph
    )
    filename_template = get_filename_template(pcm, factor)

    pcm_file = filename_template + '_pcm.alist'
    Alist.write(pcm, os.path.join(CODES_DIR, pcm_file))
    print('Creating the generator matrix...')
    try:
        generator_mtx = get_generator(pcm, k_base, factor)
        gen_mtx_file = filename_template + '_generator.txt'
        np.savetxt(os.path.join(CODES_DIR, gen_mtx_file), generator_mtx, delimiter=' ', fmt='%d')
    except KeyboardInterrupt:
        print('Interrupted. Generator matrix will not be created')
        gen_mtx_file = None

    code = {
        'pcm': filename_template + '_pcm.alist',
        'punc_idx': f'0:{2 * factor - 1}',
        'inf_bits': f'0:{pcm.shape[1] - pcm.shape[0]}'
    }
    if gen_mtx_file:
        code['generator'] = gen_mtx_file

    json_file = os.path.join(CODES_DIR, filename_template + '.json')
    with open(json_file, 'w', encoding='utf-8') as filedesc:
        json.dump(code, filedesc, indent=2)
    print(f'Successfully generated {json_file}')
    return json_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 5G-LDPC codes')
    parser.add_argument('--k', help='The number of information bits')
    parser.add_argument('--rate', help='Intended coding rate')
    parser.add_argument('--BG', help='Base graph: \'1\' or \'2\'')
    args = parser.parse_args()
    try:
        generate_5g_code(
            inf_bits_count=float(args.k),  # Intended information bits count
            coding_rate=float(args.rate),  # Intended coding rate
            base_graph=int(args.BG)
        )
    except TypeError:
        print(f'Usage: {__file__} -h')
    except ValueError:
        print('Code was not created. Try to vary the number of information bits')
        print('Factor must be [2, 3, 5, 7, 9, 11, 13, 15] X 2**N')
