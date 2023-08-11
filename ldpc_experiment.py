"""
LDPC simulation routines
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

import numpy as np
from simulator_awgn_python.simulator import DataEntry
from simulator_awgn_python.channel import AwgnQAMChannel
from simulator_awgn_python.tools import range_from_string
from ldpc_soft_py.ldpc import LdpcDecoder, Alist, lib_compile

# Global per-process instances of AWGN LDPC instance
G_LDPC_IMPL = None


class LdpcAwgn:
    """
    Run LDPC decoder with AWGN channel.
    Has a lot of bulky data, is not an Experiment instance required by the simulator.
    """
    def __init__(self, **kwargs):
        # Load generator matrix (if present)
        if 'generator' in kwargs:
            self.generator = np.loadtxt(kwargs.get('generator')).astype(np.uint8)
            self.iwd_len = self.generator.shape[0]
        else:
            self.generator = None
        # Set punctured nodes
        if 'punc_idx' in kwargs:
            self.punc_idx = range_from_string(kwargs.get('punc_idx'))
        else:
            self.punc_idx = None
        # Load decoding algorithm parameters
        self.algorithm = kwargs.get('algorithm')
        if self.algorithm not in ['sum_product', 'min_sum', 'layered_min_sum']:
            raise TypeError(f'Unknown decoding algorithm {self.algorithm}')
        if self.algorithm == 'layered_min_sum':
            self.llr_scale = kwargs.get('llr_scale')
            if self.llr_scale is None:
                raise TypeError('Provide LLR scale for layered min-sum')

        self.n_iterations = kwargs.get('n_iterations')

        # Load information bits indices. If not provided, calculate BER using a whole codeword
        self.inf_bits = kwargs.get('inf_bits')
        if self.inf_bits is not None:
            self.inf_bits = range_from_string(self.inf_bits)

        self.channel = AwgnQAMChannel(kwargs.get('modulation'))
        self.decoder_impl = LdpcDecoder(kwargs.pop('pcm'))

    def encode(self, rng):
        """
        Generate a codeword. If there is no generator matrix, return a zero codeword
        """
        # Encode
        if self.generator is None:
            return np.zeros(self.decoder_impl.block_len).astype(np.uint8)
        iwd = (rng.random(size=self.iwd_len) < 0.5).astype(np.uint8)
        return np.mod(iwd @ self.generator, 2)

    def decode(self, llr_channel):
        """
        Apply ouncturing and perform decoding
        """
        if self.punc_idx is not None:
            llr_channel[self.punc_idx] = 0
        if self.algorithm == 'sum_product':
            llr_out = self.decoder_impl.sum_product(llr_channel, self.n_iterations)
        elif self.algorithm == 'min_sum':
            llr_out = self.decoder_impl.min_sum(llr_channel, self.n_iterations)
        elif self.algorithm == 'layered_min_sum':
            llr_out = self.decoder_impl.layered_min_sum(
                llr_channel,
                self.n_iterations,
                self.llr_scale
            )
        else:
            raise NotImplementedError('Unknown decoding algorithm')
        return llr_out < 0

    def run(self, snr_db, rng):
        """
        Perform single experiment trial
        """
        cwd = self.encode(rng)
        # Modulation and AWGN channel
        use_adapter = self.generator is None
        [llr_channel, in_ber, in_ser] = self.channel.run(cwd, snr_db, rng, use_adapter=use_adapter)
        cwd_hat = self.decode(llr_channel)

        if self.inf_bits is not None:
            out_ber = np.mean(cwd_hat[self.inf_bits] != cwd[self.inf_bits])
        else:
            out_ber = np.mean(cwd_hat != cwd)

        # Fill output channel statistics
        stats = DataEntry()
        stats.in_ber = in_ber
        stats.in_ser = in_ser
        stats.out_ber = out_ber
        stats.out_fer = out_ber > 0
        stats.n_exp = 1
        return stats


class LdpcExperiment:
    """
    Class wrapper to perform independent tests within a Simulator
    """
    def __init__(self, **kwargs):
        lib_compile()

        self.params = kwargs
        self.modulation = kwargs.get('modulation')

        n_checks, blocklen = Alist.read(kwargs.get('pcm')).shape
        # Re-derive code parameters
        n_inf_bits = blocklen - n_checks
        if 'punc_idx' in kwargs:
            blocklen = blocklen - len(range_from_string(kwargs.get('punc_idx')))
        algorithm = kwargs.get('algorithm')
        alg_str = algorithm
        if algorithm == 'layered_min_sum':
            llr_scale = kwargs.get('llr_scale')
            alg_str += f'_{llr_scale:1.3f}'

        n_iter = kwargs.get('n_iterations')

        self.title = f'LDPC code k = {n_inf_bits}, n = {blocklen} bits. {alg_str} decoder,'
        self.title += f' {n_iter} iterations, {self.modulation.upper()} modulation.'

        pickle_file = f'ldpc_k{n_inf_bits}_n{blocklen}_{self.modulation}_{alg_str}_iter{n_iter}'
        self.filename = f'data/{pickle_file}.pickle'

    def get_filename(self):
        """
        Generate filename template from settings
        """
        return self.filename

    def get_title(self):
        """
        Human-readable title for plot header
        """
        return self.title

    def run(self, snr_db, rng):
        """
        Perform single experiment trial
        """
        global G_LDPC_IMPL
        return G_LDPC_IMPL.run(snr_db, rng)

    def init_worker(self):
        """
        Initialize per-process global variables
        """
        # Initialize channel
        global G_LDPC_IMPL
        G_LDPC_IMPL = LdpcAwgn(**self.params)
