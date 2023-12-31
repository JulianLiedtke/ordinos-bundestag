import abc
import logging
from math import ceil
from random import randrange

import numpy as np

import gmpy2 as gmpy
from src.util.abb_logging import ABBLogger

log = logging.getLogger(__name__)


class ABB():

    __metaclass__ = abc.ABCMeta

    def __init__(self, prot_suite, pk, num_shares, threshold, sk):
        self.op_logger = ABBLogger()
        self.prot_suite = prot_suite
        self.pk = pk
        self.num_shares = num_shares
        self.threshold = threshold
        self.sk = sk
        self.rand = gmpy.random_state(randrange(123456789))

        self.enc_zero = self.enc_no_r(0)
        self.enc_one = self.enc_no_r(1)

    @classmethod
    def gen_trustee_abbs(cls, bits, num_shares, threshold, prot_suite_cls):
        pk, sks = cls.keygen(bits, num_shares, threshold)
        prot_suits = [prot_suite_cls() for _ in range(num_shares)]
        abbs = [cls(prot_suits[i], pk, num_shares, threshold, sks[i]) for i in range(num_shares)]
        for i, prot_suit in enumerate(prot_suits):
            prot_suit.set_abb(abbs[i])
        return abbs

    def create_local_abb(self):
        """
        TODO: explanation why this is needed
        """
        return self.__class__(None, self.pk, self.num_shares, self.threshold, None)

    def get_bits_for_size(self, biggest_possible_number):
        """
        TODO describe
        """
        used_bits = 1
        if biggest_possible_number > 0:
            # convert biggest_possible_number to float, because np.log2 requires it for large number
            bits = ceil(np.log2(biggest_possible_number * 1.0))

            # allowed_bit_numbers = [2,3,4,5,6,8,10,12,16,20,24,32,40,48,64] # 111222444888
            allowed_bit_numbers = [2, 4, 8, 16, 32, 64] # 111222444888
            #allowed_bit_numbers = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            allowed_bit_numbers.reverse()
            for allowed_bit in allowed_bit_numbers:
                if bits > allowed_bit:
                    break
                used_bits = allowed_bit
        log.debug("Use bits_int=" + str(used_bits))
        return used_bits

    def convert_to_cipher(self, val):
        if not isinstance(val, Ciphertext):
            return self.enc_no_r(val)
        else:
            return val

    def deserialize_cipher(self, serialized_cipher):
        return Ciphertext.deserialize(self, serialized_cipher)

    @abc.abstractclassmethod
    def keygen(cls, bits, num_shares, threshold):
        pass

    @abc.abstractmethod
    def init_cipher(self, val):
        pass

    @abc.abstractmethod
    def get_random_plaintext(self):
        pass

    @abc.abstractmethod
    def enc_get_r(self):
        pass

    @abc.abstractmethod
    def enc(self, plain):
        pass

    @abc.abstractmethod
    def enc_no_r(self, plain):
        pass

    @abc.abstractmethod
    def dec(self, cipher):
        pass

    @abc.abstractmethod
    def eq(self, cipher1, cipher2, bits):
        pass

    @abc.abstractmethod
    def gt(self, cipher1, cipher2, bits):
        pass

    @abc.abstractmethod
    def randomize(self, cipher, r=None):
        pass

    @abc.abstractmethod
    def eval_random_randomess(self, cipher):
        pass

    @abc.abstractmethod
    def eval_add_protocol(self, cipher1, cipher2):
        pass

    @abc.abstractmethod
    def eval_sub_protocol(self, cipher1, cipher2):
        pass

    @abc.abstractmethod
    def eval_mul_protocol(self, cipher1, cipher2):
        pass

    def serialize(self):
        return {
            'prot_suite': self.prot_suite.serialize() if self.prot_suite is not None else None,
            'pk': self.pk.serialize(),
            'num_shares': self.num_shares,
            'threshold': self.threshold,
            'sk': self.sk.serialize() if self.sk is not None else None
        }

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, serialized):
        raise NotImplementedError


class Ciphertext():

    __metaclass__ = abc.ABCMeta

    def __init__(self, abb, val):
        self.abb = abb
        self.val = val

    def __str__(self):
        return '{}'.format(self.val)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return self.abb.eval_add_protocol(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.abb.eval_sub_protocol(self, other)

    def __rsub__(self, other):
        return self.abb.eval_sub_protocol(other, self)

    def __mul__(self, other):
        return self.abb.eval_mul_protocol(other, self)

    def serialize(self):
        return str(self.val)

    @classmethod
    def deserialize(cls, abb, s):
        return cls(abb, gmpy.mpz(s))
