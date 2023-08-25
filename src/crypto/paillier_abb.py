import logging
from random import randint, choice
from src.protocols.protocol_suite import ProtocolSuite

from src.crypto.abb import ABB, Ciphertext
from src.util.primes import PrimeStorage
from src.util.utils import eval_polynomial, ext_euclid, calc_lambda
import numpy as np
import hashlib

import gmpy2 as gmpy

log = logging.getLogger(__name__)


class PaillierABB(ABB):

    @classmethod
    def keygen(cls, bits, num_shares, threshold):
        """
        generates public and private key for the threshhold variant of paillier.
        """
        if threshold < 2:
            raise('Threshold should be at least 2, but is {}'.format(threshold))
        primes = PrimeStorage()
        ((p, p_), (q, q_)) = primes.getRandomSafePrimes(bits // 2)
        
        n = p * q
        m = p_ * q_

        # find secret
        d = ext_euclid(n, m)

        pk = PublicPaillierKey(n)

        # TODO: outsource shamir?
        # Shamir secret sharing: determine polynomial
        coeffs = [d] + [randint(0, n*m) for _ in range(threshold-1)]
        # determine shares
        shares = [eval_polynomial(coeffs, i, n*m)
                for i in range(1, num_shares + 1)]
        key_shares = [PrivateKeyShare(
            shares[i-1], i, len(shares), threshold, pk) for i in range(1, num_shares + 1)]

        # TODO: for verification we need:
        # - v, a generator of Z^*_(n^2)
        # - verification key for each decryption party

        return pk, key_shares

    def __init__(self, prot_suite, pk, num_shares, threshold, sk):
        super().__init__(prot_suite, pk, num_shares, threshold, sk)
        self.counter_gt = 0
        self.counter_eq = 0
        self.counter_dec = 0
        self.counter_mul = 0
        self.counter_add = 0
        self.counter_sub = 0
        self.bit_list = []
    
    def start_counter(self):
        self.counter_gt = 0
        self.counter_eq = 0
        self.counter_dec = 0
        self.counter_mul = 0
        self.counter_add = 0
        self.counter_sub = 0
        self.bit_list = []

    def init_cipher(self, val):
        return PaillierCiphertext(self, val)

    def deserialize_cipher(self, serialized_cipher):
        return PaillierCiphertext.deserialize(self, serialized_cipher)

    def get_random_plaintext(self):
        """returns a possible plaintext"""
        return gmpy.mpz_urandomb(self.rand, self.pk.bits) % self.pk.n

    def enc_get_r(self, plain):
        if not isinstance(plain, (int, gmpy.mpz().__class__)):
            raise ValueError('{} is not a valid number (it is of type {}).'.format(plain, type(plain)))
        r = self.get_r()
        x = gmpy.powmod(r, self.pk.n, self.pk.n_sq)
        cipher = (gmpy.powmod(self.pk.g, plain, self.pk.n_sq) * x) % self.pk.n_sq
        return PaillierCiphertext(self, cipher), r

    def enc(self, plain):
        if not isinstance(plain, (int, gmpy.mpz().__class__)):
            raise ValueError('{} is not a valid number (it is of type {}).'.format(plain, type(plain)))
        return self.enc_get_r(plain)[0]

    def enc_no_r(self, plain):
        if not isinstance(plain, (int, gmpy.mpz().__class__)):
            raise ValueError('{} is not a valid number (it is of type {}).'.format(plain, type(plain)))
        r = 1
        x = gmpy.powmod(r, self.pk.n, self.pk.n_sq)
        cipher = (gmpy.powmod(self.pk.g, plain, self.pk.n_sq) * x) % self.pk.n_sq
        return PaillierCiphertext(self, cipher)

    def dec(self, cipher):
        self.counter_dec += 1
        self.op_logger.log_dec()
        plaintext_share = self.sk.compute_plaintext_share(cipher)
        serialized_share = (plaintext_share[0], plaintext_share[1].serialize())
        serialized_shares = self.prot_suite.broadcast_and_receive(serialized_share)
        plaintext_shares = [(s[0], PaillierCiphertext.deserialize(self, s[1])) for _, s in serialized_shares]

        while len(plaintext_shares) > self.sk.threshold:
            plaintext_shares.remove(choice(plaintext_shares))

        return self.sk.reconstruct_plaintext(plaintext_shares)

    def eq(self, input1, input2, bits):
        self.counter_eq += 1
        cipher1 = self.convert_to_cipher(input1)
        cipher2 = self.convert_to_cipher(input2)
        top_level = self.op_logger.is_logging_active
        self.op_logger.log_eq(bits)
        if top_level:
            self.op_logger.start_op()
            top_level = True
        result = self.prot_suite.eq(cipher1, cipher2, bits)
        if top_level:
            self.op_logger.op_done()
        return result

    def gt(self, input1, input2, bits):
        self.counter_gt += 1
        self.bit_list.append(bits)
        cipher1 = self.convert_to_cipher(input1)
        cipher2 = self.convert_to_cipher(input2)
        top_level = self.op_logger.is_logging_active
        self.op_logger.log_gt(bits)
        if top_level:
            self.op_logger.start_op()
            top_level = True
        result = self.prot_suite.gt(cipher1, cipher2, bits)
        if top_level:
            self.op_logger.op_done()
        return result

    def randomize(self, cipher, r=None):
        if not r:
            r = self.get_r()
        val = (cipher.val * gmpy.powmod(r, self.pk.n, self.pk.n_sq)) % self.pk.n_sq
        return PaillierCiphertext(self, val)

    def get_r(self):
        while True:
            r = gmpy.mpz_urandomb(self.rand, self.pk.bits)
            if 0 < r < self.pk.n and gmpy.gcd(r, self.pk.n) == 1:
                return r

    def eval_add_protocol(self, cipher1, cipher2):
        self.counter_add += 1
        if isinstance(cipher1, PaillierCiphertext):
            if isinstance(cipher2, PaillierCiphertext):
                return self._add_ciphertexts(cipher1, cipher2)
            else:
                return self._add_constant(cipher1, cipher2)
        else:
            if isinstance(cipher2, PaillierCiphertext):
                return self._add_constant(cipher2, cipher1)
            else:
                raise ValueError('No ciphertext for ciphertext addition found.')

    def _add_ciphertexts(self, cipher1, cipher2):
        return PaillierCiphertext(self, (cipher1.val * cipher2.val) % self.pk.n_sq)

    def _add_constant(self, cipher, const):
        return self._add_ciphertexts(cipher, self.enc_no_r(const))

    def eval_sub_protocol(self, cipher1, cipher2):
        self.counter_sub += 1
        enc1 = None
        enc2 = None
        if isinstance(cipher1, PaillierCiphertext):
            enc1 = cipher1
        else:
            enc1 = self.enc_no_r(cipher1)
        if isinstance(cipher2, PaillierCiphertext):
            enc2 = cipher2
        else:
            enc2 = self.enc_no_r(cipher2)

        inv_enc2 = self._invert(enc2)
        return enc1 + inv_enc2

    def _invert(self, cipher):
        inv = gmpy.invert(cipher.val, self.pk.n_sq)
        return PaillierCiphertext(self, inv)

    def eval_mul_protocol(self, cipher1, cipher2):
        self.counter_mul += 1
        if isinstance(cipher1, PaillierCiphertext):
            if isinstance(cipher2, PaillierCiphertext):
                self.op_logger.log_mul()
                return self.prot_suite.mul(cipher1, cipher2)
            else:
                return self._mul_const(cipher1, cipher2)
        else:
            if isinstance(cipher2, PaillierCiphertext):
                return self._mul_const(cipher2, cipher1)
            else:
                raise ValueError('No ciphertext for ciphertext multiplication found.')

    def _mul_const(self, cipher, const):
        val = gmpy.powmod(cipher.val, const, self.pk.n_sq)
        return PaillierCiphertext(self, val)

    
    def verify_1OutOf2(self, plaintext1, plaintext2, pk, cipher, proof, voterId):
        """
        Verify ZKP for one out of two possible plaintexts

        plaintext1 : int
            Possible Plaintext
        plaintext2 : int
            Possible Plaintext
        pk : PublicPaillierKey
            Public key of encryption
        cipher : PaillierCiphertext
        proof : Array
            Proof to be checked       
        voterId : String
            To reproduce challenge    
        """
        a_1 = gmpy.mpz(proof[0])
        e_1 = gmpy.mpz(proof[1])
        z_1 = gmpy.mpz(proof[2])
        a_2 = gmpy.mpz(proof[3])
        e_2 = gmpy.mpz(proof[4])
        z_2 = gmpy.mpz(proof[5])
        
        #g**i = (1 + n_publicKey)**firstPlaintext mod n_publicKey**2
        gi_first = gmpy.powmod((1 + pk.n), plaintext1, pk.n_sq)
        #Inverse of g**i because C/g**i mod n_publicKey**2 = C * (g**i)**-1
        result1_euclid = gmpy.invert(gi_first, pk.n_sq)
        #u_1 = = C * (g**i)**-1
        u_1 = (cipher.val * result1_euclid) % pk.n_sq
        
        #g**i = (1 + n_publicKey)**secondPlaintext mod n_publicKey**2
        gi_second = gmpy.powmod((1 + pk.n), plaintext2, pk.n_sq)
        #Inverse of g**i because C/g**i mod n_publicKey**2 = C * (g**i)**-1
        result2_euclid = gmpy.invert(gi_second, pk.n_sq)
        #u_2 = C/g**i mod n_publicKey**2 = C * (g**i)**-1
        u_2 = (cipher.val * result2_euclid) % pk.n_sq

        #z_1**n_publicKey = z_1**n_publicKey mod n_publicKey**2
        z_1n = gmpy.powmod(z_1, pk.n, pk.n_sq)
        #z_2**n_publicKey = z_2**n_publicKey mod n_publicKey**2
        z_2n = gmpy.powmod(z_2, pk.n, pk.n_sq)

        #a_1 * u_1**e_1 mod n_publicKey**2
        z_1_test = ((a_1 % pk.n_sq) *  gmpy.powmod(u_1, e_1, pk.n_sq)) % pk.n_sq
        #a_2 * u_2**e_2 mod n_publicKey**2
        z_2_test = ((a_2 % pk.n_sq) *  gmpy.powmod(u_2, e_2, pk.n_sq)) % pk.n_sq

        #voterId in binary
        voterId_byte =  ' '.join(format(ord(x), 'b') for x in voterId)
        voterId_byte = gmpy.mpz(voterId_byte)
        
        #hash
        h = hashlib.sha512(str(pk.n ^ u_1 ^ u_2 ^ a_1 ^ a_2 ^ voterId_byte).encode("utf-8"))
        s_hex = h.hexdigest()
       
        s = int(s_hex, 16)
        
        #if s == e_1 + e_2 and 
        #z_1**n_publicKey == a_1 * u_1**e_1 mod n_publicKey**2 and 
        #z_2**n_publicKey == a_2 * u_2**e_2 mod n_publicKey**2
        if((s == e_1 + e_2) and (z_1n == z_1_test) and (z_2n == z_2_test)):
            return True
        else:
            return False    

    
    def verify_Range(self, lowerBound, upperBound, proofs, ciphers, trashValues, pk, voterId): 
        """
        Verify ZKP that sum of 0 or 1 ciphers are in a given interval

        lowerBound : int
            Lower bound of sum
        upperBound : int
            Upper bound of sum
        proofs : Array(String)
            One out of two proofs of trash-values, last for sum  
        ciphers : Array(PaillierCiphertext)
        trashValues : Array(PaillierCiphertext)
            trash ciphers
        pk : PublicPaillierKey
            Public key of encryption    
        voterId : String
            To reproduce challenge    
        """
        if(len(trashValues) != (upperBound - lowerBound)):
            return False
        #ZKP for each Trashvalue
        for i in range(0, (upperBound - lowerBound)):
            if not (self.verify_1OutOf2(1,0,pk,trashValues[i],proofs[i], voterId)):
                return False
        #sum up ciphers + trash-values
        encryptedSum = gmpy.mpz(1)
        for i in range(0, (len(ciphers))):
            encryptedSum = (encryptedSum * ciphers[i].val)  % pk.n_sq
        
        for i in range(0, (len(trashValues))):
            encryptedSum = (encryptedSum * trashValues[i].val)  % pk.n_sq

        encryptedSum = self.init_cipher(encryptedSum)
        
        #ZKP that sum == upperBound
        return self.verify_1OutOf2(upperBound + 1,
        upperBound, 
        pk,
        encryptedSum,
        proofs[len(proofs) -1],
        voterId
        )

    @classmethod
    def deserialize(cls, serialized):
        pk = PublicPaillierKey.deserialize(serialized['pk'])
        sk = PrivateKeyShare.deserialize(serialized['sk'], pk) if serialized['sk'] is not None else None
        prot_suite = ProtocolSuite.deserialize(serialized['prot_suite'])()
        abb = cls(prot_suite, pk, int(serialized['num_shares']), int(serialized['threshold']), sk)
        prot_suite.set_abb(abb)
        return abb


class PaillierCiphertext(Ciphertext):

    pass


class PublicPaillierKey():

    def __init__(self, n):
        self.n = n
        self.n_sq = n * n
        self.g = n + 1
        self.bits = gmpy.mpz(gmpy.rint_round(gmpy.log2(self.n)))

    def serialize(self, full=False):
        data = {'n': str(self.n)}
        if full:
            data['n_sq'] = str(self.n_sq)
        return data

    @classmethod
    def deserialize(self, s):
        return PublicPaillierKey(gmpy.mpz(s['n']))


class PrivateKeyShare():

    def __init__(self, key_share, index, n_shares, threshold, pk):
        self.key_share = key_share
        self.index = index
        self.n_shares = n_shares
        self.threshold = threshold
        self.n = pk.n
        self.n_sq = pk.n_sq
        self._precompute_data()

    def _precompute_data(self):
        self.precomputed_fac = gmpy.fac(self.n_shares)
        self.precomputed_plaintext_share_factor = (
            self.precomputed_fac) * self.key_share
        self.precomputed_reconstruction_factor = gmpy.invert(
            (self.precomputed_fac ** 2), self.n)

    def compute_plaintext_share(self, ciphertext):
        if not isinstance(ciphertext, PaillierCiphertext):
            raise ValueError('{} is not a Paillier ciphertext (it is of type {}).'.format(
                ciphertext, type(ciphertext)))
        return (self.index, ciphertext * self.precomputed_plaintext_share_factor)

    def reconstruct_plaintext(self, shares):
        if not len(shares) == self.threshold:
            raise('Num of shares does not match threshold.')
        # combine shares
        c_comb = self.share_combining(shares)
        # find exponent
        i = (c_comb.val - 1) // self.n
        # decrypt
        return (i * self.precomputed_reconstruction_factor) % self.n

    def share_combining(self, shares):
        """ this function combines threshold many shares. It needs exactly threshold many shares """
        mod_shares = [share[1] * (calc_lambda(shares,
                                                  share[0], self.precomputed_fac)) for i, share in enumerate(shares)]
        return sum(mod_shares)

    def serialize(self, full=False):
        data = {'key_share': str(self.key_share), 'index': str(self.index), 'n_shares': str(self.n_shares), 'threshold': str(self.threshold)}
        if full:
            data['precomputed_plaintext_share_factor'] = str(self.precomputed_plaintext_share_factor)
            data['precomputed_reconstruction_factor'] = str(self.precomputed_reconstruction_factor)
        return data

    @classmethod
    def deserialize(self, l, pk):
        return PrivateKeyShare(gmpy.mpz(l['key_share']), int(l['index']), int(l['n_shares']), int(l['threshold']), pk)


def shared_decryption(sk_shares, ciphertext):
    """ Decrypts the given ciphertext with the given shares. It need at least as many shares as the threshold. 
    If it gets more shares, if will select shares at random """
    if len(sk_shares) < 1:
        raise('No shares given')
    used_shares = get_correct_num_of_shares(sk_shares)
    plaintext_shares = [sk_share.compute_plaintext_share(
        ciphertext) for sk_share in used_shares]
    lambdas = [calc_lambda(plaintext_shares, share[0], sk_shares[0].precomputed_fac) for i, share in enumerate(plaintext_shares)]
    return sk_shares[0].reconstruct_plaintext(plaintext_shares), plaintext_shares, lambdas


def get_correct_num_of_shares(shares):
    """ It need at least as many shares as the threshold. 
    If it gets more shares, if will select shares at random"""
    threshold = shares[0].threshold
    if len(shares) < threshold:
        raise('Not enough shares for decryption.')
    used_shares = [sk_share for sk_share in shares]
    # remove unnecessary shares
    while len(used_shares) > threshold:
        # remove share at random
        used_shares.remove(choice(used_shares))
    return used_shares

   