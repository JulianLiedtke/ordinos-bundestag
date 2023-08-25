from locale import currency
from src.crypto.abb import ABB
from src.crypto.paillier_abb import PaillierCiphertext
import numpy as np
import math as m
import logging
log = logging.getLogger(__name__)

class Fraction():

    def __init__(self, numerator, denominator, max_numerator = None, max_denominator = None):
        """
        Implemented is only the special case where the numerator is a Paillier Ciphertext and the denominator an int.
        """
        self.numerator = numerator
        self.denominator = denominator
        if max_numerator == None:
            if isinstance(numerator, PaillierCiphertext):
                raise ValueError("Define max numerator")
            else:
                self.max_numerator = numerator
        else: 
            self.max_numerator = max_numerator

        if max_denominator == None:
            if isinstance(denominator, PaillierCiphertext):
                raise ValueError("Define max denominator")
            else:
                self.max_denominator = denominator
        else: 
            self.max_denominator = max_denominator

    def str_dec(self, abb: ABB):
        if isinstance(self.denominator, PaillierCiphertext):
            denominator = abb.dec(self.denominator)
        elif isinstance(self.denominator, int):
            denominator = self.denominator
        return str(abb.dec(self.numerator)) + "/" + str(denominator)

    def str_dec_all(self, abb: ABB):
        if isinstance(self.denominator, PaillierCiphertext):
            denominator = abb.dec(self.denominator)
        elif isinstance(self.denominator, int):
            denominator = self.denominator
        return str(abb.dec(self.numerator)) + "/" + str(denominator) + "<=" + str(self.max_numerator) + "/" + str(self.max_denominator)
        


class FractionABB():
    """
        Implemented is only the special case where the numerator is a Paillier Ciphertext and the denominator an int.
    """
    def __init__(self, abb: ABB):
        self.abb = abb

    def gt(self, a: Fraction, b: Fraction):
      # TODO: simplify if denominator is instance of int
        biggest_number = max(a.max_numerator * b.max_denominator, a.max_denominator * b.max_numerator)
        bits = self.abb.get_bits_for_size(biggest_number)
        new_a = self.abb.eval_mul_protocol(a.numerator, b.denominator)
        new_b = self.abb.eval_mul_protocol(b.numerator, a.denominator)
        gt = self.abb.gt(new_a, new_b, bits)
        return gt
    
    def eq(self, a: Fraction, b: Fraction):
        # TODO: simplify if denominator is instance of int
        biggest_number = max(a.max_numerator * b.max_denominator, a.max_denominator * b.max_numerator)
        bits = self.abb.get_bits_for_size(biggest_number)
        new_a = self.abb.eval_mul_protocol(a.numerator, b.denominator)
        new_b = self.abb.eval_mul_protocol(b.numerator, a.denominator)
        eq = self.abb.eq(new_a, new_b, bits)
        return eq

    def add(self, a: Fraction, b: Fraction):
        if isinstance(a.denominator, int) and isinstance(a.denominator, int):
            if a == b:
                numerator = self.abb.eval_add_protocol(a.numerator, b.numerator)
                return Fraction(numerator, a.denominator, a.max_numerator + b.max_numerator, a.max_denominator)
            elif m.gcd(a.denominator, b.denominator) > 1:
                denominator = m.lcm(a.denominator, b.denominator)
                numerator_a = self.abb.eval_mul_protocol(a.numerator, int(denominator/a.denominator))
                numerator_b = self.abb.eval_mul_protocol(b.numerator, int(denominator/b.denominator))
                numerator = self.abb.eval_add_protocol(numerator_a, numerator_b)
                max_numerator = a.max_numerator * int(denominator/a.denominator) + b.max_numerator * int(denominator/b.denominator)
                return Fraction(numerator, denominator, max_numerator, denominator)
        new_a = self.abb.eval_mul_protocol(a.numerator, b.denominator)
        new_b = self.abb.eval_mul_protocol(b.numerator, a.denominator)
        numerator = self.abb.eval_add_protocol(new_a, new_b)
        denominator = self.abb.eval_mul_protocol(a.denominator, b.denominator)
        max_numerator = a.max_numerator * b.max_denominator + b.max_numerator * a.max_denominator
        max_denominator = a.max_denominator * b.max_denominator
        return Fraction(numerator, denominator, max_numerator, max_denominator)
    
    def mul_cipher(self, a: PaillierCiphertext, b: Fraction, max_cipher: int):
        numerator = self.abb.eval_mul_protocol(a, b.numerator)
        denominator = b.denominator
        max_numerator = max_cipher * b.max_numerator
        return Fraction(numerator, denominator, max_numerator, b.max_denominator)

    def mul_fractions(self, a: Fraction, b: Fraction):
        numerator = self.abb.eval_mul_protocol(a.numerator, b.numerator)
        denominator = a.denominator * b.denominator
        max_numerator = a.max_numerator * b.max_numerator
        max_denominator = a.max_denominator * b.max_denominator
        return Fraction(numerator, denominator, max_numerator, max_denominator)

    
    def negate(self, boolean):
        return self.abb.eval_sub_protocol(1, boolean)

    """
    Special methods for the SL-Evaluation --> uses constraints on fractions to simplify
    """
    def select_fraction(self, possible_fractions, indices, max_numerator, max_denominator, max_column):
        """
        Indices is an array containing zeros and exactly one 1.
        Returns the fraction on the position of the one
        """
        numerator = self.abb.enc_zero
        denominator = self.abb.enc_zero
        for i in range(max_column):
            numerator = self.abb.eval_add_protocol(numerator, self.abb.eval_mul_protocol(indices[i], possible_fractions[i].numerator))
            denominator = self.abb.eval_add_protocol(denominator, self.abb.eval_mul_protocol(indices[i], possible_fractions[i].denominator))
        return Fraction(numerator, denominator, max_numerator, max_denominator)



    def get_max_as_index(self, values: list):
        """
        Returns a list of indices with index[i] = 1 if i is the maximum element
        """
        indices = [self.abb.enc_zero for i in range(len(values))]
        indices[0] = self.abb.enc_one
        current_max = values[0]

        for i in range(1, len(values)):
            gt = self.gt(values[i], current_max)
            # update indices
            indices[i] = gt
            for j in range(i):
                indices[j] = self.abb.eval_mul_protocol(indices[j], self.negate(gt))
            # update current maximum
            current_max = self.index_mult([gt, self.negate(gt)], [values[i], current_max], 2)
        return indices
    
    def gt_with_ranks(self, value1, value2, rank1, rank2, n_parties):
        biggest_number = max(value1.max_numerator * value2.max_denominator, value1.max_denominator * value2.max_numerator)
        biggest_number = biggest_number * (n_parties+1) - 1
        bits = self.abb.get_bits_for_size(biggest_number)
        new_value1 = self.abb.eval_mul_protocol(value1.numerator, value2.denominator)
        new_value2 = self.abb.eval_mul_protocol(value2.numerator, value1.denominator)
        # insert tie-breaking
        new_value1 = self.abb.eval_mul_protocol(new_value1, n_parties) + rank1
        new_value2 = self.abb.eval_mul_protocol(new_value2, n_parties) + rank2
        gt = self.abb.gt(new_value1, new_value2, bits)
        return gt
    
    
    def get_min_pairwise_tie_as_index(self, values: list, ranks):
        indices = [self.abb.enc_zero for i in range(len(values))]
        indices[0] = self.abb.enc_one
        current_min = values[0]
        current_rank = ranks[0]

        for i in range(1, len(values)):
            lq = self.gt_with_ranks(current_min, values[i], current_rank, ranks[i], len(values))
            gt = self.abb.eval_sub_protocol(1, lq)
            # update indices
            indices[i] = lq
            for j in range(i):
                indices[j] = self.abb.eval_mul_protocol(indices[j], gt)
            # update current maximum and rank
            current_min = self.index_mult([lq, gt], [values[i], current_min], 2)
            current_rank = lq * ranks[i] + gt * current_rank
        return indices

    def get_max_pairwise_tie_as_index(self, values: list, ranks):
        indices = [self.abb.enc_zero for i in range(len(values))]
        indices[0] = self.abb.enc_one
        current_max = values[0]
        current_rank = ranks[0]

        for i in range(1, len(values)):
            gt = self.gt_with_ranks(values[i], current_max, ranks[i], current_rank, len(values))
            
            # update indices
            indices[i] = gt
            for j in range(i):
                indices[j] = self.abb.eval_mul_protocol(indices[j], self.negate(gt))
            # update current maximum and rank
            n_gt = self.negate(gt)
            current_max = self.index_mult([gt, self.negate(gt)], [values[i], current_max], 2)
            current_rank = gt * ranks[i] + n_gt * current_rank
        return indices

    def get_max_as_index_equality_version(self, values: list, ranks):
        n = len(values)
        all_max = self.get_all_max_as_index(values)
        mod_indices = [all_max[i] * ranks[i] for i in range(n)]
        return self.get_integer_max_as_index(mod_indices, self.abb.get_bits_for_size(n-1))
 
    

    def get_integer_max_as_index(self, values: list, bits):
        """
        Returns a list of indices with index[i] = 1 if i is the maximum element
        """
        indices = [self.abb.enc_zero for i in range(len(values))]
        indices[0] = self.abb.enc_one
        current_max = values[0]
        for i in range(1, len(values)):
            gt = self.abb.gt(values[i], current_max, bits)
            
            # update indices
            indices[i] = gt
            for j in range(i):
                indices[j] = self.abb.eval_mul_protocol(indices[j], self.negate(gt))
            current_max = gt * values[i] + self.negate(gt) * current_max
        return indices
        


    def get_min_as_index(self, values: list):
        """
        Returns a list of indices with index[i] = 1 if i is the minimum element
        """
        indices = [self.abb.enc_zero for i in range(len(values))]
        indices[0] = self.abb.enc_one
        current_min = values[0]

        for i in range(1, len(values)):
            lq = self.gt(current_min, values[i])
            # update indices
            indices[i] = lq
            for j in range(i):
                indices[j] = self.abb.eval_mul_protocol(indices[j], self.negate(lq))
            # update current minimum
            current_min = self.index_mult([lq, self.negate(lq)], [values[i], current_min], 2)
        return indices

    def get_nonzero_min_as_index(self, values: list):
        """
        Returns a list of indices with index[i] = 1 if i is the minimum element
        """
        indices = [self.abb.enc_zero for i in range(len(values))]
        indices[0] = self.abb.enc_one
        current_min = values[0]
        max_numerator = 0
        for v in values:
            z = v.max_numerator
            if z >= max_numerator:
                max_numerator = z
        is_zero = [self.abb.eq(v.numerator, self.abb.enc_zero, self.abb.get_bits_for_size(max_numerator)) for v in values]
        all_zero = [None for v in values]
        for i in range(len(values)):
            if(i==0):
                all_zero[i] = is_zero[i]
            else:
                all_zero[i] = is_zero[i] * all_zero[i-1]

        for i in range(1, len(values)):
            lq = self.gt(current_min, values[i])
            # update if lq = 1
            lq = lq * self.negate(is_zero[i]) # dont update if new element is zero
            # update if initial only zeros before
            lq = self.if_then_else(all_zero[i-1], self.abb.enc_one, lq)
            # update indices
            indices[i] = lq
            for j in range(i):
                indices[j] = self.abb.eval_mul_protocol(indices[j], self.negate(lq))
            # update current minimum
            current_min = self.index_mult([lq, self.negate(lq)], [values[i], current_min], 2)
        return indices

    def if_then_else(self, cond, true_val, false_val):
        neg_cond = self.abb.eval_sub_protocol(1, cond)
        prod1 = self.abb.eval_mul_protocol(cond, true_val)
        prod2 = self.abb.eval_mul_protocol(neg_cond, false_val)
        return self.abb.eval_add_protocol(prod1, prod2)

    def index_mult(self, index, list, max_iter): 
        """
        Returns index[0] * list[0] + ... + index[n] * list[n] where list contains fractions and exactly one index is 1
        """
        numerator = self.abb.enc_zero
        denominator = self.abb.enc_zero
        max_numerator = 1
        max_denominator = 1
        for i in range(max_iter):
            numerator = self.abb.eval_add_protocol(numerator, self.abb.eval_mul_protocol(index[i], list[i].numerator))
            denominator = self.abb.eval_add_protocol(denominator, self.abb.eval_mul_protocol(index[i], list[i].denominator))
            max_numerator = max(max_numerator, list[i].max_numerator)
            max_denominator = max(max_denominator, list[i].max_denominator)
        return Fraction(numerator, denominator, max_numerator, max_denominator)

    

    def get_all_max_as_index(self, values: list):
        """
        Returns a list of indices with index[i] = 1 if i is the maximum element
        """
        indices = [self.abb.enc_zero for i in values]
        indices[0] = self.abb.enc_one
        current_max = values[0]

        for i in range(1, len(values)):
            gt = self.gt(values[i], current_max)
            eq = self.eq(values[i], current_max)
            # update indices
            indices[i] = gt
            factor = self.abb.eval_add_protocol(self.negate(gt), eq)
            for j in range(i):
                indices[j] = self.abb.eval_mul_protocol(indices[j], factor)
            # update current maximum
            current_max = self.index_mult([gt, self.negate(gt)], [values[i], current_max], 2)
        return indices

        
