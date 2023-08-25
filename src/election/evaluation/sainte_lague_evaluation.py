import logging
import numpy as np
import math as m
import gmpy2 as gmpy
from src.crypto.abb import ABB
from src.crypto.paillier_abb import PaillierCiphertext
from src.election.tie_breaking.tie_breaking import TieBreaking
from src.util.fraction import Fraction, FractionABB
from src.util.csv_writer import CSV_Writer
from time import time
import threading as threading


log = logging.getLogger(__name__)

class SainteLagueEvaluation():
    
    def __init__(self, abb: ABB):
        self.id = threading.get_ident()
        self.abb = abb
        self.f_abb = FractionABB(self.abb)


    def highest_divisors_unencrypted(self, n_seats : int, population_distribution):
        n_parties = len(population_distribution)
        höchstzahlen = np.zeros((n_parties, n_seats))
        for i in range(n_parties):
            for j in range(n_seats):
                höchstzahlen[i][j] = population_distribution[i] / float(2*j + 1)
        pointer = [0 for i in range(n_parties)]
        relevant_höchstzahlen = [0 for i in range(n_parties)]
        seats = np.zeros(n_parties, dtype=int)
        for s in range(n_seats):
            for j in range(n_parties):
                relevant_höchstzahlen[j] = höchstzahlen[j][pointer[j]]
            max_index = relevant_höchstzahlen.index(max(relevant_höchstzahlen))
            pointer[max_index] = pointer[max_index] + 1
            seats[max_index] = seats[max_index] + 1
        return seats
    
    def get_seat_distribution(self, n_seats: int, vote_distribution: list, max_votes: int, tie_breaking_iterator: TieBreaking, text = "None"):
        """
        Optimized version of the Sainte-Lague method of highest divisors.
        This method is called "baseline quotient method" in the paper
        """ 
        f_abb = FractionABB(self.abb)
        n_parties = len(vote_distribution)
        current_elements = np.empty(n_parties, dtype=object)

        for p in range(n_parties):
            current_elements[p] = Fraction(vote_distribution[p], 1, max_votes, 1)
        
        seats_per_party = [self.abb.enc_zero for p in range(n_parties)]
       
        # calculate seats
        normal_seats = max(n_seats - n_parties + 1, 0)
       
        for s in range(normal_seats):
            start_time = time()
            current_elements, seats_per_party = self.seat_iteration(f_abb, s, n_parties, current_elements, seats_per_party)
            eval_time = time() - start_time
            CSV_Writer.write_times_benchmarks(eval_time, "normal iteration for seat " + str(s) + "in step " + str(text))
        for s in range(max(n_seats - n_parties + 1, 0), n_seats):
            if tie_breaking_iterator.has_next(n_parties):
                start_time = time()
                ranks = tie_breaking_iterator.next(n_parties)
                current_elements, seats_per_party = self.tie_breaking_seat_iteration(f_abb, s, n_parties, current_elements, seats_per_party, ranks)
                eval_time = time() - start_time
                CSV_Writer.write_times_benchmarks(eval_time, "tie breaking iteration for seat " + str(s) + "in step " + str(text))
            else:
                raise IndexError("Not enough ranks.")
        
        return seats_per_party

    def get_seat_distribution_enc_seats(self, enc_seats, n_seats: int, vote_distribution: list, max_votes: int, tie_breaking_iterator: TieBreaking, text = "None"):
        """
        Optimized version of the Sainte-Lague method of highest divisors.
        This method is called "baseline quotient method" in the paper
        """ 
        f_abb = FractionABB(self.abb)
        n_parties = len(vote_distribution)
        current_elements = np.empty(n_parties, dtype=object)

        for p in range(n_parties):
            current_elements[p] = Fraction(vote_distribution[p], 1, max_votes, 1)
        
        seats_per_party = [self.abb.enc_zero for p in range(n_parties)]
       
        index_array = self.get_index_array(enc_seats, n_seats)       
        
        for s in range(n_seats):
            if tie_breaking_iterator.has_next(n_parties):
                seats_per_party_old = seats_per_party.copy()
                start_time = time()
                ranks = tie_breaking_iterator.next(n_parties)
                current_elements, seats_per_party_pre = self.tie_breaking_seat_iteration(f_abb, s, n_parties, current_elements, seats_per_party, ranks)
                # update seats per party only when necessary
                for i in range(len(seats_per_party)):
                    seats_per_party[i] = self.if_then_else(index_array[s], seats_per_party_pre[i], seats_per_party_old[i])
        
                eval_time = time() - start_time
                CSV_Writer.write_times_benchmarks(eval_time, "tie breaking iteration for seat " + str(s) + "in step " + str(text))
            else:
                raise IndexError("Not enough ranks.")
        
        return seats_per_party
    
    def seat_iteration(self, f_abb: FractionABB, current_seat, n_parties, current_element, seats_per_party):
        log.info("seat iteration")
        s = current_seat
        
        # get maximum from relevant elements
        index_max = f_abb.get_max_as_index(current_element)
        
        # update current_element
        for p in range(n_parties):
            cur = current_element[p]
            add_to_denominator = self.abb.eval_add_protocol(index_max[p], index_max[p])
            new_denominator = self.abb.eval_add_protocol(cur.denominator, add_to_denominator)
            current_element[p] = Fraction(cur.numerator, new_denominator, cur.max_numerator, cur.max_denominator + 2)

        # update seats
        for p in range(n_parties):
            seats_per_party[p] = self.abb.eval_add_protocol(seats_per_party[p], index_max[p])
        
        return current_element, seats_per_party

    def tie_breaking_seat_iteration(self, f_abb: FractionABB, n_parties, current_element, seats_per_party, ranks, returnIndex = False):
        
        index_max = f_abb.get_max_pairwise_tie_as_index(current_element, ranks)
        # update current_element
        for p in range(n_parties):
            cur = current_element[p]
            add_to_denominator = self.abb.eval_add_protocol(index_max[p], index_max[p])
            new_denominator = self.abb.eval_add_protocol(cur.denominator, add_to_denominator)
            current_element[p] = Fraction(cur.numerator, new_denominator, cur.max_numerator, cur.max_denominator + 2)

        # update seats
        for p in range(n_parties):
            seats_per_party[p] = self.abb.eval_add_protocol(seats_per_party[p], index_max[p])
        if returnIndex:
            return current_element, seats_per_party, index_max
        else: 
            return current_element, seats_per_party


    def move_pointer(self, pointer, is_max):
        update = is_max
        sum = self.abb.enc_zero
        for i in range(len(pointer)):
            new_sum = sum + pointer[i]
            pointer[i] = sum * update
            if not i == 0:
                update = update * self.negate(pointer[i-1])
            sum = new_sum
        return pointer
    

    

    def get_seat_distribution_many_seats(self, n_seats: int, vote_distribution: list, n_votes, max_votes: int, tie_breaking_iterator: TieBreaking, text = None):
        """
        sainte-lague based on floor division
        """
        start_time = time()
        max_seats = n_seats
        n_parties = len(vote_distribution)
        preliminary_seats = self.compute_seats_secret(n_seats, vote_distribution, n_votes, max_votes, n_parties, max_seats)
        current_num_seats = self.abb.enc_zero
        for i in range(n_parties):
            current_num_seats += preliminary_seats[i]
        f_abb = FractionABB(self.abb)
        end_time = time()
        CSV_Writer.write_times_benchmarks(self.id, "improved sainte lague, predistribution for " + str(text), end_time - start_time)
        # part 1: ascending
        start_time = time()
        index_last_updated_party = [self.abb.enc_zero for i in range(n_parties)]
        seats_per_party = preliminary_seats.copy()
        current_elements = np.empty(n_parties, dtype=object)

        for p in range(n_parties):
            current_elements[p] = Fraction(vote_distribution[p], (seats_per_party[p] + seats_per_party[p]) + self.abb.enc_one, max_votes, 2 * max_seats + 1)
        initial_elements = current_elements.copy()
        max_residual_seats = n_parties 
        index_array = self.get_index_array(n_seats - current_num_seats, max_residual_seats)
        
        for s in range(max_residual_seats):
            seats_per_party_old = seats_per_party.copy()
            ranks_parties = tie_breaking_iterator.next(n_parties)
            current_elements, seats_per_party_pre, last_index_pre = self.tie_breaking_seat_iteration(f_abb, n_parties, current_elements, seats_per_party, ranks_parties, returnIndex=True)
            
            # update seats per party only when necessary
            for i in range(len(seats_per_party)):
                seats_per_party[i] = self.if_then_else(index_array[s], seats_per_party_pre[i], seats_per_party_old[i])
                index_last_updated_party[i] = self.if_then_else(index_array[s], last_index_pre[i], index_last_updated_party[i])
        
        seats_per_party_ascending = seats_per_party.copy()

        # test if seats per party are correct
        last_inserted_elements = [Fraction(f.numerator, self.if_then_else(self.abb.eq(f.denominator, self.abb.enc_one, self.abb.get_bits_for_size(f.max_denominator)), self.abb.enc_zero, f.denominator - self.abb.enc_no_r(2)), f.max_numerator, f.max_denominator )for f in initial_elements]
        min_test_index = f_abb.get_min_as_index(last_inserted_elements)
        min_test_element = f_abb.index_mult(min_test_index, last_inserted_elements, len(min_test_index))
        next_elements = current_elements.copy()
        next_elements = list(next_elements)
        next_elements.append(min_test_element)
        max_next_element = f_abb.get_max_pairwise_tie_as_index(next_elements, tie_breaking_iterator.next(len(next_elements)))
        ascending_valid = max_next_element[len(max_next_element)-1]
        
        end_time = time()
        CSV_Writer.write_times_benchmarks(self.id, "improved sainte lague, understimation for: " + str(text), end_time - start_time)


        # part 2: descending
        start_time = time()
        seats_per_party = preliminary_seats.copy()
        current_elements = np.empty(n_parties, dtype=object)

        for p in range(n_parties):
            current_elements[p] = Fraction(vote_distribution[p], (seats_per_party[p] + seats_per_party[p]) + self.abb.enc_one, max_votes, 2 * max_seats + 1)
        
        for p in range(n_parties):
            seats_per_party[p] = seats_per_party[p] + self.abb.enc_one

        max_residual_seats = n_parties 
        index_array = self.get_index_array(current_num_seats + n_parties - n_seats, max_residual_seats)
        

        for s in range(max_residual_seats):
            seats_per_party_old = seats_per_party.copy()
            ranks_parties = tie_breaking_iterator.next(n_parties)
            current_elements, seats_per_party_pre = self.reverse_tie_breaking_seat_iteration(f_abb, n_parties, current_elements, seats_per_party, ranks_parties)
            
            # update seats per party only when necessary
            for i in range(len(seats_per_party)):
                seats_per_party[i] = self.if_then_else(index_array[s], seats_per_party_pre[i], seats_per_party_old[i])
        
        seats_per_party_descending = seats_per_party.copy()
        final_seats = []
        for i in range(n_parties):
            seats = self.if_then_else(ascending_valid, seats_per_party_ascending[i], seats_per_party_descending[i])
            final_seats.append(seats)
        end_time = time()

        CSV_Writer.write_times_benchmarks(self.id, "improved sainte lague, overestimation for: " + str(text), end_time - start_time)
        return final_seats


    def reverse_tie_breaking_seat_iteration(self, f_abb: FractionABB, n_parties, current_element, seats_per_party, ranks, min_two_seats= False):
        """
        min_one_seat = True if it is garanteed, that every party gets at least two seats
        """
        
        # get minimum from relevant elements
        index_max = f_abb.get_min_pairwise_tie_as_index(current_element, ranks)
        
        # update current_element
        for p in range(n_parties):
            cur = current_element[p]
            sub_from_denominator = self.abb.eval_add_protocol(index_max[p], index_max[p])
            
            if min_two_seats:
                new_denominator = self.abb.eval_sub_protocol(cur.denominator, sub_from_denominator)
            else:
                # v_i/1 cannot be the smallest element, else negative seats
                # --> replace v_i/1 with v_i/0
                isOne = self.abb.eq(cur.denominator, self.abb.enc_one, self.abb.get_bits_for_size(cur.max_denominator))
                new_denominator = self.if_then_else(isOne, self.abb.enc_zero, cur.denominator - sub_from_denominator)

            current_element[p] = Fraction(cur.numerator, new_denominator, cur.max_numerator, cur.max_denominator)

            
        # update seats
        for p in range(n_parties): 
            seats_per_party[p] = self.abb.eval_sub_protocol(seats_per_party[p], index_max[p])
        

        return current_element, seats_per_party

    def compute_seats_secret(self, num_seats, votes, current_sum_votes, sum_votes, n_parties, max_seats):
        """
        modified Hare-Niemeyer --> partly copied from seat_distribution.py
        The alternative to the binary search. Needed for the secret version.
        :return: the encrypted result (number of seats)
        """
        if isinstance(num_seats, int):
            num_seats = self.abb.enc_no_r(num_seats)
        seats_per_party = [None for i in range(n_parties)]
        for i in range(n_parties):
            start_time = time()
            product = self.abb.eval_mul_protocol(votes[i],  num_seats)
            bits = self.abb.get_bits_for_size(sum_votes * max_seats)
            # here floor division(product, sum_votes, num_seats)
            seats_counter = self.floor_division(product, current_sum_votes, max_seats, bits)
            seats_per_party[i] = seats_counter
            end_time = time()
            CSV_Writer.write_times_benchmarks(self.id, "initial distribution for party : " + str(i), end_time - start_time)
            
        return seats_per_party

    def sl_known_divisor(self, abb: ABB, vote_distribution: list, max_votes: int, divisor: PaillierCiphertext, min_claim: list = None, max_claim: int = None):
        """
        Calculate a sainte-lague seat distribution using the method of highest numbers with an unknown number of seats to distribute
        and returns the unencrypted (!) result.
        vote_distribution: array with the votes per party encrypted as Pallier ciphertextes
        max_votes: maximum number of votes one party can have 
        divisor: a possible divisor
        min_claim: list of minimal seats per party
        return: array of Plaintexts representing the number of allocated seats for each party
        """
        n_parties = len(vote_distribution)
        n_seats = [0 for i in range(n_parties)]
        
        for i in range(n_parties):
            start_time = time()
            doubled_votes = abb.eval_add_protocol(vote_distribution[i], vote_distribution[i])

            # first approximation: find out in which hundred the number of seats the party gets is 
            hundred_seats = 0
            isLess = True
            while(isLess):
                isLess = self.test_seats(hundred_seats, doubled_votes, abb, max_votes, divisor, min_claim[i], max_claim)
                if isLess:
                    hundred_seats += 100
            
            # approximation within the given hundred with binary search
            lower_bound = hundred_seats - 100
            upper_bound = hundred_seats
            while(lower_bound < upper_bound):
                test_seats = int(lower_bound + 1/2 *(upper_bound - lower_bound))
                isLess = self.test_seats(test_seats, doubled_votes, abb, max_votes, divisor, min_claim[i], max_claim)
                if isLess:
                    lower_bound = test_seats
                else: 
                    upper_bound = test_seats
                if lower_bound + 1 == upper_bound:
                    break    

            n_seats[i] = lower_bound
            end_time = time()
            CSV_Writer.write_times_benchmarks(self.id, "2. top distribution for party: " + str(i), end_time - start_time)
        return n_seats

    def test_seats(self, seats: int, doubled_votes: PaillierCiphertext, abb: ABB, max_votes: int, divisor: PaillierCiphertext, min_claim_party: PaillierCiphertext, max_claim = int):
        """
        Helper method for the optimised höchstzahlverfahren with a given divisor.
        Tests if a given number of seats is correct.
        Returns True if the party deserves at least the given number of seats and false if the party deserves less seats.
        """
        if min_claim_party == None:
            has_min_claim = False
        else:
            has_min_claim = True
        
        # basically one iteration of the unoptimized höchstzahlverfahren
        x = 2 * seats - 1
        bits = abb.get_bits_for_size(max(2 * max_votes, max_votes * x))
        comp = abb.eval_mul_protocol(divisor, x)
        has_seat_divisor = abb.gt(doubled_votes, comp, bits)
        if has_min_claim:
            bits = abb.get_bits_for_size(max(max_claim, seats))
            has_seat_claim = abb.gt(min_claim_party, abb.enc_no_r(seats), bits)
            has_seat_enc = self.enc_or(has_seat_divisor, has_seat_claim)
        else:
            has_seat_enc = has_seat_divisor

        has_seat = abb.dec(has_seat_enc)
        if(has_seat == 1):
            # Party has at least seats seats
            return True
        else:
            # Party has less seats
            return False




    """
    General helper methods
    """
    def if_then_else(self, cond, val1, val2):
        neg_cond = self.negate(cond)
        prod1 = self.abb.eval_mul_protocol(cond, val1)
        prod2 = self.abb.eval_mul_protocol(neg_cond, val2)
        return self.abb.eval_add_protocol(prod1, prod2)

    def negate(self, boolean):
        return self.abb.eval_sub_protocol(1, boolean)
    
    def floor_division(self, a, b, n: int, bits):
        """
        return i = floor(a/b) , where i <= n
        """
        length = n.bit_length()
        bits_array = [2 ** (length - 1 - i) for i in range(length)]
        set_bits = [self.abb.enc_zero for i in range(length)]
        lower = self.abb.enc_zero

        for i in range(length):
            j = lower + bits_array[i]
            prod = self.abb.eval_mul_protocol(j,  b)
            gt = self.abb.gt(a, prod, bits)
            lower += self.abb.eval_mul_protocol(bits_array[i],gt)
        return lower
    
    def get_index_array(self, k, n):
        """
        Returns an array of length n with k one followed by zeros
        """
        array = [None for i in range(n)]
        for i in range(n):
            array[i] = self.abb.eq(k, i+1, self.abb.get_bits_for_size(n))
        val = array[n-1]
        for i in range(n-2, -1, -1):
            val += array[i]
            array[i] = val
        return array
    
    def enc_or(self, cond1, cond2):
        """
        Do not use comparison but addition and multiplication because it's cheaper.
        """
        neg1 = self.negate(cond1)
        neg2 = self.negate(cond2)
        result_neg = self.abb.eval_mul_protocol(neg1, neg2)
        return self.negate(result_neg)