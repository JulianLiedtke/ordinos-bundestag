import logging
import numpy as np
import math as math
from src.crypto.paillier_abb import PaillierCiphertext
import threading as threading


from src.election.evaluation.evaluation_protocol import EvaluationProtocol
from src.election.evaluation.sainte_lague_evaluation import SainteLagueEvaluation
from src.crypto.abb import ABB
from src.election.real_world_elections.helpers import PartiesAcrossConstituencies
from src.election.tie_breaking.tie_breaking import TieBreaking, BreakingTypes
from src.util.csv_writer import CSV_Writer
from src.util.fraction import Fraction, FractionABB
from src.election.real_world_elections.bundestag_data import Data2021
from time import time

log = logging.getLogger(__name__)

class BundestagEvaluation(EvaluationProtocol):
    """
    Evaluation of the election of the german parliament.
    Attention: Does not work for arbitrary values it can initialized with, if these values are fixed in the constituion (especially number of seats, number of constituencies)
    """
    abb = None

    def __init__(self, abb: ABB):
        self.abb = abb

        # variables to save encrypted interim results
        self.primary_votes = None
        self.secondary_votes = None
        self.direct_mandates_constituency = None
        self.direct_mandates_state = None
        self.direct_mandates_party = None
        self.large_parties = None
        self.relevant_secondary_votes = {}
        super().__init__()

    def run(self, votes):
        # new run method for real-world results
        log.info("bundestag evaluation (final version)")
        self.id = threading.get_ident() #Thread-ID

        self.debug = False # set on False for run with Tally-Hiding

        # insert real word data (shortcut, no aggregation)
        self.primary_votes, self.n_parties, self.n_constituencies, self.n_valid_primary_votes, self.all_parties, self.const_state_mapping, self.state_names, self.constituency_names = Data2021.get_primary_votes(self.abb)
        self.parties = self.all_parties.names_parties
        self.secondary_votes, self.n_valid_secondary_votes, self.parties_per_state, self.total_valid_secondary_votes, self.minority_parties, self.population_distribution, self.min_seats_contingent = Data2021.get_secondary_votes(self.abb)
        
        log.info("real-world data inserted")
        
        # Tie Breaking
        # direct mandates
        tie_list = [i for i in range(100)] # in setup phase --> precalculating to much ranks is no problem
        max_ties = 2000
        tie_breaking_iterator = TieBreaking(max_ties, self.abb)
        tie_breaking_iterator.setup_multiple_arrays(self.abb, tie_list, BreakingTypes.random)
        self.tie_breaking_iterator = tie_breaking_iterator
        
        CSV_Writer.init_writer()
        
        self.evaluate_first_votes()
        log.info("first vote evaluated")
        final_seats, min_claim, overhang = self.evaluate_secondary_votes()
        log.info("secondary votes finished")
        final_seats_per_state = self.second_down_distribution(final_seats)
        final_seats_per_state = self.remaining_overhang(final_seats, final_seats_per_state)
        
        #self.map_to_delegates()
        return final_seats_per_state


    def map_to_delegates(self):
        # As there wasn't complete data in the official documentation, which person had which list position on the federal lists of the parties, 
        # we show the last step on the bundestag evaluation by an example where we reconstructed the party list of the federal state north rhine westphalia and the party SPD.
        # However, this information is publicly known in elections.
        # For each candidate who contested for a direct mandate, we figured out of this candidate is also on the party list in their federal state.
        # If yes, we put the position in the array "list position", if not, we wrote "None" instead representing that the candidate didn't chose to also be on the party list but only competet for the first votes.

        # the following data could be extracted from the public data and the evaluation result, but we choose here, to write it down directly as it's only an example
        # public data for federal state nrw and party SPD
        seats = 49
        list_position = [11, 14, 6, 7, 2, 23, None, None, None, 20, None, 5, 4, 15, 18, 21, 16, 29, 8, 22, None, 1, 9, 3, 13, 12, 26, 19, 10, 17, ]

        # data that can be extracted out of the election result (from the party "evaluation of first votes")
        won_constituency_dec = [1,0,1,1,1,1,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1]
        won_constituency = [self.abb.enc_no_r(s) for s in won_constituency_dec]
        

        # encryption of the candidate mapping in a tally-hiding way
        n_delegates = 0
        id_direct = []
        
        # direct mandates who aren't on the party list
        arr = []
        for i in range(len(won_constituency)):
            if list_position[i] is None:
                if int(self.abb.dec(won_constituency[i])) == 1:
                    n_delegates = n_delegates + 1
                    id_direct.append(i)
            else:
                arr.append(list_position[i])
        
        # candidates that are also on the party lsit
        arr.reverse()
        for c in arr:
            if seats - n_delegates < c:
                i = list_position.index(c)
                if int(self.abb.dec(won_constituency[i])) == 1:
                    n_delegates = n_delegates + 1
                    id_direct.append(i)
        n_liste = seats - n_delegates
        
        # remaining part is filled by first list positions
        delegates_id = id_direct + [i+1 for i in range(n_liste)]
        return delegates_id

    
    

    def evaluate_first_votes(self):
        """
        Calculates direct mandates and sum of direct mandates per state.
        """
        start_time = time()
        self.calc_direct_mandates()
        end_time = time()
        CSV_Writer.write_times_benchmarks(self.id, "calculate direct mandates", end_time-start_time)

        start_time = time()
        self.add_direct_mandates()
        end_time = time()
        CSV_Writer.write_times_benchmarks(self.id, "add direct mandates", end_time-start_time)

    def evaluate_secondary_votes(self):
        # 5%-clause
        start_time = time()
        self.get_secondary_votes_per_party()
        end_time = time()
        CSV_Writer.write_times_benchmarks(self.id, "get secondary votes per party", end_time-start_time)
        start_time = time()
        self.get_large_parties()
        end_time = time()
        CSV_Writer.write_times_benchmarks(self.id, "get large parties", end_time-start_time)
        start_time = time()
        all_large_parties = PartiesAcrossConstituencies(len(self.large_parties), self.large_parties, len(self.state_names), self.parties_per_state)
        self.relevant_secondary_votes = all_large_parties.get_empty_general(self.abb)
        for s in range(len(self.state_names)):
            self.relevant_secondary_votes = all_large_parties.add_votes_to_general(self.abb, self.relevant_secondary_votes, s, list(self.secondary_votes[s].values()))

        end_time = time()
        
        CSV_Writer.write_times_benchmarks(self.id, "pool relevant secondary votes", end_time-start_time)
        min_claim, overhang = self.calc_min_seats()
        final_seats_party = self.calculate_total_seats(min_claim, overhang)

        final_seats = final_seats_party
        self.final_seats = final_seats
        return final_seats, min_claim, overhang

    
    
    def evaluate_to_final_result(self):
        delegates = [None for s in self.state_names]
        for s in self.state_names:
            state_delegates = [None for p in self.large_parties]
            for p in self.large_parties:
                state_delegates[p] = self.evaluate_to_final_seats(self.direct_mandates_state[s][self.large_parties[p]], self.final_seats_per_state[s][p])
            delegates[s] = state_delegates
        return delegates


    def evaluate_to_final_seats(self, direct, n_list):
        """
        Evaluates the final result for a state for a party given a list of all possible direct mandates (direct) with indices one if the candidate is a direct mandat and zero else
        and the number of delegates the party has (n_list)
        """
        mandates = direct
        
        for i in range(len(mandates)):
            sum = self.sum_all(mandates)
            gt = self.abb.gt(sum, self.abb.enc_no_r(n_list), self.abb.get_bits_for_size(n_list))
            mandates[i] = self.if_then_else(mandates[i], mandates[i], gt)
        decrypted_mandates = [self.abb.dec(m) for m in mandates]
        #log.info("decrypted: " + str(decrypted_mandates ))
        return decrypted_mandates


    def sum_all(self, list):
        sum = self.abb.enc_zero
        for i in range(len(list)):
            sum = sum + list[i]
        return sum

    def add_direct_mandates(self):
        """
        Adds the direct mandates both by state and by party.
        """
        log.info("add direct mandates")
        
        # direct mandates per state
        direct_mandates_state = {}
        for state in self.state_names:
            direct_mandates_state[state] = self.all_parties.get_empty_general(self.abb)
        for const in self.constituency_names:
            corresponding_state = self.const_state_mapping[const]
            direct_mandates_state[corresponding_state] = self.all_parties.add_votes_to_general(self.abb, direct_mandates_state[corresponding_state], const, self.direct_mandates_constituency[const])
        self.direct_mandates_state = direct_mandates_state  
        # direct mandates per party
        direct_mandates_party = {}
        for party in self.parties:
            direct_mandates_party[party] = self.abb.enc_zero

        for state in self.state_names:
            for party in self.parties:
                direct_mandates_party[party] = self.abb.eval_add_protocol(direct_mandates_party[party], direct_mandates_state[state][party])
        self.direct_mandates_party = direct_mandates_party
        if(self.debug):
            for state in direct_mandates_state:
                mandates_dec = [self.abb.dec(i) for i in list(direct_mandates_state[state].values())]  
                CSV_Writer.write_result_benchmarks(self.id, "direct mandates state " + str(state),str(mandates_dec))
            mandates_dec = [self.abb.dec(i) for i in list(direct_mandates_party.values())]  
            CSV_Writer.write_result_benchmarks(self.id, "direct mandates party",str(mandates_dec))
        


        

    def calc_direct_mandates(self):
        """
        Calculates the winner (= relative majority) of each constituency.
        """
        log.info("calculate direct mandates")
        direct_mandates = {}
        for const in self.primary_votes:
            start_time = time()
            values = self.cand_dict_to_array(self.primary_votes[const])
            rank = self.tie_breaking_iterator.next(len(values))
            direct_mandates[const] = self.get_winner(values, self.n_valid_primary_votes[const], rank, False)
            end_time = time()
            CSV_Writer.write_times_benchmarks(self.id, "first votes " + str(const+1), end_time - start_time)
            if(self.debug):
                indices_dec = [self.abb.dec(i) for i in direct_mandates[const]]
                CSV_Writer.write_result_benchmarks(self.id, "first votes " + str(const+1), str(indices_dec))
        self.direct_mandates_constituency = direct_mandates
        return direct_mandates


        
    def get_winner(self, values, max_element, rank, decrypt = True):
        """
        Get indices, that is one if the element is the winner and zero else, tie braking with rank. Decrypts winner if decrypt = True
        """
        n = len(values)
        # tie_breaking
        for i in range(n):
            values[i] = (values[i] * self.abb.enc_no_r(n)) + rank[i]
        max_element = max_element * n + (n-1)
        bits = self.abb.get_bits_for_size(max_element)

        # get Maximum
        indices = [self.abb.enc_zero for i in range(n)]
        indices[0] = self.abb.enc_one
        current_max = values[0]
        for i in range(1, len(values)):
            gt = self.abb.gt(values[i], current_max, bits)  
            # update indices
            indices[i] = gt
            for j in range(i):
                indices[j] = self.abb.eval_mul_protocol(indices[j], self.negate(gt))
            # update current maximum
            current_max = (gt * values[i]) + (self.negate(gt) * current_max)
        if decrypt:
            for i in range(len(indices)):
                d = self.abb.dec(indices[i])
                if d == 1:
                    return [i]
        else:
            return indices

    def cand_dict_to_array(self, cand_dict: dict):
        array = []
        for party in cand_dict.keys():
            votes = cand_dict[party]
            array.append(votes)
        return array

    def get_secondary_votes_per_party(self):
        all_parties_states = PartiesAcrossConstituencies(self.all_parties.n_parties, self.all_parties.names_parties, len(self.state_names), self.parties_per_state)
        votes_per_party = all_parties_states.get_empty_general(self.abb)
        for state in self.state_names:
            state = state - 1
            votes_per_party = all_parties_states.add_votes_to_general(self.abb, votes_per_party, state, list((self.secondary_votes[state]).values()))
        self.secondary_votes_per_party = votes_per_party

    
    def get_large_parties(self): 
        """
        Returns a decrypted array containing all parties over 5%.
        """
        CSV_Writer.write_result_benchmarks(self.id, "5% threshold at", int(np.ceil(self.total_valid_secondary_votes * 0.05)))
        min_votes = self.abb.enc_no_r(int(np.ceil(self.total_valid_secondary_votes * 0.05)))
        bits1 = self.abb.get_bits_for_size(self.total_valid_secondary_votes)
        bits2 = self.abb.get_bits_for_size(len(self.constituency_names))
        three = self.abb.enc_no_r(3)
        large_parties = []
        for party in self.parties:
            if party in self.minority_parties:
                large_parties.append(party)
            else:
                min_indicator_5 = self.abb.gt(self.secondary_votes_per_party[party], min_votes, bits1)
                min_indicator_direct = self.abb.gt(self.direct_mandates_party[party], three, bits2)
                min_indicator_added = self.abb.eval_add_protocol(min_indicator_5, min_indicator_direct)
                min_indicator = self.abb.gt(min_indicator_added, self.abb.enc_no_r(1), self.abb.get_bits_for_size(2))
                min_indicator_dec = self.abb.dec(min_indicator) # here decryption is ok
                if min_indicator_dec == 1:
                    large_parties.append(party)
        self.large_parties = large_parties
        CSV_Writer.write_result_benchmarks(self.id, "large parties: ", str(self.large_parties))
        return large_parties
        
    
    def calc_min_seats(self):
        """
        Calculates the minimal number of seats a party must get.
        Returns: dict with a number of seats for each party
        """  
        
        # calculate how many seats a federal state gets
        # see https://bundeswahlleiter.de/dam/jcr/cbceef6c-19ec-437b-a894-3611be8ae886/btw21_heft3.pdf, "erste Oberverteilung"
        log.info("calc min seats")
        start_time = time()
        seats_per_states = SainteLagueEvaluation(self.abb).highest_divisors_unencrypted(self.min_seats_contingent, self.population_distribution)
        end_time = time()
        CSV_Writer.write_times_benchmarks(self.id, "first top distribution", end_time - start_time)
        if(self.debug):
            CSV_Writer.write_result_benchmarks(self.id, "first top distribution", str(seats_per_states))

        # calculate per federal state how many seats a party gets (of all seats for this federal state) 
        # accourding to the secondary votes in each federal state
        # see "1. Unterverteilung"
        seats_state = {}
        for state in range(len(self.state_names)):
            
            relevant_parties = list(filter(lambda x: x in self.large_parties, self.parties_per_state[state]))
        
            relevant_votes = [self.secondary_votes[state][party] for party in relevant_parties]
            n_votes = self.abb.enc_zero
            for v in relevant_votes:
                n_votes = n_votes + v
            start_time = time()
            seats_array = SainteLagueEvaluation(self.abb).get_seat_distribution_many_seats(int(seats_per_states[state]), relevant_votes, n_votes, self.n_valid_secondary_votes[state], self.tie_breaking_iterator, text = "first down distribution for state " + str(state))
            end_time = time()
            CSV_Writer.write_times_benchmarks(self.id, "first down distribution for state: " + str(state), end_time - start_time)
            if(self.debug):
                CSV_Writer.write_result_benchmarks(self.id, "first down distribution for state: " +  str(state), [self.abb.dec(s) for s in seats_array])
            
            seats_state[state] = seats_array
        
                   
        # calculate minimum seats per party 
        # see "Feststellung Mindestsitzanzahlen der Parteien"
        
        # reorder because first votes and secondary votes federal seats have different orderings
        mandates_old = self.direct_mandates_state.copy()
        self.direct_mandates_state[0] = mandates_old[0]
        self.direct_mandates_state[1] = mandates_old[12]
        self.direct_mandates_state[2] = mandates_old[1]
        self.direct_mandates_state[3] = mandates_old[2]
        self.direct_mandates_state[4] = mandates_old[3]
        self.direct_mandates_state[5] = mandates_old[11]
        self.direct_mandates_state[6] = mandates_old[14]
        self.direct_mandates_state[7] = mandates_old[10]
        self.direct_mandates_state[8] = mandates_old[4]
        self.direct_mandates_state[9] = mandates_old[13]
        self.direct_mandates_state[10] = mandates_old[5]
        self.direct_mandates_state[11] = mandates_old[15]
        self.direct_mandates_state[12] = mandates_old[6]
        self.direct_mandates_state[13] = mandates_old[8]
        self.direct_mandates_state[14] = mandates_old[7]
        self.direct_mandates_state[15] = mandates_old[9]
    
        
        log.info("begin 5th column")
        # calculate min seats (5th column)
        min_seats_party = [self.abb.enc_zero for i in range(len(self.large_parties))]
        for i in range(len(self.large_parties)):
            for state in self.state_names:
                start_time = time()
                if self.large_parties[i] in self.parties_per_state[state-1]:
                    pos_partie = 0
                    for j in range(i):
                        if self.large_parties[j] in self.parties_per_state[state-1]:
                            pos_partie = pos_partie + 1 
                    tmp = self.determine_min_seats((seats_state[state-1])[pos_partie], self.direct_mandates_state[state-1][i], self.min_seats_contingent, seats_per_states[state-1])
                    min_seats_party[i] = self.abb.eval_add_protocol(min_seats_party[i], tmp)
                end_time = time()
                CSV_Writer.write_times_benchmarks(self.id, "5th column for state " + str(state) + " and party " + str(i), end_time - start_time)

        if(self.debug):
            tmp = []
            for i in range(len(min_seats_party)):
                tmp.append(self.abb.dec(min_seats_party[i]))
            log.info("mindestsitze" + str(tmp))
            CSV_Writer.write_result_benchmarks(self.id, "mindestsitze, 5. Spalte: ", str(tmp))
    
        

        # calculate min contingent (first column)
        log.info("begin first column")
        start_time = time()
        min_contingent_party = [self.abb.enc_zero for i in range(len(self.large_parties))]
        for i in range(len(self.large_parties)):
            for state in self.state_names:
                if self.large_parties[i] in self.parties_per_state[state-1]:
                    pos_partie = self.pos_partie(state, i)
                    min_contingent_party[i] = self.abb.eval_add_protocol(min_contingent_party[i], (seats_state[state-1])[pos_partie])

        min_claim_party = [None for i in range(len(self.large_parties))]
        bits = self.abb.get_bits_for_size(299)
        for i in range(len(self.large_parties)):
            gt = self.abb.gt(min_seats_party[i], min_contingent_party[i], bits)
            min_claim_party[i] = gt * min_seats_party[i] + (self.abb.enc_one - gt) * min_contingent_party[i]
        overhang = self.calc_overhang(seats_state)
        end_time = time()
        CSV_Writer.write_times_benchmarks(self.id, "calculate surplus", end_time - start_time)
        
         # remove parties that don't get any seats --> only possible for minority parties
        remove = []
        for i in range(len(self.large_parties)):
            if self.large_parties[i] in self.minority_parties:
                # test if minority party has at least one seat
                has_min_one_seat = self.abb.gt(min_claim_party[i], self.abb.enc_no_r(1), self.abb.get_bits_for_size(self.min_seats_contingent))
                has_min_one_seat_dec = self.abb.dec(has_min_one_seat)
                if has_min_one_seat_dec == 0:
                    remove.append(i)
        
        # delete removed parties from relevant variables
        if not len(remove) == 0:
            large_parties_old = self.large_parties
            min_claim_party_old = min_claim_party
            overhang_old = overhang
            self.large_parties = []
            min_claim_party = []
            overhang = []
            for i in range(len(large_parties_old)):
                if not i in remove:
                    self.large_parties.append(large_parties_old[i])
                    min_claim_party.append(min_claim_party_old[i])
                    overhang.append(overhang_old[i])
        
        self.min_claim_party = min_claim_party

        if(self.debug):
            min_claim_dec = [self.abb.dec(x) for x in min_claim_party]
            log.info("min_claim_dec" + str(min_claim_dec))
            CSV_Writer.write_result_benchmarks(self.id, "min_claim: ", str(min_claim_dec))

            sur_dec = [self.abb.dec(x) for x in overhang]
            CSV_Writer.write_result_benchmarks(self.id, "surplus: ", str(sur_dec))


        return min_claim_party, overhang
    
    def pos_partie(self, state_normal, party_index):
        
        pos_partie = 0
        for j in range(party_index):
            if self.large_parties[j] in self.parties_per_state[state_normal-1]:
                pos_partie = pos_partie + 1
        return pos_partie

    def calc_overhang(self, seats_kontingent: dict):
        overhang = []
        bits = self.abb.get_bits_for_size(299)
        for i in range(len(self.large_parties)):
            overhang_party = self.abb.enc_zero
            for state in self.state_names:
                if self.large_parties[i] in self.parties_per_state[state-1]:
                    pos_partie = self.pos_partie(state, i)
                    #self.abb.eval_add_protocol(min_contingent_party[i], (seats_state[state-1])[pos_partie])
                    has_overhang = self.abb.gt(self.direct_mandates_state[state-1][i], (seats_kontingent[state-1])[pos_partie], bits)
                    n_overhang = self.abb.eval_sub_protocol(self.direct_mandates_state[state-1][i], (seats_kontingent[state-1])[pos_partie])
                    add_overhang = self.abb.eval_mul_protocol(has_overhang, n_overhang)
                    overhang_party = self.abb.eval_add_protocol(overhang_party, add_overhang)
            overhang.append(overhang_party)
        return overhang
        
    
    def determine_min_seats(self, seat_kontingent: PaillierCiphertext, n_direct_seats: PaillierCiphertext, max_contingent: int, max_direct_seats: int):
        """
        Calculates the minimal number of seats for a party in a federal state. 
        See "Feststellung Mindestsitzanzahlen der Parteien"
        seat_kontingent: number of seats a party gets in a state according to the prior calculation with the secondary votes.
        n_direct_seats: number of direct mandates a party gets in a state
        max_contingent: maximum possible seat_kontingent
        max_direct_seats: maximum possible number of n_direct_seats
        returns: n_direct_seats, if seat_kontingent <= n_direct_seats
                or ceil((seat_kontingent + n_direct_seats)/2) , if seat_kontingent > n_direct_seats
        """
        max_number = max(max_contingent, max_direct_seats)
        max_diff = max_contingent
        bits = self.abb.get_bits_for_size(max_number)

        direct_greater = self.abb.gt(n_direct_seats, seat_kontingent, bits)
        direct_smaller = self.abb.eval_sub_protocol(1, direct_greater)
        mean = self.calc_rounded_mean(seat_kontingent, n_direct_seats, max_diff)
        prod1 = self.abb.eval_mul_protocol(direct_greater, n_direct_seats)
        prod2 = self.abb.eval_mul_protocol(direct_smaller, mean)
        return self.abb.eval_add_protocol(prod1, prod2)


    def calc_rounded_mean(self, a: PaillierCiphertext, b: PaillierCiphertext, max_a: int):
        """
        Calculates the round up arithmetic mean between the two numbers if a > b
        max_a: upper bound for a
        return ceil((a+b)/2), if a > b, else probably rubbish 
        """
        diff = self.abb.eval_sub_protocol(a, b)
        diff_one =diff + self.abb.enc_one
        bits = self.abb.get_bits_for_size(2 * max_a)
        floor_res = self.floor_division(diff_one, self.abb.enc_no_r(2), max_a + 1, bits)
        result = floor_res + b
        return result

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


    def calculate_total_seats(self, min_claim: list, overhang: list):
        """
        See second top distribution
        """
        print("calculate total seats")
        
        highest_divisor_fraction = self.calculate_highest_divisor(overhang, min_claim)
        #highest_divisor_fraction = Fraction(self.abb.enc_no_r(4805654),self.abb.enc_no_r(83),46442023,598)
        start_time = time()        
        highest_divisor = highest_divisor_fraction.numerator
        multiplied = highest_divisor_fraction.denominator
        max_votes = highest_divisor_fraction.max_denominator * self.total_valid_secondary_votes

        multiplied_votes = [self.relevant_secondary_votes[i] * multiplied for i in range(len(self.large_parties))]
        final_seats_party = SainteLagueEvaluation(self.abb).sl_known_divisor(self.abb, multiplied_votes, max_votes, highest_divisor, min_claim, self.min_seats_contingent)
        end_time = time()
        CSV_Writer.write_times_benchmarks(self.id, "second top distribution", end_time-start_time)
        if self.debug:
            CSV_Writer.write_result_benchmarks(self.id, "second top distribution", final_seats_party)
        return final_seats_party

    
    def calculate_highest_divisor(self, overhang: list, preliminary_seats: list):
        start_time = time()
        bits = self.abb.get_bits_for_size(self.min_seats_contingent)
        no_surplus_seats = [None for i in range(len(overhang))]
        min_one_surplus_seat = [None for i in range(len(overhang))]
        min_two_surplus_seats = [None for i in range(len(overhang))] 
        min_three_surplus_seats = [None for i in range(len(overhang))]

        for i in range(len(overhang)):
            no_surplus_seats[i] = self.if_then_else(self.abb.eq(overhang[i], self.abb.enc_no_r(0), bits), self.abb.enc_one, self.abb.enc_zero)
            min_one_surplus_seat[i] = self.negate(no_surplus_seats[i])
            min_two_surplus_seats[i] = self.abb.gt(overhang[i], self.abb.enc_no_r(2), bits)
            min_three_surplus_seats[i] = self.abb.gt(overhang[i], self.abb.enc_no_r(3), bits)
        end_time = time()
        CSV_Writer.write_times_benchmarks(self.id, "preperation divisor calculation", end_time-start_time)
        
        start_time = time()
        min_no_surplus = self.calc_divisor_no_surplus(preliminary_seats)
        end_time = time()
        CSV_Writer.write_times_benchmarks(self.id, "divisor without surplus", end_time-start_time)
        
        start_time = time()
        min_surplus = self.calc_divisor_surplus(preliminary_seats, no_surplus_seats, min_one_surplus_seat, min_two_surplus_seats, min_three_surplus_seats)
        end_time = time()
        CSV_Writer.write_times_benchmarks(self.id, "divisor with surplus", end_time-start_time)
        
        
        f_abb = FractionABB(self.abb)
        start_time = time()
        gt = f_abb.gt(min_no_surplus, min_surplus) # be careful: wrong intermediate result in official documentation
        # if then else
        neg = self.abb.eval_sub_protocol(1, gt)
        min_numerator = gt * min_no_surplus.numerator + neg * min_surplus.numerator
        min_denominator = gt * min_no_surplus.denominator + neg * min_surplus.denominator
        minimum = Fraction(min_numerator, min_denominator, self.total_valid_secondary_votes, 598)
        end_time = time()
        CSV_Writer.write_times_benchmarks(self.id, "calculation divisor final", end_time-start_time)
        return minimum 


    def calc_divisor_no_surplus(self, preliminary_seats: list):
        """
        Calculate minimal divisor without surplus seats for the first part of the second distribution.
        """
        log.info("divisor no surplus")

        n = len(preliminary_seats)
        f_abb = FractionABB(self.abb)
        divisors = []
        for i in range(n):
            divisors.append(Fraction(self.abb.enc_no_r(2) * self.relevant_secondary_votes[i], self.abb.enc_no_r(2) * preliminary_seats[i] - self.abb.enc_no_r(1), 2 * self.total_valid_secondary_votes, 2 * self.min_seats_contingent -1))
        min_index = f_abb.get_min_as_index(divisors)
        minimum = f_abb.index_mult(min_index, divisors, len(divisors))
        return minimum

    def calc_divisor_surplus(self, preliminary_seats: list, no_surplus_seats: list, min_one_surplus_seat: list, min_two_surplus_seats: list, min_three_surplus_seats: list):
        """
        Return 4^th minimal divisor of parties with surplus seats.
        """
        log.info("divisor surplus")
        
        n = len(preliminary_seats)
        f_abb = FractionABB(self.abb)
        pointer = [[self.abb.enc_one, self.abb.enc_zero, self.abb.enc_zero, self.abb.enc_zero, self.abb.enc_zero] for i in range(n)]
        divisors = [[None for j in range(4)] for i in range(n)]
        for i in range(n):
            divisors[i][0] = Fraction(self.abb.enc_no_r(2) * self.relevant_secondary_votes[i] * min_one_surplus_seat[i], self.abb.enc_no_r(2) * preliminary_seats[i] - self.abb.enc_no_r(1), 2 * self.total_valid_secondary_votes, 2 * self.min_seats_contingent -1)
            divisors[i][1] = Fraction(self.abb.enc_no_r(2) * self.relevant_secondary_votes[i] * min_one_surplus_seat[i], self.abb.enc_no_r(2) * preliminary_seats[i] - self.abb.enc_no_r(3), 2 * self.total_valid_secondary_votes, 2 * self.min_seats_contingent -1)
            divisors[i][2] = Fraction(self.abb.enc_no_r(2) * self.relevant_secondary_votes[i] * min_two_surplus_seats[i], self.abb.enc_no_r(2) * preliminary_seats[i] - self.abb.enc_no_r(5), 2 * self.total_valid_secondary_votes, 2 * self.min_seats_contingent -1)
            divisors[i][3] = Fraction(self.abb.enc_no_r(2) * self.relevant_secondary_votes[i] * min_three_surplus_seats[i], self.abb.enc_no_r(2) * preliminary_seats[i] - self.abb.enc_no_r(7), 2 * self.total_valid_secondary_votes, 2 * self.min_seats_contingent -1)
        for i in range(4):
            current_divisors = []
            for i in range(n):
                numerator = pointer[i][0]*divisors[i][0].numerator + pointer[i][1]*divisors[i][1].numerator + pointer[i][2]*divisors[i][2].numerator + pointer[i][3]*divisors[i][3].numerator
                denominator = pointer[i][0]*divisors[i][0].denominator + pointer[i][1]*divisors[i][1].denominator + pointer[i][2]*divisors[i][2].denominator + pointer[i][3]*divisors[i][3].denominator
                current_divisors.append(Fraction(numerator, denominator, 2 * self.total_valid_secondary_votes, 2 * self.min_seats_contingent -1))
            min_index = f_abb.get_nonzero_min_as_index(current_divisors)
            # update pointer
            for i in range(n):
                for j in range(3,0,-1):
                    pointer[i][j] = self.negate(min_index[i]) * pointer[i][j] + min_index[i] * pointer[i][j-1]
                pointer[i][0] = self.negate(min_index[i]) * pointer[i][0]
        minimum = f_abb.index_mult(min_index, current_divisors, len(divisors))
        return minimum


    
    def second_down_distribution(self, final_seats):
        print("second down distribution")
        self.final_seats_per_state = [[self.abb.enc_zero for i in self.large_parties] for j in self.state_names]
        for i in range(len(self.large_parties)):
            if i == 6 or i==7: # parties that only compete in one federal state
                for c in self.state_names:
                    if self.large_parties[i] in self.parties_per_state[c-1]:
                        self.final_seats_per_state[c-1][i] = self.abb.enc_no_r(final_seats[i])
                    else:
                        self.final_seats_per_state[c-1][i] = self.abb.enc_zero
                if self.debug:
                    CSV_Writer.write_result_benchmarks(self.id, "second down distribution for party " + str(i), [[self.abb.dec(a) for a in b] for b in self.final_seats_per_state])
            else:
                start_time = time()
                votes_party_per_state = []
                for c in self.state_names:
                    if self.large_parties[i] in self.parties_per_state[c-1]:
                        votes_party_per_state.append(self.secondary_votes[c-1][self.large_parties[i]])
                    else:
                        votes_party_per_state.append(self.abb.enc_zero)
                sum_votes = self.abb.enc_zero
                for v in votes_party_per_state:
                    sum_votes = sum_votes + v
                result = SainteLagueEvaluation(self.abb).get_seat_distribution_many_seats(final_seats[i], votes_party_per_state, sum_votes, self.total_valid_secondary_votes, self.tie_breaking_iterator, text = "second down distribution for " + str(i))
                for c in self.state_names:
                    self.final_seats_per_state[c-1][i] = result[c-1]
                end_time = time()
                CSV_Writer.write_times_benchmarks(self.id, "second down distribution for party " + str(i), end_time-start_time)
                if self.debug:
                    CSV_Writer.write_result_benchmarks(self.id, "second down distribution for party " + str(i), [self.abb.dec(r) for r in result])
        if self.debug:
                CSV_Writer.write_result_benchmarks(self.id, "second down distribution ", [[self.abb.dec(a) for a in b] for b in self.final_seats_per_state])

    def remaining_overhang(self, final_seats, final_seats_per_state):
        print("remaining overhang")
        summed_state_seats = [self.abb.enc_zero for p in self.large_parties]
        for p in range(len(self.large_parties)):
            for c in self.state_names:
                summed_state_seats[p] = final_seats_per_state[c-1][p]
        f_abb = FractionABB(self.abb)
        
        for u in range(3):
            überhang = []
            for p in range(8):
                for j in range(len(summed_state_seats)):
                    überhang.append(final_seats[j] - summed_state_seats[j])
            # distribute remaining overhang seats
            for p in range(8):
                start_time = time()
                if p == 6: # CSU (only competing in the federal state Bavaria)
                    final_seats_per_state[14][p] = final_seats_per_state[14][p] + überhang[p]
                if p == 7: # SSW (only competing in the federal state Schleswig-Holstein)
                    final_seats_per_state[1][p] = final_seats_per_state[1][p] + überhang[p]

                L = []
                is_zero = self.abb.eq(überhang[p], self.abb.enc_zero, self.abb.get_bits_for_size(3))
                is_one = self.negate(is_zero)
                is_two = self.abb.gt(überhang[p], self.abb.enc_no_r(2), self.abb.get_bits_for_size(3))
                is_three = self.abb.gt(überhang[p], self.abb.enc_no_r(3), self.abb.get_bits_for_size(3))
                local_s = 0
                local_to_global = []                
                for s in self.state_names:
                    if self.large_parties[p] in self.parties_per_state[s-1]:
                        pos_partie = p
                        local_s = local_s + 1
                        local_to_global.append(s-1)
                        local_L = []
                        local_L.append(Fraction(self.abb.enc_no_r(2) * self.secondary_votes[s-1][self.large_parties[pos_partie]] * is_one, self.abb.enc_no_r(2) * self.min_claim_party[pos_partie] - self.abb.enc_no_r(1), 2 * self.total_valid_secondary_votes, 2 * self.min_seats_contingent))
                        local_L.append(Fraction(self.abb.enc_no_r(2) * self.secondary_votes[s-1][self.large_parties[pos_partie]] * is_two, self.abb.enc_no_r(2) * self.min_claim_party[pos_partie] - self.abb.enc_no_r(3), 2 * self.total_valid_secondary_votes, 2 * self.min_seats_contingent))
                        local_L.append(Fraction(self.abb.enc_no_r(2) * self.secondary_votes[s-1][self.large_parties[pos_partie]] * is_three, self.abb.enc_no_r(2) * self.min_claim_party[pos_partie] - self.abb.enc_no_r(5), 2 * self.total_valid_secondary_votes, 2 * self.min_seats_contingent))
                        L.append(local_L)
                indices = [[self.abb.enc_one, self.abb.enc_zero, self.abb.enc_zero] for s in self.state_names]
                for i in range(3):
                    current_elements = []
                    for s in range(local_s):
                        current_elements.append(f_abb.index_mult(indices[s], L[s], 3))
                    index = f_abb.get_nonzero_min_as_index(current_elements)
                    for s in range(local_s):
                        print("s: " + str(s))
                        if i==0:
                            a = is_one
                        if i==1:
                            a = is_two
                        if i==2:
                            a = is_three
                        final_seats_per_state[local_to_global[s]][p] = final_seats_per_state[local_to_global[s]][pos_partie] + index[s] * a
                end_time = time()
                CSV_Writer.write_times_benchmarks(self.id, "overhang for " + str(p), end_time -start_time)
        
        # decrypt final seats per state
        for s in self.state_names:
            for p in range(len(self.large_parties)):
                final_seats_per_state[s-1][p] = int(self.abb.dec(final_seats_per_state[s-1][p]))
        if self.debug:
            CSV_Writer.write_result_benchmarks(self.id, "final seats per state", final_seats_per_state)
        return final_seats_per_state
    

    def if_then_else(self, cond, true_val, false_val):
        neg_cond = self.abb.eval_sub_protocol(1, cond)
        prod1 = self.abb.eval_mul_protocol(cond, true_val)
        prod2 = self.abb.eval_mul_protocol(neg_cond, false_val)
        return self.abb.eval_add_protocol(prod1, prod2)

    def negate(self, cipher):
        return self.abb.eval_sub_protocol(1, cipher)

    
    
    def get_min(self, list: list, biggest_possible_number: int):
        """
        Returns the minimum element of a list of Ciphertexts.
        """
        bits = self.abb.get_bits_for_size(biggest_possible_number)
        minimum = list[0]
        for i in range(1, len(list)):
            update = self.abb.gt(minimum, list[i], bits)
            minimum = self.if_then_else(update, list[i], minimum)
        return minimum

    
    
    

   