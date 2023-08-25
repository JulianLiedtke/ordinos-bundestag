import logging
from src.election.aggregation.simple_addition import SimpleAdditionVoteAggregation

from src.election.evaluation.evaluation_protocol import EvaluationProtocol
from src.election.evaluation.single_winner_evaluation import \
    SingleWinnerEvaluation

log = logging.getLogger(__name__)


class SimpleWinnerEvaluation(EvaluationProtocol):

    def __init__(self, bits, num_winners = 1, encrypted=False, enc_result = False, abb=None):
        super().__init__()
        self.num_winners = num_winners
        self.enc_threshold = encrypted
        self.enc_result = enc_result
        self.bits = bits
        self.abb = None
        
        self.set_abb(abb)

    def get_abb(self):
        return self.abb

    def run(self, vote_aggregation, fewer_bits = None):
        bits_changed = False
        if fewer_bits is not None:
            # change self.bits but only for this run
            bits_changed = True
            old_bits = self.bits
            self.bits = fewer_bits

        self.bits_for_candidates = self.abb.get_bits_for_size(len(vote_aggregation))
        self.debug_cipher_list(log, 'Winner election of: %s', vote_aggregation)
        if self.enc_threshold is False and self.num_winners == 1:
            return self.run_subprotocol(SingleWinnerEvaluation(pow(2, self.bits), enc_result=self.enc_result), [vote_aggregation])
        duel_matrix = self.create_duel_matrix(vote_aggregation)
        if self.enc_threshold:
            num_votes = self.abb.enc_no_r(len(vote_aggregation))
        else:
            num_votes = len(vote_aggregation)
        threshold = num_votes - self.num_winners
        log.debug('Comparator: {}'.format(threshold))
        win_vec = self.create_wins_vector(duel_matrix)
        self.debug_cipher_list(log, 'wins: %s', win_vec)
        enc_winner = self.find_gt_candidates(win_vec, threshold, self.enc_threshold)
        if self.enc_threshold or self.enc_result:
            return enc_winner

        indicator_winner = {}
        for i, val in enc_winner.items():
            indicator_winner[i] = self.abb.dec(val)
        winners = []
        for i, val in indicator_winner.items():
            if val == 1:
                winners.append(i)

        # rechange bits:
        if bits_changed:
            self.bits = old_bits
        return winners

    def create_duel_matrix(self, votes):
        """ outputs a ranking matrix """
        matrix = {} 
        for i, a in votes.items():
            matrix[i] = {}
        for i, a in votes.items():
            for j, b in votes.items():
                if i == j:
                    matrix[i][j] = self.abb.enc_zero
                    matrix[j][i] = self.abb.enc_zero
                elif j < i:
                    gt = self.abb.gt(a, b, self.bits)
                    eq = self.abb.eq(a, b, self.bits)
                    matrix[i][j] = gt
                    matrix[j][i] = 1 - gt + eq
        return matrix

    def create_wins_vector(self, matrix):
        """ outputs a vector consisting of wins per candidate """
        wins_vec = {}
        for i, row in matrix.items():
            wins_vec[i] = self.abb.enc_zero
            for j, val in row.items():
                wins_vec[i] += val
        return wins_vec

    def find_gt_candidates(self, wins_vector, threshold_wins, encrypted=False):
        """ find all candidates which have at least threshold many wins """
        enc_cand_indicator = {}
        wins = threshold_wins
        enc_wins = None
        if encrypted:
            enc_wins = wins
        else:
            enc_wins = self.abb.enc_no_r(wins)
        for i, val in wins_vector.items():
            enc_cand_indicator[i] = self.abb.gt(val, enc_wins, self.bits_for_candidates)
        return enc_cand_indicator

    def find_lt_candidates(self, wins_vector, threshold_wins):
        """ find all candidates which have at most threshold many wins """
        enc_cand_indicator = {}
        wins = threshold_wins
        enc_wins = self.abb.enc_no_r(wins)
        for i, val in wins_vector.items():
            enc_cand_indicator[i] = self.abb.eq(val, enc_wins, self.bits_for_candidates) - self.abb.gt(val, enc_wins, self.bits_for_candidates) + 1
        return enc_cand_indicator

    def find_eq_candidates(self, wins_vector, threshold_wins):  
        """ find all candidates which have exactly threshold many wins """
        enc_cand_indicator = {}
        wins = threshold_wins
        enc_wins = self.abb.enc_no_r(wins)
        for i, val in wins_vector.items():
            enc_cand_indicator[i] = self.abb.eq(val, enc_wins, self.bits_for_candidates)
        return enc_cand_indicator
