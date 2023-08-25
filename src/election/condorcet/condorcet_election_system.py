import logging
import math
from src.election.evaluation.evaluation_protocol import EvaluationProtocol
from src.protocols.protocol import Protocol
from src.election.ballotformat import BallotFormat

from src.election.condorcet.condorcet_evaluation import CondorcetEvaluation
from src.election.election_properties import ElectionProperties
from src.election.condorcet.condorcet_aggregation import CondorcetVoteAggregation

log = logging.getLogger(__name__)


class Condorcet(ElectionProperties):

    aggregator = CondorcetVoteAggregation

    def __init__(self, candidates, additional_evaluators=None, leak_better_half=False, evaluate_condorcet=True, **base_settings):
        """if leak_better_half is set True, the group of candidates, who get the half of borda points, will be computed
        if there is a winner, he has to be out of this group <<https://en.wikipedia.org/wiki/Nanson%27s_method>>"""
        base_settings["logger_name_extension"] = str(leak_better_half)
        super().__init__(candidates, **base_settings)
        self.additional_evaluators = additional_evaluators
        self.leak_better_half = leak_better_half
        self.evaluate_condorcet = evaluate_condorcet

    def serialize(self):
        condorcet_specific_settings = {
            "additional_evaluators": [evaluator.serialize() for evaluator in self.additional_evaluators] if self.additional_evaluators is not None else [],
            "leak_better_half": self.leak_better_half,
            "evaluate_condorcet": self.evaluate_condorcet
        }
        settings = super().serialize()
        settings["evaluation"].setdefault("settings", {}).update(condorcet_specific_settings)
        return settings

    @classmethod
    def serialized_to_args(cls, serialized):
        condorcet_specific_args = {
            "additional_evaluators": [EvaluationProtocol.deserialize(evaluator)
                for evaluator in serialized["evaluation"]["settings"].get("additional_evaluators", [])],
            "leak_better_half": serialized["evaluation"]["settings"].get("leak_better_half"),
            "evaluate_condorcet": serialized["evaluation"]["settings"].get("evaluate_condorcet"),
        }
        args = super().serialized_to_args(serialized)
        args.update(condorcet_specific_args)
        return args

    def get_evaluator(self, n_votes, abb):
        return CondorcetEvaluation(n_votes, self.additional_evaluators, self.leak_better_half, self.evaluate_condorcet)

    def generate_valid_vote(self, generic_vote, abb):
        """ generate encrypted gt-matrix and borda points if optimization activated out of generic vote """

        gt_matrix_enc = [[abb.enc(e) for e in row] for row in generic_vote.get_duel_matrix()]

        borda_points_enc = []
        if (self.leak_better_half):
            borda_winner_points =  [i for i in range(self.n_cand, 0, -1)]

            doubled_borda_points = {}
            for position, candidates in generic_vote.get_ranking_map().items():
                points_to_devide = 0
                if position > 0:
                    for i in range(len(candidates)):
                        points_to_devide += borda_winner_points[position - 1 + i]
                    for cand in candidates:
                        doubled_borda_points[cand] = math.ceil(2*points_to_devide/len(candidates))
                else:
                    # -1 would mean the candidate is ignored and has same preference as all other
                    # the average of borda points will be given
                    for cand in candidates:
                        doubled_borda_points[cand] = self.n_cand + 1

            borda_points_enc = [abb.enc(doubled_borda_points[i]) for i in range(len(doubled_borda_points))]
        
        return [gt_matrix_enc, borda_points_enc]

    def serialize_vote(self, valid_vote):
        serialized_vote = {"choices": [
            [[vote.serialize() for vote in ranking_map] for ranking_map in valid_vote[0]],
            [vote.serialize() for vote in valid_vote[1]]
        ]}
        return serialized_vote

    def deserialize_vote(self, serialized_vote, abb):
        deserialized_vote = [
            [[abb.deserialize_cipher(vote) for vote in ranking_map] for ranking_map in serialized_vote["choices"][0]],
            [abb.deserialize_cipher(vote) for vote in serialized_vote["choices"][1]]
        ]
        return deserialized_vote