import logging

from src.election.election_properties import ElectionProperties
from src.election.evaluation.bundestag_evaluation import BundestagEvaluation
from src.election.aggregation.simple_addition import SimpleAdditionVoteAggregation
from src.election.real_world_elections.helpers import TwoVotesCandidatesSimplified
from src.election.votes.vote import ConstituencyVote
from src.election.votes.one_candidate_vote import VoteType
from src.util.point_vote import IllegalVoteException
from src.crypto.abb import ABB
from src.election.tie_breaking.tie_breaking import TieBreaking, BreakingTypes

log = logging.getLogger(__name__)


class BundestagElection(ElectionProperties):
    aggregator = SimpleAdditionVoteAggregation

    
    def __init__(self, candidates, all_candidates: TwoVotesCandidatesSimplified, constituencies=299, states = 16, const_state_mapping: dict = None,  min_seats=598, 
                 population_distribution=None, minority_parties: list = [], break_ties = True, max_ties = 0, tie_breaking_iterator = None,
                 ballot_format=None, start_date=None, due_date=None, title="", expected_votes_n=None, use_constituencies = True,
                 logger_name_extension="", peer_server_address="", authentication_server_address="", bulletin_board_address=""):
        # TODO: set population_distribution and const_state_mapping a reasonable default value
        """candidates, constituencies and states could either be the amount of cands or a list of names
        min_seats: Minimal number of seats that has to be allocated
        population_distribution: number of citizens per constituency
        """
        super().__init__(candidates, ballot_format, start_date, due_date, title, expected_votes_n, use_constituencies, 
                 break_ties, max_ties, tie_breaking_iterator, 
                 logger_name_extension, peer_server_address, authentication_server_address, bulletin_board_address)
        if isinstance(constituencies, list):
            self.n_constituencies = len(constituencies)
            self.constituency_names = constituencies
        else:
            self.n_constituencies = constituencies
            self.constituency_names = [i for i in range(self.n_constituencies)]
        if isinstance(states, list):
            self.n_states = len(states)
            self.state_names = states
        else:
            self.n_states = states
            self.state_names = [i for i in range(self.n_states)]
        self.const_state_mapping = const_state_mapping
        self.min_seats = min_seats
        self.population_distribution = population_distribution
        self.minority_parties = minority_parties
        self.all_candidates = all_candidates

    def setup(self, abb):
        # used to generate ties
        # only for random ties
        tie_breaking_iterator = TieBreaking(10)
        list = [i for i in range(1, max(self.n_constituencies, self.n_cand, 11))]
        tie_breaking_iterator.setup_multiple_arrays(abb, list, BreakingTypes.random)
        self.tie_breaking_iterator = tie_breaking_iterator
    

    def serialize(self):
        bundestagswahl_specific_settings = {
            "constituencies": self.constituency_names,
            "states": self.state_names,
            "const_state_mapping": self.const_state_mapping,
            "min_seats": self.min_seats,
            "population_distribution": self.population_distribution,
            "minority_parties": self.minority_parties,
            "all_candidates": self.all_candidates.serialize()
        }
        settings = super().serialize()
        settings["evaluation"].setdefault("settings", {}).update(bundestagswahl_specific_settings)
        return settings

    @classmethod
    def serialized_to_args(cls, serialized):
        bundestagswahl_specific_args = {
            "all_candidates": TwoVotesCandidatesSimplified.deserialize(serialized["evaluation"]["settings"].get("all_candidates")),
            "constituencies": serialized["evaluation"]["settings"].get("constituencies"),
            "states": serialized["evaluation"]["settings"].get("states"),
            "const_state_mapping": serialized["evaluation"]["settings"].get("const_state_mapping"),
            "min_seats": serialized["evaluation"]["settings"].get("min_seats"),
            "population_distribution": serialized["evaluation"]["settings"].get("population_distribution"),
            "minority_parties": serialized["evaluation"]["settings"].get("minority_parties"),
            
        }
        args = super().serialized_to_args(serialized)
        args.update(bundestagswahl_specific_args)
        return args

    
    def aggregator(self):
        return SimpleAdditionVoteAggregation()
 
    def get_evaluator(self, n_votes, abb):
        # real-world data in inserted later
        return BundestagEvaluation(abb)

   

    def generate_valid_vote(self, generic_vote : ConstituencyVote, abb: ABB):
        """
        Test if a generic vote is a valid vote for this election.
        If yes, the vote is encrypted and returned
        """
        if not (len(generic_vote.getChoices()) == self.n_cand):
            raise IllegalVoteException("Wrong number of candidates used.")
        if not( 0 <= generic_vote.getConstitency() < self.n_constituencies):
            raise IllegalVoteException("Constituency does not exist.")
        count_zeros = 0
        count_ones = 0
        for choice in generic_vote.getChoices():
            if choice == 0:
                count_zeros += 1
            if choice == 1:
                count_ones +=1
        if not (count_zeros == self.n_cand - 1 and count_ones == 1):
            raise IllegalVoteException("The selected candidate must be marked with one and all others with zero.")
        
        if not generic_vote.getType() in [VoteType.first_vote, VoteType.secondary_vote, None]:
            raise IllegalVoteException("Vote type is not valid.")

        # encrypt votes
        choices = generic_vote.getChoices()
        for i in range(len(choices)):
            choices[i] = abb.enc_no_r(choices[i])
        generic_vote.setChoices(choices)
        return generic_vote

    
    def serialize_vote(self, valid_vote: ConstituencyVote):
        serialized_vote = {
            "choices": [
                choice.serialize() for choice in valid_vote.getChoices()
            ],
            "constituency": valid_vote.getConstitency(),
            "type": valid_vote.getType()
        }
        return serialized_vote

    def deserialize_vote(self, serialized_vote, abb: ABB):
        choices = [
            abb.deserialize_cipher(choice) for choice in serialized_vote["choices"]
        ]
        constituency = serialized_vote["constituency"]
        type = serialized_vote["type"]
        return ConstituencyVote(choices, constituency, type)
    
    def get_choices(self, serialized_vote, abb: ABB):
        return [abb.deserialize_cipher(choice) for choice in serialized_vote["choices"]]
