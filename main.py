"""Main file of the Ordinos sw"""

import logging
import random
import numpy as np
import pandas as pd

from src.crypto.paillier_abb import PaillierABB
from src.election.borda.borda_election_system import Borda
from src.election.borda.borda_implementations import (
    EscElection, FisWorldCup, MedalTableSystem)
from src.election.condorcet.condorcet_election_system import Condorcet
from src.election.condorcet.condorcet_implementations import (
    Copeland, MiniMaxMarginsSmith, MiniMaxWinningVotesSmith)
from src.election.condorcet.condorcet_no_winner_evaluations import MiniMaxMarginsEvaluation, SchulzeEvaluation, SmithEvaluation, SmithFastEvaluation, WeakCondorcetEvaluation
from src.election.election_authority import ElectionAuthority
from src.election.instant_runoff_voting.irv_election_system import (
    IRVElectionSystemAlternative, IRVElectionSystemNormal)
from src.election.single_vote.single_vote_election_system import \
    SingleVoteElection
from src.election.parliamentary_ballot.parliamentary_ballot_properties import ParliamentaryBallotProperties
from src.protocols.sublinear import SubLinearProtocolSuite
from src.util.csv_writer import CSV_Writer
from src.util.logging import setup_logging
from src.util.point_vote import PointVote
from src.util.position_vote import PositionVote
from src.election.votes.one_candidate_vote import OneCandidateVote
from src.election.votes.one_candidate_vote import VoteType
from src.election.ballotformat import CheckboxBallotFormat

from src.election.real_world_elections.bundestagelection import BundestagElection
from src.election.real_world_elections.helpers import TwoVotesCandidatesSimplified, PartiesAcrossConstituencies
from src.election.trustee import trustee, trustee_client
import multiprocessing as mp
log = logging.getLogger(__name__)

if __name__ == '__main__':
    CSV_Writer.init_writer()
    # set general parameters of Ordinos
    abbs = PaillierABB.gen_trustee_abbs(2048, 2, 2, SubLinearProtocolSuite) #to run with Paillier
    # To run the trustees locally use:
    # key_generator = lambda bulletin_board: trustee.init_trustees(bulletin_board, abbs, [i for i, _ in enumerate(abbs)])
    # To use the trustee servers use:
    key_generator = lambda bulletin_board: trustee_client.init_trustees(bulletin_board, abbs, [i for i, _ in enumerate(abbs)])
    # Currently at least the bulletin board server has to be running to any evaluation. Start it via start_bb_server.py script.

    setup_logging(logging.INFO)


    """
    There is sample data for the Bundestag evaluation included, but this data is ignored. Instead, the real-world-data from the 2021 election is used.
    """
    n_cand = 5
    n_votes = 500
    random.seed(3)
    votes = PointVote.generate_random_point_votes(n_votes=n_votes, n_cand=n_cand, max_points=5)
    votes = PositionVote.generate_random(n_votes, n_cand)

    n_cand = 3
    constituencies = 3
    parties_per_const = [[0,1,2],[0,1],[0,2]]
    all_parties = PartiesAcrossConstituencies(3, [0, 1, 2], constituencies, parties_per_const)
    direct_p1 = {}
    direct_p1[0] = 0#"a0"
    direct_p1[1] = 2#"a2"
    direct_p1[2] = 5#"a5"

    direct_p2 = {}
    direct_p2[0] = 0#"b0"
    direct_p2[1] = 8#"b8"
    direct_p2[2] = 5#"b5"

    direct_p3 = {}
    direct_p3[0] = 0#"c0"
    direct_p3[1] = 3#"c3"
    direct_p3[2] = 5#"c5"

    direct_p4 = {}
    direct_p4[0] = 0#"d0"
    direct_p4[1] = 5#"d5"
    direct_p4[2] = 6#"d6"

    direct_candidates = {}
    direct_candidates[0] = direct_p1
    direct_candidates[1] = direct_p2
    direct_candidates[2] = direct_p3
    direct_candidates[3] = direct_p4

    list_candidates = {}
    list_candidates[0] = ["a" + str(i) for i in range(10)]
    list_candidates[1] = ["b" + str(i) for i in range(10)]
    list_candidates[2] = ["c" + str(i) for i in range(10)]
    list_candidates[3] = ["d" + str(i) for i in range(10)]

    all_parties = TwoVotesCandidatesSimplified(direct_candidates, list_candidates)
    seats_per_constituencies = [3, 4, 5]
    n_seats = 14
    population_distribution = [300, 400, 500]
    no_real_parties = [3]

    votes = []
    n_cand_per_const = [3, 3, 3]
    for j in range(2):
        for i in range(len(n_cand_per_const)):
            choices = OneCandidateVote.generate_random_choice(n_cand_per_const[i])
            votes.append(OneCandidateVote(choices, i, VoteType.first_vote))
    for j in range(2):
        for i in range(len(n_cand_per_const)):
            choices = OneCandidateVote.generate_random_choice(n_cand_per_const[i])
            votes.append(OneCandidateVote(choices, i, VoteType.secondary_vote))

    ElectionAuthority(key_generator, BundestagElection(n_cand, all_parties, constituencies=constituencies)).add_votes_and_evaluate(votes)



    """
    ElectionAuthority(key_generator, ParliamentaryBallotProperties(n_cand, 10, False, 0)).add_votes_and_evaluate(votes)
    ElectionAuthority(key_generator, SingleVoteElection(n_cand)).add_votes_and_evaluate(votes)
    # # borda variants
    ElectionAuthority(key_generator, Borda(n_cand, ballot_format=CheckboxBallotFormat(1, 3), point_limit=int((n_cand + 1) * n_votes / 2))).add_votes_and_evaluate(votes) # original Borda point limit
    ElectionAuthority(key_generator, Borda(n_cand, num_winners=4)).add_votes_and_evaluate(votes) # original Borda set winner count
    ElectionAuthority(key_generator, Borda(n_cand)).add_votes_and_evaluate(votes) # original Borda
    ElectionAuthority(key_generator, Borda(n_cand, list_of_points=[1])).add_votes_and_evaluate(votes) # equals SingleVoteSystem
    ElectionAuthority(key_generator, Borda(n_cand, list_of_points=[3,2,1])).add_votes_and_evaluate(votes)
    ElectionAuthority(key_generator, Borda(n_cand, list_of_points=[10,9,8,7,6,5,4,3,2,1], allow_less_cands=True, begin_with_last=False)).add_votes_and_evaluate(votes)
    ElectionAuthority(key_generator, Borda(n_cand, list_of_points=[10,9,8,7,6,5,4,3,2,1], allow_less_cands=True, begin_with_last=True)).add_votes_and_evaluate(votes)
    ElectionAuthority(key_generator, EscElection(n_cand)).add_votes_and_evaluate(votes)
    ElectionAuthority(key_generator, MedalTableSystem(n_cand, n_votes)).add_votes_and_evaluate(votes)
    ElectionAuthority(key_generator, FisWorldCup(n_cand, n_votes)).add_votes_and_evaluate(votes)

    # # condorcet variants
    ElectionAuthority(key_generator, Condorcet(n_cand, leak_better_half=False)).add_votes_and_evaluate(votes)
    ElectionAuthority(key_generator, Condorcet(n_cand, [WeakCondorcetEvaluation], evaluate_condorcet=False)).add_votes_and_evaluate(votes)
    ElectionAuthority(key_generator, Copeland(n_cand, leak_max_points=True, evaluate_condorcet=False)).add_votes_and_evaluate(votes)
    ElectionAuthority(key_generator, Copeland(n_cand, leak_max_points=False, evaluate_condorcet=False)).add_votes_and_evaluate(votes)
    ElectionAuthority(key_generator, Condorcet(n_cand, [MiniMaxMarginsEvaluation], evaluate_condorcet=False)).add_votes_and_evaluate(votes)
    ElectionAuthority(key_generator, Condorcet(n_cand, [SmithEvaluation], evaluate_condorcet=False)).add_votes_and_evaluate(votes)
    ElectionAuthority(key_generator, Condorcet(n_cand, [SmithFastEvaluation], evaluate_condorcet=False)).add_votes_and_evaluate(votes)
    ElectionAuthority(key_generator, Condorcet(n_cand, [SchulzeEvaluation], evaluate_condorcet=False)).add_votes_and_evaluate(votes)

    # irv variants
    ElectionAuthority(key_generator, IRVElectionSystemNormal(n_cand, bits_int=16, sys_id=0)).add_votes_and_evaluate(votes)
    ElectionAuthority(key_generator, IRVElectionSystemAlternative(n_cand, bits_int=16, sys_id=5)).add_votes_and_evaluate(votes) # generic vote couldn't be adapted
    """