import logging
import random as rd
from time import time
import numpy as np
import enum

log = logging.getLogger(__name__)

from src.election.votes.vote import ConstituencyVote
from src.crypto.abb import ABB


class VoteType(str, enum.Enum):
    first_vote: str = "first_vote"
    secondary_vote: str = "secondary_vote"



class OneCandidateVote(ConstituencyVote):
    
    @staticmethod
    def generate_random_votes(n_votes, n_cand, n_constituencies, type: VoteType = None):
        start_time = time()
        votes = []
        for i in range(n_votes):
            choices = OneCandidateVote.generate_random_choice(n_cand)
            constituency = OneCandidateVote.generate_random_constituency(n_constituencies)
            vote = OneCandidateVote(choices, constituency, type)
            votes.append(vote)
        end_time = time()
        
        log.info("Computation time to generate random votes: " + str(end_time - start_time))
        if n_votes <= 10:
            log.info("Generated votes: " + str(votes))
        return votes

    
    @staticmethod
    def generate_encrypted_votes(abb: ABB, n_votes, n_cand, n_const):
        """
        Generate random votes and encrypt them.
        TODO: probably not useful
        """
        start_time = time()
        votes = OneCandidateVote.generate_random_votes(n_votes, n_cand, n_const)
        for vote in votes:
            choices = vote.getChoices()
            for choice in choices:
                choice = abb.enc_no_r(choice)
            vote.setChoices(choices)
        end_time = time()
        
        log.info("Computation time to generate random votes: " + str(end_time - start_time))
        if n_votes <= 10:
            log.info("Generated votes: " + str(votes))
        return votes

    @staticmethod
    def generate_random_choice(n_cand):
        """
        Generate random votes voting for one candidate, e.g. [0,0,1,0] elects the third candidate.
        """
        choice = [0 for k in range(n_cand)]
        choice[rd.randint(0, n_cand -1)] = 1
        return choice

    def generate_random_constituency(n_constituencies):
        return rd.randint(0, n_constituencies -1)    
