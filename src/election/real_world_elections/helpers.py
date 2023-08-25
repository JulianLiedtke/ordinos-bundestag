from src.crypto.abb import ABB
import json
import logging
log = logging.getLogger(__name__)


class PartiesAcrossConstituencies():
    """
    Object to store which parties are in which constituencies and to transform the votes per constituencies to general votes
    """
    def __init__(self, n_parties, names_parties, n_const, parties_per_const):
        self.n_parties = n_parties
        self.names_parties = names_parties
        self.n_const = n_const
        self.parties_per_const = parties_per_const

    def get_n_parties_per_const(self, const):
        return len(self.parties_per_const[const])
    
    def add_votes_to_general(self, abb: ABB, votes_general, const, votes_const):
        """
        add constituency votes to a list of votes containing all parties
        """
        for i in range(len(self.parties_per_const[const])):
            p = self.parties_per_const[const][i]
            if p in self.names_parties:
                votes_general[p] = abb.eval_add_protocol(votes_general[p], votes_const[i])
        return votes_general
    
    
    def get_empty_general(self, abb:ABB):
        if type(self.names_parties[0]) == int:
            return [abb.enc_zero for i in range(self.n_parties)]
        else:
            zeros = [abb.enc_zero for i in range(self.n_parties)]
            return dict(zip(self.names_parties,zeros))

    def serialize(self):
        serialized = {
            "n_parties": json.dumps(self.n_parties),
            "names_parties": json.dumps(self.names_parties),
            "n_const": json.dumps(self.n_const),
            "parties_per_const": json.dumps(self.parties_per_const),
        }
        log.info("serialized: " + str(serialized))
        return serialized
    
    @classmethod
    def deserialize(cls, serialized):
        n_parties = int(json.loads(serialized["n_parties"])),
        names_parties = json.loads(serialized["names_parties"]),
        n_const = json.loads(serialized["n_const"]),
        parties_per_const = json.loads(serialized["parties_per_const"])
        log.info("in des: " + str(n_parties))
        return PartiesAcrossConstituencies(n_parties, names_parties, n_const, parties_per_const)

class TwoVotesCandidatesSimplified():

    def __init__(self, direct_canidates:dict, list_candidates: dict):
        """
        Object to store all candidates, simplified assuming that there is a national list for each party.
        direct_candidates: dict which contains a dict for each party containing the index of the direct candidate for each constituency
        list_candidates: dict which contains a list of candidates (high-priority = low index) for each party
        """

        self.direct_candidates = direct_canidates
        self.list_candidates = list_candidates
    
    def get_index_direct_candidate(self, party, constituency):
        if party in self.direct_candidates:
            return (self.direct_candidates[party])[constituency]
        else:
            # should not happen, else error in ballot
            raise KeyError("This party has no direct candidate")

    def get_list_candidate(self, party, list_position): 
        """
        list positions from 0 to n-1
        """
        if party in self.list_candidates:
            return (self.list_candidates[party])[list_position]
        else:
            raise KeyError

    def get_empty_candidates_list(self, abb: ABB, party):
        """
        Returns a list with all list candidates as keys and enc_zero as value
        """
        return [abb.enc_zero for i in range(len(self.list_candidates[party]))]

    def serialize(self):
        serialized = {
            "direct_candidates": json.dumps(self.direct_candidates),
            "list_candidates": json.dumps(self.list_candidates)
        }
        return serialized
    
    @classmethod
    def deserialize(cls, serialized):
        direct_candidates = json.loads(serialized["direct_candidates"])
        list_candidates = json.loads(serialized["list_candidates"])
        return TwoVotesCandidatesSimplified(direct_candidates, list_candidates)

class TwoVotesCandidatesReal():

    def __init__(self, direct_canidates:dict, list_candidates: dict):
        """
        Object to store all candidates
        direct_candidates: dict which contains a dict for each party containing the index of the direct candidate for each constituency
        list_candidates: dict for each federal state of (dict which contains a dict for each a list of candidates (high-priority = low index) for each party)
        """

        self.direct_candidates = direct_canidates
        self.list_candidates = list_candidates
    
    def get_index_direct_candidate(self, party, constituency):
        if party in self.direct_candidates:
            return (self.direct_candidates[party])[constituency]
        else:
            # should not happen, else error in ballot
            raise KeyError("This party has no direct candidate")

    def get_list_candidate(self, state, party, list_position): 
        """
        list positions from 0 to n-1
        """
        if state in self.list_candidates: 
            if party in self.list_candidates[state]:
                return (self.list_candidates[state][party])[list_position]
            else:
                raise KeyError
        else:
            raise KeyError

    def get_empty_candidates_list(self, abb: ABB, state, party):
        """
        Returns a list with all list candidates as keys and enc_zero as value
        """
        return [abb.enc_zero for i in range(len(self.list_candidates[state][party]))]

    def serialize(self):
        serialized = {
            "direct_candidates": json.dumps(self.direct_candidates),
            "list_candidates": json.dumps(self.list_candidates)
        }
        return serialized
    
    @classmethod
    def deserialize(cls, serialized):
        direct_candidates = json.loads(serialized["direct_candidates"])
        list_candidates = json.loads(serialized["list_candidates"])
        return TwoVotesCandidatesReal(direct_candidates, list_candidates)

        