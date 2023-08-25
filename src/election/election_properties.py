import abc
import logging
from datetime import datetime
from src.util.utils import get_class_from_name, subclasses_recursive
from time import time

from src.util.csv_writer import CSV_Writer
from src.util.point_vote import IllegalVoteException
from src.election.ballotformat import BallotFormat
from src.election.tie_breaking.tie_breaking import TieBreaking

log = logging.getLogger(__name__)


class ElectionProperties(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, candidates, ballot_format=None, start_date=None, due_date=None, title="", expected_votes_n=None, use_constituencies = False, break_ties = False, max_ties = 0, tie_breaking_iterator = None,
                 logger_name_extension="", peer_server_address="", authentication_server_address="", bulletin_board_address=""):
        """candidates could either be the amount of cands or a list of names"""
        if isinstance(candidates, list):
            self.n_cand = len(candidates)
            self.candidate_names = candidates
        else:
            self.n_cand = candidates
            self.candidate_names = [i for i in range(self.n_cand)]
        self.system_name = str(self.__class__.__name__) + ("-" if logger_name_extension != "" else "") + logger_name_extension
        self.expected_votes_n = expected_votes_n
        self.ballot_format = ballot_format
        self.start_date = start_date
        self.due_date = due_date
        self.title = title
        self.peer_server_address = peer_server_address
        self.authentication_server_address = authentication_server_address
        self.bulletin_board_address = bulletin_board_address
        self.use_constituencies = use_constituencies
        self.break_ties = break_ties
        self.max_ties = max_ties
        self.tie_breaking_iterator = tie_breaking_iterator

    def serialize(self):
        serialized = {
            "candidates": self.candidate_names,
            "ballot": self.ballot_format.serialize() if self.ballot_format else None,
            "evaluation": {
                "type": self.__class__.__name__,
                "settings": {
                    "use_constituencies": self.use_constituencies,
                    "break_ties": self.break_ties,
                    "tie_breaking_iterator": None if self.tie_breaking_iterator is None else self.tie_breaking_iterator.serialize()
                }
            },
            "election": {
                "title": self.title,
                "start": self.start_date.isoformat() if self.start_date else None,
                "due": self.due_date.isoformat() if self.due_date else None,
            },
            "communication": {
                "peerServerIP": self.peer_server_address,
                "authenticationServerIP": self.authentication_server_address,
                "bulletinBoardServerIP": self.bulletin_board_address
            }
        }
        if self.expected_votes_n:
            serialized["evaluation"]["settings"]["expectedVoteCount"] = self.expected_votes_n
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        subclass = get_class_from_name(serialized["evaluation"]["type"], subclasses_recursive(cls))
        args = subclass.serialized_to_args(serialized)
        no_none_args = dict(filter(lambda x: x[1] is not None, args.items()))
        return subclass(**no_none_args)

    @classmethod
    def serialized_to_args(cls, serialized):
        args = {
            "candidates": serialized["candidates"],
            "ballot_format": BallotFormat.deserialize(serialized.get("ballot")),
            "start_date": datetime.fromisoformat(serialized["election"]["start"]) if serialized["election"].get("start", None) is not None else None,
            "due_date": datetime.fromisoformat(serialized["election"]["due"]) if serialized["election"].get("due", None) is not None else None,
            "title": serialized["election"].get("title"),
            "expected_votes_n": serialized["evaluation"]["settings"].get("expectedVoteCount"),
            "use_constituencies": serialized["evaluation"]["settings"].get("use_constituencies"),
            "break_ties": serialized["evaluation"]["settings"].get("break_ties"),
            "tie_breaking_iterator": TieBreaking.deserialize(serialized["evaluation"]["settings"].get("tie_breaking_iterator")),
            "peer_server_address": serialized["communication"].get("peerServerIP"),
            "authentication_server_address": serialized["communication"].get("authenticationServerIP"),
            "bulletin_board_address": serialized["communication"].get("bulletinBoardServerIP"),
        }
        return args

    @property
    @abc.abstractmethod
    def aggregator(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_evaluator(self, n_votes, abb):
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_valid_vote(self, generic_vote, abb):
        raise NotImplementedError()

    @abc.abstractmethod
    def serialize_vote(self, valid_vote):
        raise NotImplementedError()

    @abc.abstractmethod
    def deserialize_vote(self, serialized_vote, abb):
        raise NotImplementedError()

    def setup(self, abb):
        pass
