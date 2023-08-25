import abc
import logging
from src.util.utils import get_class_from_name, subclasses_recursive
from time import time
import sys

log = logging.getLogger(__name__)


class BaseVoteAggregation:
    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def get_initial_vote_aggregation(abb, n_cand):
        return []

    @staticmethod
    @abc.abstractmethod
    def aggregate_vote(vote_aggregation, new_vote):
        pass
    
    @classmethod
    def serialize(cls):
        return cls.__name__

    @classmethod
    def deserialize(cls, s):
        return get_class_from_name(s, subclasses_recursive(cls))
