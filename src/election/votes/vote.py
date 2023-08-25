from abc import abstractmethod
import logging

log = logging.getLogger(__name__)


class ConstituencyVote():
    """
    Abstract class for votes with a constituency attribute
    """
    @abstractmethod
    def generate_random_votes(n_votes, n_cand, n_constituencies):
        raise NotImplementedError

    @abstractmethod
    def generate_random_choice(n_cand):
        raise NotImplementedError

    @abstractmethod
    def generate_random_constituency(n_constituencies):
        raise NotImplementedError

    def __init__(self, choices, constituency, type):
        self.constituency = constituency
        self.choices = choices
        self.type = type

    def getConstitency(self):
        return self.constituency

    def getChoices(self):
        return self.choices

    def setChoices(self, choices):
        self.choices = choices
    
    def getType(self):
        return self.type

    def __repr__(self):
        return "\nconstituency: "+ str(self.constituency) + ", choices: " + str(self.choices) + ", type: " + str(self.type)
