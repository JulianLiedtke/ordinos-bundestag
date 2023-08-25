import enum
from itertools import count
from numpy import array
from numpy.random import default_rng
from src.crypto.abb import ABB, Ciphertext
from src.crypto.paillier_abb import PaillierABB, PaillierCiphertext
import logging
import json
log = logging.getLogger(__name__)

class BreakingTypes(enum.Enum):
    random = 0
    # TODO: add other breaking types and support for all breaking types below
    age = 1


class TieBreaking():
    """
    This class precalculates arrays of permutations of the numbers 0, ..., n_cand-1 to help breaking ties.
    """
    def __init__(self,  max_ties, abb: ABB = None):
        self.max_ties = max_ties
        self.all_arrays = {}
        self.all_arrays_serialized = {}
        self.counter = None
        self.abb = abb

    def setup_arrays(self, abb: ABB, n_cand, breaking_type, plus1 = False):
        self.abb = abb
      #  self.n_cand = n_cand
        self.breaking_type = breaking_type
        if self.breaking_type == BreakingTypes.random:
            self.setup_arrays_random(n_cand, plus1)
        else:
            raise NotImplementedError()

    def setup_multiple_arrays(self, abb: ABB, numbers: list, breaking_type):
        """
        Setup arrays for multiple candidate numbers.
        """
        self.numbers = numbers
        #self.counter = [0 for i in numbers]
        self.counter = {n:0 for n in numbers}
        for n in numbers:
            self.all_arrays[n] = []
            self.setup_arrays(abb, n, breaking_type)

    
    def setup_arrays_random(self, n_cand, plus1):
        # TODO: this setup can only be executed by a honest party --> add support for distributed system

        # TODO: the seed is necessary so that both trustees have to same values for randomize
        # --> setup for the tie-breaking has to be moved somewhere up the setup phase before everything is executed parallel by different trustees
        seed = 4
        rng = default_rng(seed)
        if plus1: 
            sequence = [i+1 for i in range(n_cand)]
        else:
            sequence = [i for i in range(n_cand)]
        
        for i in range(self.max_ties):
            perm = (rng.permutation(sequence)).tolist()
            perm_enc = [self.abb.enc_no_r(perm[j]) for j in range(n_cand)]
            self.all_arrays[n_cand].append(perm_enc)
                 
    
    def has_next(self, n):
        try:
            return self.counter[n] < self.max_ties
        except:
            raise ValueError("error in tie breaking has_next, value = " + str(n))
    
    def next(self, n):
        try:
            perm = (self.all_arrays[n])[self.counter[n]]
            self.counter[n] += 1
            return perm
        except:
            raise ValueError("no further tie breaker available --> you should always check with hasNext, value = " + str(n))
        

    def serialize(self):
        serialized_arrays = {}
        for n in self.numbers:
            serialized_arrays[n] = [self.serialize_permutation((self.all_arrays[n])[i]) for i in range(len(self.all_arrays[n]))]
        serialized = {
            "numbers": self.numbers,
            "counter": self.counter,
            "max_ties": self.max_ties,
           # "all_arrays": [self.serialize_permutation(self.all_arrays[i]) for i in range(len(self.all_arrays))],
            "all_arrays": json.dumps(serialized_arrays),
        }
        return serialized
    
    @classmethod
    def deserialize(cls, serialized):
        
        if serialized is None:
            return None
        numbers = serialized["numbers"]
        counter = serialized["counter"]
        # convert keys to integers:
        counter =  {int(key):value for key,value in counter.items()}
        max_ties = serialized["max_ties"]
        arrays_dict = json.loads(serialized["all_arrays"])
        # convert keys to integers:
        arrays_dict =  {int(key):value for key,value in arrays_dict.items()}

        for n in numbers:
            arrays_n = []
            for serialized_array in arrays_dict[n]:
                arrays_n.append(serialized_array["elements"])
            arrays_dict[n] = arrays_n
        #all_arrays_serialized = []
        #for serialized_array in serialized["all_arrays"]:
        #    all_arrays_serialized.append(serialized_array["elements"])
        obj = TieBreaking(max_ties)
        obj.numbers = numbers
        obj.counter = counter
        obj.all_arrays_serialized = arrays_dict
        #obj.all_arrays_serialized = all_arrays_serialized
        return obj
    
    def serialize_permutation(self, permutation):
        serialized_permutation = {
            "elements": [
               element.serialize() for element in permutation
            ]
        }
        return serialized_permutation

    def deserialize_arrays(self, abb: ABB):
        all_arrays = {}
        for n in self.numbers:
            all_arrays[n] = []
            for serialized_perm in self.all_arrays_serialized[n]:
                perm = []
                for p in serialized_perm:
                    perm.append(abb.deserialize_cipher(p))
                all_arrays[n].append(perm)
        self.all_arrays = all_arrays


        #all_arrays = []
        #for serialized_permutation in self.all_arrays_serialized:
        #    perm = []
        #    for p in serialized_permutation:
        #        perm.append(abb.deserialize_cipher(p))
        #    all_arrays.append(perm)
        #self.all_arrays = all_arrays
        return all_arrays