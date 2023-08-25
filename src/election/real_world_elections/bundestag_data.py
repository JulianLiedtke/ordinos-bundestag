import pandas as pd
import numpy as np
import os
from src.crypto.abb import ABB
import logging
from src.election.real_world_elections.helpers import PartiesAcrossConstituencies
log = logging.getLogger(__name__)

class Data2021():
    def get_primary_votes(abb: ABB):
        path = os.getcwd()
        path = os.path.join(path,'src')
        path = os.path.join(path,'election')
        path = os.path.join(path,'real_world_elections')
        input_file = os.path.join(path,'bt_2021_first_votes.csv')
        data = pd.read_csv(input_file)
        constituencies = (data['Nr'].to_numpy()).tolist()
        constituencies = [int(c)-1 for c in constituencies]
        first_votes = {}
        parties_list = list(data)
        to_remove= ['Unnamed: 0', 'Nr', 'Gebiet', 'gehört zu']
        for c in to_remove:
            parties_list.remove(c)
        n_parties = len(parties_list)
        # const i has index i-1

        const_state_mapping = (data['gehört zu'].to_numpy()).tolist()
        state_names = list(dict.fromkeys(const_state_mapping))  # remove all duplicates
        state_names.sort()
        n_constituencies = len(constituencies)
        n_valid_primary_votes = []
        parties_per_const = []
        for c in constituencies:
            valid_votes = 0
            votes_constituency = {}
            parties_constituency = []
            for p in parties_list:
                if not (data[p][c] == 0 or np.isnan(data[p][c])):                    
                    votes_constituency[p] = abb.enc_no_r(int(data[p][c]))
                    valid_votes = valid_votes + int(data[p][c])
                    parties_constituency.append(p)
            first_votes[c] = votes_constituency
            parties_per_const.append(parties_constituency)
            n_valid_primary_votes.append(valid_votes)
        num_parties = [len(p) for p in parties_per_const]
        all_parties = PartiesAcrossConstituencies(n_parties, parties_list, n_constituencies, parties_per_const)
        return first_votes, n_parties, n_constituencies, n_valid_primary_votes, all_parties, const_state_mapping, state_names, constituencies

    def get_secondary_votes(abb: ABB):
        path = os.getcwd()
        path = os.path.join(path,'src')
        path = os.path.join(path,'election')
        path = os.path.join(path,'real_world_elections')
        input_file = os.path.join(path,'bt_2021_secondary_votes.csv')
        data = pd.read_csv(input_file)

        states = (data['Nr'].to_numpy()).tolist()
        states.sort()
        secondary_votes = {}
        parties_list = list(data)
        to_remove= ['Unnamed: 0', 'Nr', 'Gebiet', 'gehört zu']
        for c in to_remove:
            parties_list.remove(c)
        n_valid_secondary_votes = []
        parties_per_state = []
        for c in states:
            c = c-1
            valid_votes = 0
            votes_state = {}
            parties_state = []
            for p in parties_list:
                if not (data[p][c] == 0 or np.isnan(data[p][c])):                    
                    votes_state[p] = abb.enc_no_r(int(data[p][c]))
                    valid_votes = valid_votes + int(data[p][c])
                    parties_state.append(p)
            secondary_votes[c] = votes_state
            parties_per_state.append(parties_state)
            n_valid_secondary_votes.append(valid_votes)
        total_valid_secondary_votes = sum(n_valid_secondary_votes)
        minority_parties = ["Südschleswigscher Wählerverband"]
        population_distribution = [2659792, 1532412, 1537766, 7207587, 548941, 2397701, 2056177, 2942960, 15415642, 3826905, 5222158, 1996822, 3610865, 11328866, 9313413, 865191]
        min_seats_contingent = 598
        return secondary_votes, n_valid_secondary_votes, parties_per_state, total_valid_secondary_votes, minority_parties, population_distribution, min_seats_contingent


    def preprocess_data(self):

        path = os.getcwd()
        path = os.path.join(path,'election')
        path = os.path.join(path,'real_world_elections')
        input_file = os.path.join(path,'bt_2021_full.csv')
        print(input_file)
        
        data = pd.read_csv(input_file, sep =";")

        # drop unused columns
        unused = ['Wahlberechtigte', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Wählende', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Ungültige Stimmen', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Gültige Stimmen', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18']
        for c in unused:
            del data[c]
        del data['Unnamed: 211']

        for c in list(data):
            if data[c][1] == 'Vorperiode':
                del data[c]
        data = data.drop(1)

        # first votes
        data_first_votes = data.copy()
        for c in list(data_first_votes):
            if data_first_votes[c][0] == 'Zweitstimmen':
                del data_first_votes[c]
        data_first_votes = data_first_votes.drop(0)
        

        # delete everything which isn't a constituency
        rows = data_first_votes.shape[0]
        data_first_votes = data_first_votes.drop(rows+1)
        for i in range(2, rows+1):
            if int(data_first_votes['gehört zu'][i]) == 99.0:
                data_first_votes = data_first_votes.drop(i)

        data_first_votes = data_first_votes.astype({'gehört zu': int})

        input_file = os.path.join(path,'bt_2021_first_votes.csv')  
        data_first_votes.to_csv(input_file)  


        # secondary_votes
        data_secondary_votes = data.copy()
        columns = list(data_secondary_votes)
        for i in range(len(columns)):
            if data_secondary_votes[columns[i]][0] == 'Zweitstimmen':
                name = columns[i-1]
                del data_secondary_votes[columns[i-1]]
                data_secondary_votes = data_secondary_votes.rename(columns={columns[i]: name})
                
        
        
        data_secondary_votes = data_secondary_votes.drop(0)
        
        
        # delete everything which isn't a federal state
        rows = data_secondary_votes.shape[0]
        data_secondary_votes = data_secondary_votes.drop(rows+1)
        for i in range(2, rows+1):
            if not int(data_secondary_votes['gehört zu'][i]) == 99.0:
                data_secondary_votes = data_secondary_votes.drop(i)

        data_secondary_votes = data_secondary_votes.astype({'gehört zu': int})
        data_secondary_votes = data_secondary_votes.astype({'Nr  ': int})

        input_file = os.path.join(path,'bt_2021_secondary_votes.csv')  
        data_secondary_votes.to_csv(input_file)  



  
"""
d = Data2021()
d.get_erststimmen()
"""