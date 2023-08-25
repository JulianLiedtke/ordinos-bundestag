from src.election.condorcet.condorcet_no_winner_evaluations import CopelandEvaluationFast, CopelandEvaluationSafe, MiniMaxMarginsEvaluation, MiniMaxWinningVotesEvaluation, SmithEvaluation, SmithFastEvaluation
from src.election.condorcet.condorcet_election_system import Condorcet


class MiniMaxMarginsSmith(Condorcet):
    def __init__(self, candidates, leak_better_half=False, smith_leak_min_copeland=False, **base_settings):
        base_settings.update({
            "leak_better_half": leak_better_half
        })
        if smith_leak_min_copeland:
            base_settings["additional_evaluators"] = [SmithFastEvaluation, MiniMaxMarginsEvaluation]
        else:
            base_settings["additional_evaluators"] = [SmithEvaluation, MiniMaxMarginsEvaluation]
        super().__init__(candidates, **base_settings)


class MiniMaxWinningVotesSmith(Condorcet):
    def __init__(self, candidates, leak_better_half=False, smith_leak_min_copeland=False, **base_settings):
        base_settings.update({
            "leak_better_half": leak_better_half
        })
        if smith_leak_min_copeland:
            base_settings["additional_evaluators"] = [SmithFastEvaluation, MiniMaxWinningVotesEvaluation]
        else:
            base_settings["additional_evaluators"] = [SmithEvaluation, MiniMaxWinningVotesEvaluation]
        super().__init__(candidates, **base_settings)


class Copeland(Condorcet):
    def __init__(self, candidates, leak_better_half=False, leak_max_points=False,
                 evaluate_condorcet=False, **base_settings):
        base_settings.update({
            "leak_better_half": leak_better_half,
            "evaluate_condorcet": evaluate_condorcet
        })
        if leak_max_points:
            base_settings["additional_evaluators"] = [CopelandEvaluationFast]
        else:
            base_settings["additional_evaluators"] = [CopelandEvaluationSafe]
        super().__init__(candidates, **base_settings)
