from src.util.utils import get_class_from_name, subclasses_recursive


class BallotFormat:
    def __init__(self, category, **kwargs):
        self.category = category
        self.kwargs = kwargs

    def serialize(self):
        serialized = {"category": self.category}
        serialized.update(self.kwargs)
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        if serialized is None:
            serialized = {}
        try:
            category = serialized.setdefault("category", "")
            found_class = get_class_from_name(f"{category}BallotFormat", subclasses_recursive(cls))
            return found_class(**serialized.get("settings", {}))
        except ValueError:
            pass
        return cls(**serialized)

class RatingBallotFormat(BallotFormat):
    def __init__(self, points, allowDraw,
                 minNumberOfChoices, spreadPointsAcross, **kwargs):
        BallotFormat.__init__(self, "Rating")
        self.points = points
        self.allowDraw = allowDraw
        self.minNumberOfChoices = minNumberOfChoices
        self.spreadPointsAcross = spreadPointsAcross

    def serialize(self):
        serialized = super().serialize()
        serialized["settings"] = {"points": self.points,
                                  "allowDraw": self.allowDraw,
                                  "minNumberOfChoices": self.minNumberOfChoices,
                                  "spreadPointsAcross": self.spreadPointsAcross }
        return serialized


class RankingBallotFormat(BallotFormat):
    def __init__(self, numberOfEqualRanks,
                 numberOfCountingRanks, pointDistribution, **kwargs):
        BallotFormat.__init__(self, "Ranking")
        self.numberOfEqualRanks = numberOfEqualRanks
        self.numberOfCountingRanks = numberOfCountingRanks
        self.pointDistribution = pointDistribution

    def serialize(self):
        serialized = super().serialize()
        serialized["settings"] = {"numberOfEqualRanks": self.numberOfEqualRanks,
                                  "numberOfCountingRanks": self.numberOfCountingRanks,
                                  "pointDistribution": self.pointDistribution }
        return serialized


class CheckboxBallotFormat(BallotFormat):
    def __init__(self, minNumberOfChoices, maxNumberOfChoices, **kwargs):
        BallotFormat.__init__(self, "Checkbox")
        self.minNumberOfChoices = minNumberOfChoices
        self.maxNumberOfChoices = maxNumberOfChoices

    def serialize(self):
        serialized = super().serialize()
        serialized["settings"] = {"minNumberOfChoices": self.minNumberOfChoices,
                                  "maxNumberOfChoices": self.maxNumberOfChoices }
        return serialized
