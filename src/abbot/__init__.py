"""abbot — an active-learning-for-labs layer on top of BoTorch.

From A to B: a Bayesian guide for lab experiments. abbot is NOT a competing
optimization framework; it is the composite-SBI + lab-experiment-orchestration
layer on top of botorch (which already provides the surrogate-agnostic
Model/Posterior layer, the BO/US loop primitives, async/batch, and
serialization). If it drifts into "framework", stop.
"""

from abbot.experiment import ManualExperiment, SyntheticExperiment
from abbot.goal import (
    BandIntersection,
    FunctionGoal,
    Goal,
    LevelSet,
    PercentileSet,
    TopK,
    Wishlist,
    goals_from_spec,
)
from abbot.history import DataStore, History, Record
from abbot.loop import run_campaign
from abbot.model import (
    EnsembleConstructor,
    ModelConstructor,
    ModelListConstructor,
    SingleTaskGPConstructor,
    default_constructor,
)
from abbot.policy import (
    BAXPolicy,
    ParetoBO,
    Policy,
    SimpleBO,
    UncertaintySampling,
    default_policy,
)
from abbot.problem import (
    Box,
    ContinuousDomain,
    EnumeratedDomain,
    InequalitySpec,
    ObjectiveSpec,
    OutputSpec,
    ProblemSpec,
    StructuredDomain,
)
from abbot.reduce import BoxMembershipReducer, ObjectiveReducer
from abbot.sbi import sbi_evaluate
from abbot.session import (
    Campaign,
    CampaignResult,
    CampaignState,
    default_initial_designs,
)

__version__ = "0.1.0"

__all__ = [
    "BAXPolicy",
    "BandIntersection",
    "Box",
    "BoxMembershipReducer",
    "Campaign",
    "CampaignResult",
    "CampaignState",
    "ContinuousDomain",
    "DataStore",
    "EnsembleConstructor",
    "EnumeratedDomain",
    "FunctionGoal",
    "Goal",
    "History",
    "InequalitySpec",
    "LevelSet",
    "ManualExperiment",
    "ModelConstructor",
    "ModelListConstructor",
    "ObjectiveReducer",
    "ObjectiveSpec",
    "OutputSpec",
    "ParetoBO",
    "PercentileSet",
    "Policy",
    "ProblemSpec",
    "Record",
    "SimpleBO",
    "SingleTaskGPConstructor",
    "StructuredDomain",
    "SyntheticExperiment",
    "TopK",
    "UncertaintySampling",
    "Wishlist",
    "__version__",
    "default_constructor",
    "default_initial_designs",
    "default_policy",
    "goals_from_spec",
    "run_campaign",
    "sbi_evaluate",
]
