"""abbot — an active-learning-for-labs layer on top of BoTorch.

From A to B: a Bayesian guide for lab experiments. abbot is NOT a competing
optimization framework; it is the composite-SBI + lab-experiment-orchestration
layer on top of botorch (which already provides the surrogate-agnostic
Model/Posterior layer, the BO/US loop primitives, async/batch, and
serialization). If it drifts into "framework", stop.
"""

from abbot.experiment import ManualExperiment, SyntheticExperiment
from abbot.history import DataStore, History, Record
from abbot.loop import run_campaign
from abbot.model import (
    ModelConstructor,
    ModelListConstructor,
    SingleTaskGPConstructor,
    default_constructor,
)
from abbot.policy import ParetoBO, Policy, SimpleBO, UncertaintySampling
from abbot.problem import (
    Box,
    ContinuousDomain,
    EnumeratedDomain,
    InequalitySpec,
    ObjectiveSpec,
    ProblemSpec,
    StructuredDomain,
)
from abbot.session import Campaign, CampaignResult, CampaignState

__version__ = "0.1.0"

__all__ = [
    "Box",
    "Campaign",
    "CampaignResult",
    "CampaignState",
    "ContinuousDomain",
    "DataStore",
    "EnumeratedDomain",
    "History",
    "InequalitySpec",
    "ManualExperiment",
    "ModelConstructor",
    "ModelListConstructor",
    "ObjectiveSpec",
    "ParetoBO",
    "Policy",
    "ProblemSpec",
    "Record",
    "SimpleBO",
    "SingleTaskGPConstructor",
    "StructuredDomain",
    "SyntheticExperiment",
    "UncertaintySampling",
    "__version__",
    "default_constructor",
    "run_campaign",
]
