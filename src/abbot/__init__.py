"""abbot — an active-learning-for-labs layer on top of BoTorch.

From A to B: a Bayesian guide for lab experiments. abbot is NOT a competing
optimization framework; it is the composite-SBI + lab-experiment-orchestration
layer on top of botorch (which already provides the surrogate-agnostic
Model/Posterior layer, the BO/US loop primitives, async/batch, and
serialization). If it drifts into "framework", stop.
"""

from abbot.composite import (
    BoxMembershipReducer,
    ObjectiveReducer,
    Rescore,
    composite_evaluate,
)
from abbot.experiment import (
    Experiment,
    ManualExperiment,
    Supervisor,
    SyntheticExperiment,
    TensorExperiment,
    Verdict,
)
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
from abbot.history import DataStore, History, Record, ResumeMismatchError
from abbot.interop.sbi import PosteriorSampler, TrainedPosterior, sbi_evaluate
from abbot.loop import drive, run_campaign
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
from abbot.session import (
    Campaign,
    CampaignResult,
    CampaignState,
    Candidate,
    PersistentFaultError,
    Targets,
    apply_targets,
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
    "Candidate",
    "ContinuousDomain",
    "DataStore",
    "EnsembleConstructor",
    "EnumeratedDomain",
    "Experiment",
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
    "PersistentFaultError",
    "Policy",
    "PosteriorSampler",
    "ProblemSpec",
    "Record",
    "Rescore",
    "ResumeMismatchError",
    "SimpleBO",
    "SingleTaskGPConstructor",
    "StructuredDomain",
    "Supervisor",
    "SyntheticExperiment",
    "Targets",
    "TensorExperiment",
    "TopK",
    "TrainedPosterior",
    "UncertaintySampling",
    "Verdict",
    "Wishlist",
    "__version__",
    "apply_targets",
    "composite_evaluate",
    "default_constructor",
    "default_initial_designs",
    "default_policy",
    "drive",
    "goals_from_spec",
    "run_campaign",
    "sbi_evaluate",
]
