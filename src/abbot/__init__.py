"""abbot — an active-learning-for-labs layer on top of BoTorch.

From A to B: a Bayesian guide for lab experiments. abbot is NOT a competing
optimization framework; it is the composite-SBI + lab-experiment-orchestration
layer on top of botorch (which already provides the surrogate-agnostic
Model/Posterior layer, the BO/US loop primitives, async/batch, and
serialization). If it drifts into "framework", stop.
"""

from abbot.composite import (
    ObjectiveReducer,
    Rescore,
    composite_evaluate,
)
from abbot.cost import (
    CallableCost,
    CostModel,
    FittedCost,
    FixedCost,
    PowerLawCost,
    SourceCost,
)
from abbot.experiment import (
    BatchExperiment,
    Experiment,
    ManualExperiment,
    MockAnalyticExperiment,
    MockDatasetExperiment,
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
    MultiFidelityGPConstructor,
    MultiSourceGPConstructor,
    MultiTaskGPConstructor,
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
    IntegerDomain,
    LinearConstraint,
    ObjectiveSpec,
    OutputSpec,
    ProblemSpec,
    Source,
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
from abbot.supervision import Supervisor

__version__ = "0.2.1"

__all__ = [
    "BAXPolicy",
    "BandIntersection",
    "BatchExperiment",
    "Box",
    "CallableCost",
    "Campaign",
    "CampaignResult",
    "CampaignState",
    "Candidate",
    "ContinuousDomain",
    "CostModel",
    "DataStore",
    "EnsembleConstructor",
    "EnumeratedDomain",
    "Experiment",
    "FittedCost",
    "FixedCost",
    "FunctionGoal",
    "Goal",
    "History",
    "InequalitySpec",
    "IntegerDomain",
    "LevelSet",
    "LinearConstraint",
    "ManualExperiment",
    "MockAnalyticExperiment",
    "MockDatasetExperiment",
    "ModelConstructor",
    "ModelListConstructor",
    "MultiFidelityGPConstructor",
    "MultiSourceGPConstructor",
    "MultiTaskGPConstructor",
    "ObjectiveReducer",
    "ObjectiveSpec",
    "OutputSpec",
    "ParetoBO",
    "PercentileSet",
    "PersistentFaultError",
    "Policy",
    "PosteriorSampler",
    "PowerLawCost",
    "ProblemSpec",
    "Record",
    "Rescore",
    "ResumeMismatchError",
    "SimpleBO",
    "SingleTaskGPConstructor",
    "Source",
    "SourceCost",
    "StructuredDomain",
    "Supervisor",
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
