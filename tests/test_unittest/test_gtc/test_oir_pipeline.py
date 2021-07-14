from gtc.passes.oir_optimizations.horizontal_execution_merging import GreedyMerging
from gtc.passes.oir_pipeline import OirPipeline, hash_step

from .oir_utils import StencilFactory


def test_no_skipping():
    pipeline = OirPipeline(StencilFactory())
    pipeline.full()

    steps = tuple(hash_step(i) for i in pipeline.steps())

    assert steps in pipeline._cache


def test_skip_one():
    pipeline = OirPipeline(StencilFactory())
    pipeline.full(skip=[GreedyMerging.visit])

    steps = tuple(hash_step(i) for i in pipeline.steps())
    skipped = tuple(i for i in steps if i != hash_step(GreedyMerging.visit))

    assert steps not in pipeline._cache
    assert skipped in pipeline._cache