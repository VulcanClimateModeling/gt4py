from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from eve import NodeVisitor
from eve.utils import XIterator
from gt4py.definitions import Extent
from gtc import gtir
from gtc.common import HorizontalInterval, LevelMarker


def _iter_field_names(node: Union[gtir.Stencil, gtir.ParAssignStmt]) -> XIterator[gtir.FieldAccess]:
    return node.iter_tree().if_isinstance(gtir.FieldDecl).getattr("name").unique()


def _iter_assigns(node: gtir.Stencil) -> XIterator[gtir.ParAssignStmt]:
    return node.iter_tree().if_isinstance(gtir.ParAssignStmt)


def _ext_from_off(offset: gtir.CartesianOffset) -> Extent:
    return Extent(
        ((min(offset.i, 0), max(offset.i, 0)), (min(offset.j, 0), max(offset.j, 0)), (0, 0))
    )


def _overlap_with_extent(
    interval: HorizontalInterval, axis_extent: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    """Return a tuple of the distances to the edge of the compute domain, if overlapping."""
    if interval.start.level == LevelMarker.START:
        start_diff = axis_extent[0] - interval.start.offset
    else:
        start_diff = None

    if interval.end.level == LevelMarker.END:
        end_diff = axis_extent[1] - interval.end.offset
    else:
        end_diff = None

    if start_diff is not None and start_diff > 0 and end_diff is None:
        if interval.end.offset <= axis_extent[0]:
            return None
    elif end_diff is not None and end_diff < 0 and start_diff is None:
        if interval.start.offset > axis_extent[1]:
            return None

    start_diff = min(start_diff, 0) if start_diff is not None else 0
    end_diff = max(end_diff, 0) if end_diff is not None else 0
    return (start_diff, end_diff)


def _compute_extent_diff(intervals: Dict[str, HorizontalInterval]) -> Optional[Extent]:
    parallel_axes_names = sorted(intervals.keys())
    compute_extent = Extent.zeros()
    diffs = []

    for axis, extent in zip(parallel_axes_names, compute_extent):
        diff = _overlap_with_extent(intervals[axis], extent)
        if diff:
            diffs.append(diff)
            continue
        return None

    return Extent(diffs + [(0, 0)])


FIELD_EXT_T = Dict[str, Extent]


class LegacyExtentsVisitor(NodeVisitor):
    @dataclass
    class AssignContext:
        left_extent: Extent
        assign_extents: FIELD_EXT_T = field(default_factory=dict)

    @dataclass
    class StencilContext:
        assign_conditions: Dict[int, List[gtir.FieldAccess]] = field(default_factory=dict)

    def visit_Stencil(self, node: gtir.Stencil, **kwargs: Any) -> FIELD_EXT_T:
        field_extents = {name: Extent.zeros() for name in _iter_field_names(node)}
        ctx = self.StencilContext()
        for field_if in node.iter_tree().if_isinstance(gtir.FieldIfStmt):
            self.visit(field_if, ctx=ctx)
        # for horizontal_region in node.iter_tree().if_isinstance(gtir.HorizontalRegion):
        #     self.visit(horizontal_region, ctx=ctx, field_extents=field_extents)
        for assign in reversed(_iter_assigns(node).to_list()):
            self.visit(assign, ctx=ctx, field_extents=field_extents)
        return field_extents

    def visit_ParAssignStmt(
        self,
        node: gtir.ParAssignStmt,
        *,
        ctx: StencilContext,
        field_extents: FIELD_EXT_T,
        **kwargs: Any,
    ) -> None:
        self._visit_assign(node, ctx=ctx, field_extents=field_extents, **kwargs)

    def visit_SerialAssignStmt(
        self,
        node: gtir.SerialAssignStmt,
        *,
        ctx: StencilContext,
        field_extents: FIELD_EXT_T,
        **kwargs: Any,
    ) -> None:
        self._visit_assign(node, ctx=ctx, field_extents=field_extents, **kwargs)

    def _visit_assign(
        self,
        node: Union[gtir.ParAssignStmt, gtir.SerialAssignStmt],
        *,
        ctx: StencilContext,
        field_extents: FIELD_EXT_T,
        **kwargs: Any,
    ):
        left_extent = field_extents.setdefault(node.left.name, Extent.zeros())
        pa_ctx = self.AssignContext(left_extent=left_extent)
        self.visit(
            ctx.assign_conditions.get(id(node), []),
            field_extents=field_extents,
            pa_ctx=pa_ctx,
            **kwargs,
        )
        self.visit(node.right, field_extents=field_extents, pa_ctx=pa_ctx, **kwargs)
        for key, value in pa_ctx.assign_extents.items():
            field_extents[key] |= value

    def visit_FieldIfStmt(
        self, node: gtir.FieldIfStmt, *, ctx: StencilContext, **kwargs: Any
    ) -> None:
        for assign_id in node.iter_tree().if_isinstance(gtir.ParAssignStmt).map(id):
            ctx.assign_conditions.setdefault(assign_id, []).extend(
                node.cond.iter_tree().if_isinstance(gtir.FieldAccess).to_list()
            )

    def visit_HorizontalRegion(
        self,
        node: gtir.HorizontalRegion,
        *,
        field_extents: FIELD_EXT_T,
        ctx: StencilContext,
        **kwargs: Any,
    ) -> None:
        block = node.block
        mask = node.mask
        horizontal_extent = _compute_extent_diff({"I": mask.i, "J": mask.j})

        field_accesses = block.iter_tree().if_isinstance(gtir.FieldAccess).to_list()
        for assign_id in (
            block.iter_tree().if_isinstance(gtir.ParAssignStmt, gtir.SerialAssignStmt).map(id)
        ):
            ctx.assign_conditions.setdefault(assign_id, []).extend(field_accesses)

        for field_access in field_accesses:
            if field_access.name in field_extents:
                field_extents[field_access.name] |= horizontal_extent

    def visit_FieldAccess(
        self,
        node: gtir.FieldAccess,
        *,
        field_extents: FIELD_EXT_T,
        pa_ctx: AssignContext,
        **kwargs: Any,
    ) -> None:
        pa_ctx.assign_extents.setdefault(
            node.name, field_extents.setdefault(node.name, Extent.zeros())
        )
        pa_ctx.assign_extents[node.name] |= pa_ctx.left_extent + _ext_from_off(node.offset)


def compute_legacy_extents(node: gtir.Stencil) -> FIELD_EXT_T:
    return LegacyExtentsVisitor().visit(node)
