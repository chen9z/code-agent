from __future__ import annotations

import asyncio
import copy
import time
import warnings
from typing import Any, Dict, Iterable, Optional


class BaseNode:
    """Minimal synchronous node with lifecycle hooks."""

    def __init__(self) -> None:
        self.params: Dict[str, Any] = {}
        self.successors: Dict[str, "BaseNode"] = {}

    def set_params(self, params: Dict[str, Any]) -> None:
        self.params = params

    def next(self, node: "BaseNode", action: str = "default") -> "BaseNode":
        if action in self.successors:
            warnings.warn(f"Overwriting successor for action '{action}'", stacklevel=2)
        self.successors[action] = node
        return node

    def prep(self, shared: Dict[str, Any]) -> Any:  # pragma: no cover - default hook
        return shared

    def exec(self, prep_res: Any) -> Any:  # pragma: no cover - default hook
        return prep_res

    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Any) -> Any:
        return exec_res

    def _exec(self, prep_res: Any) -> Any:
        return self.exec(prep_res)

    def _run(self, shared: Dict[str, Any]) -> Any:
        prep_res = self.prep(shared)
        exec_res = self._exec(prep_res)
        return self.post(shared, prep_res, exec_res)

    def run(self, shared: Dict[str, Any]) -> Any:
        if self.successors:
            warnings.warn("Node won't run successors. Use Flow.", stacklevel=2)
        return self._run(shared)

    def __rshift__(self, other: "BaseNode") -> "BaseNode":
        return self.next(other)

    def __sub__(self, action: str) -> "_ConditionalTransition":
        if not isinstance(action, str):
            raise TypeError("Action must be a string")
        return _ConditionalTransition(self, action)


class _ConditionalTransition:
    def __init__(self, src: BaseNode, action: str) -> None:
        self.src = src
        self.action = action

    def __rshift__(self, tgt: BaseNode) -> BaseNode:
        return self.src.next(tgt, self.action)


class Node(BaseNode):
    def __init__(self, max_retries: int = 1, wait: float = 0.0) -> None:
        super().__init__()
        self.max_retries = max_retries
        self.wait = wait

    def exec_fallback(self, prep_res: Any, exc: Exception) -> Any:
        raise exc

    def _exec(self, prep_res: Any) -> Any:
        for self.cur_retry in range(self.max_retries):
            try:
                return self.exec(prep_res)
            except Exception as exc:  # pragma: no cover - retry path
                if self.cur_retry == self.max_retries - 1:
                    return self.exec_fallback(prep_res, exc)
                if self.wait > 0:
                    time.sleep(self.wait)
        raise RuntimeError("Node retry loop exited unexpectedly")


class BatchNode(Node):
    def _exec(self, items: Iterable[Any]) -> Any:
        return [super()._exec(item) for item in items or []]


class Flow(BaseNode):
    def __init__(self, start: Optional[BaseNode] = None) -> None:
        super().__init__()
        self.start_node = start

    def start(self, start: BaseNode) -> BaseNode:
        self.start_node = start
        return start

    def get_next_node(self, curr: BaseNode, action: Optional[str]) -> Optional[BaseNode]:
        action_key = action or "default"
        nxt = curr.successors.get(action_key)
        if not nxt and curr.successors:
            warnings.warn(
                f"Flow ends: '{action_key}' not found in {list(curr.successors)}",
                stacklevel=2,
            )
        return nxt

    def _orch(self, shared: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Any:
        if not self.start_node:
            raise RuntimeError("Flow has no start node configured")
        current = copy.copy(self.start_node)
        merged_params = params or {**self.params}
        last_action = None
        while current:
            current.set_params(merged_params)
            last_action = current._run(shared)
            current = copy.copy(self.get_next_node(current, last_action))
        return last_action

    def _run(self, shared: Dict[str, Any]) -> Any:
        prep_res = self.prep(shared)
        exec_res = self._orch(shared)
        return self.post(shared, prep_res, exec_res)

    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Any) -> Any:
        return exec_res


class BatchFlow(Flow):
    def _run(self, shared: Dict[str, Any]) -> Any:
        prep_results = self.prep(shared) or []
        for batch_params in prep_results:
            merged = {**self.params, **batch_params}
            self._orch(shared, merged)
        return self.post(shared, prep_results, None)


class AsyncNode(Node):
    async def prep_async(self, shared: Dict[str, Any]) -> Any:  # pragma: no cover - default hook
        return shared

    async def exec_async(self, prep_res: Any) -> Any:  # pragma: no cover - default hook
        return prep_res

    async def exec_fallback_async(self, prep_res: Any, exc: Exception) -> Any:
        raise exc

    async def post_async(self, shared: Dict[str, Any], prep_res: Any, exec_res: Any) -> Any:
        return exec_res

    async def _exec(self, prep_res: Any) -> Any:
        for self.cur_retry in range(self.max_retries):
            try:
                return await self.exec_async(prep_res)
            except Exception as exc:  # pragma: no cover - retry path
                if self.cur_retry == self.max_retries - 1:
                    return await self.exec_fallback_async(prep_res, exc)
                if self.wait > 0:
                    await asyncio.sleep(self.wait)
        raise RuntimeError("Async node retry loop exited unexpectedly")

    async def run_async(self, shared: Dict[str, Any]) -> Any:
        if self.successors:
            warnings.warn("Node won't run successors. Use AsyncFlow.", stacklevel=2)
        return await self._run_async(shared)

    async def _run_async(self, shared: Dict[str, Any]) -> Any:
        prep_res = await self.prep_async(shared)
        exec_res = await self._exec(prep_res)
        return await self.post_async(shared, prep_res, exec_res)

    def _run(self, shared: Dict[str, Any]) -> Any:
        raise RuntimeError("Use run_async for AsyncNode")


class AsyncBatchNode(AsyncNode, BatchNode):
    async def _exec(self, items: Iterable[Any]) -> Any:
        return [await AsyncNode._exec(self, item) for item in items]


class AsyncParallelBatchNode(AsyncNode, BatchNode):
    async def _exec(self, items: Iterable[Any]) -> Any:
        coros = [AsyncNode._exec(self, item) for item in items]
        return await asyncio.gather(*coros)


class AsyncFlow(Flow, AsyncNode):
    async def _orch_async(self, shared: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Any:
        if not self.start_node:
            raise RuntimeError("Flow has no start node configured")
        current = copy.copy(self.start_node)
        merged_params = params or {**self.params}
        last_action = None
        while current:
            current.set_params(merged_params)
            if isinstance(current, AsyncNode):
                last_action = await current._run_async(shared)
            else:
                last_action = current._run(shared)
            current = copy.copy(self.get_next_node(current, last_action))
        return last_action

    async def _run_async(self, shared: Dict[str, Any]) -> Any:
        prep_res = await self.prep_async(shared)
        exec_res = await self._orch_async(shared)
        return await self.post_async(shared, prep_res, exec_res)

    async def post_async(self, shared: Dict[str, Any], prep_res: Any, exec_res: Any) -> Any:
        return exec_res


class AsyncBatchFlow(AsyncFlow, BatchFlow):
    async def _run_async(self, shared: Dict[str, Any]) -> Any:
        prep_results = await self.prep_async(shared) or []
        for batch_params in prep_results:
            merged = {**self.params, **batch_params}
            await self._orch_async(shared, merged)
        return await self.post_async(shared, prep_results, None)


class AsyncParallelBatchFlow(AsyncFlow, BatchFlow):
    async def _run_async(self, shared: Dict[str, Any]) -> Any:
        prep_results = await self.prep_async(shared) or []
        coros = [self._orch_async(shared, {**self.params, **batch}) for batch in prep_results]
        await asyncio.gather(*coros)
        return await self.post_async(shared, prep_results, None)

