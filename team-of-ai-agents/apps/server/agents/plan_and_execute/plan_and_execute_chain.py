from typing import Any, Dict, List, Optional
from pydantic import Field
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain

from langchain_experimental.plan_and_execute.executors.base import BaseExecutor
from langchain_experimental.plan_and_execute.planners.base import BasePlanner
from langchain_experimental.plan_and_execute.schema import (
    BaseStepContainer,
    ListStepContainer,
)


class PlanAndExecuteChain(Chain):
    planner: BasePlanner
    executor: BaseExecutor

    def on_thoughts(x: List[Dict]) -> None:
        return None

    step_container: BaseStepContainer = Field(default_factory=ListStepContainer)
    input_key: str = "input"
    output_key: str = "output"
    thoughts: List[Dict] = []

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key, "chat_history"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        plan, user_steps = self.planner.plan(
            inputs,
            callbacks=run_manager.get_child() if run_manager else None,
        )

        self.thoughts = [
            {
                "id": index + 1,
                "title": step,
                "result": None,
                "loading": True,
            }
            for index, step in enumerate(user_steps)
        ]

        self.on_thoughts(self.thoughts)

        if run_manager:
            run_manager.on_text(str(plan), verbose=self.verbose)

        for index, step in enumerate(plan.steps):
            _new_inputs = {
                "previous_steps": self.step_container,
                "current_step": step,
                "objective": inputs[self.input_key],
            }
            new_inputs = {**_new_inputs, **inputs}

            response = self.executor.step(
                new_inputs,
                callbacks=run_manager.get_child() if run_manager else None,
            )

            self.thoughts[index]["result"] = response.response
            self.thoughts[index]["loading"] = False

            self.on_thoughts(self.thoughts)

            if run_manager:
                run_manager.on_text(
                    f"*****\n\nStep: {step.value}", verbose=self.verbose
                )
                run_manager.on_text(
                    f"\n\nResponse: {response.response}", verbose=self.verbose
                )
            self.step_container.add_step(step, response)
        return {self.output_key: self.step_container.get_final_response()}
