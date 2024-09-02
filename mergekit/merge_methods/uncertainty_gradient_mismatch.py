from typing import Any, Dict, List, Optional, Union
import torch
from torch import Tensor

from mergekit.architecture import WeightInfo
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.common import ImmutableMap, ModelReference
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)


class UncertaintyGradientMismatchTask(Task[torch.Tensor]):
    gather_tensors: GatherTensors
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    weight_info: WeightInfo
    base_model: Optional[ModelReference]

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self,
        tensors: Dict[ModelReference, torch.Tensor],
    ) -> Tensor:

        len_models = (
            len(tensors) // 2
        )  # first half are models, second half are hessians

        base_model_tensor = list(tensors.values())[0]

        if self.skip_tensor(base_model_tensor):
            return base_model_tensor

        ft_models_tensors = list(tensors.values())[1:len_models]
        base_hessian_tensor = list(tensors.values())[len_models]
        base_hessian_tensor = torch.ones_like(base_hessian_tensor)  # TODO
        ft_hessians_tensors = list(tensors.values())[len_models + 1 :]

        delta_0 = [v["delta_0"] for v in self.tensor_parameters.values()][0]
        alpha_t = [v["alpha_t"] for v in self.tensor_parameters.values()][1:len_models]

        assert len(ft_models_tensors) > 0, "No fine-tuned models provided"
        assert len(ft_models_tensors) == len(
            ft_hessians_tensors
        ), f"Mismatched number of FT models and Hessians {len(ft_models_tensors)} != {len(ft_hessians_tensors)}"
        assert len(ft_models_tensors) == len(
            alpha_t
        ), f"Mismatched number of FT models and alpha_t {len(ft_models_tensors)} != {len(alpha_t)}"

        return self.merging(
            base_model_tensor=base_model_tensor,
            base_hessian_tensor=base_hessian_tensor,  # TODO
            ft_models_tensors=ft_models_tensors,
            ft_hessians_tensors=ft_hessians_tensors,
            delta_0=delta_0,
            alpha_t=alpha_t,
        )

    def merging(
        self,
        base_model_tensor: Union[Tensor, None],
        base_hessian_tensor: Tensor,
        ft_models_tensors: List[Tensor],
        ft_hessians_tensors: List[Tensor],
        delta_0: float = 1e-12,
        alpha_t: Union[float, List[float]] = 1.0,
    ) -> Tensor:

        if base_model_tensor is not None:
            merged_model_tensor = torch.clone(base_model_tensor)
        else:
            merged_model_tensor = torch.zeros_like(ft_models_tensors[0])
            base_model_tensor = merged_model_tensor
            
        with torch.no_grad():
            if not self.skip_tensor(merged_model_tensor, use_tensor_name=True):
                # common part of preconditioner: \bar{H}^{-1}
                preconditioner = 1.0 / (
                    sum(
                        [
                            a_t * ft_hessian
                            for a_t, ft_hessian in zip(alpha_t, ft_hessians_tensors)
                        ]
                    )
                    + base_hessian_tensor
                    + delta_0
                )

                # summing the task-specific parts: \sum_t: \alpha_t * \bar{H}^{-1} * (H_{0+t}) * (\theta_ft - \llm)
                summed = sum(
                    [
                        (
                            a_t
                            * preconditioner
                            * (ft_hessian_tensor + (delta_0 + base_hessian_tensor))
                        )
                        * (ft_model_tensor - base_model_tensor)
                        for a_t, ft_model_tensor, ft_hessian_tensor in zip(
                            alpha_t, ft_models_tensors, ft_hessians_tensors
                        )
                    ]
                )

                merged_model_tensor += summed
        return merged_model_tensor

    def skip_tensor(self, tensor: torch.Tensor, use_tensor_name: bool = False) -> bool:
        name = tensor.names if use_tensor_name else self.weight_info.name
        return (
            "position_ids" in name
            or "attn.masked_bias" in name
            or tensor.dtype == torch.bool
        )


class UncertaintyGradientMismatch(MergeMethod):
    def parameters(self) -> List[ConfigParameterDef]:
        return []

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="delta_0", required=False, default_value=1e-12),
            ConfigParameterDef(name="alpha_t", required=False, default_value=1.0),
        ]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **_kwargs,
    ) -> Task:
        return UncertaintyGradientMismatchTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            base_model=base_model,
            weight_info=output_weight,
        )
