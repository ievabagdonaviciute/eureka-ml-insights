"""This file contains an implementation of the Physion++ eval: https://dingmyu.github.io/physion_v2/.
"""

import os
from typing import Any

from eureka_ml_insights.core import (
    PromptProcessing,
    Inference,
    DataProcessing,
    EvalReporting,
)
from eureka_ml_insights.metrics.metrics_base import ExactMatch
from eureka_ml_insights.metrics.reports import CountAggregator

from eureka_ml_insights.data_utils import (
    ColumnRename,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
    AddColumn,
)

from ..data_utils.physics_reasoning_utils import ExtractYesNoTransform

from eureka_ml_insights.configs import (
    ExperimentConfig,
    PromptProcessingConfig,
    InferenceConfig,
    DataProcessingConfig,
    EvalReportingConfig,
    DataSetConfig,
    PipelineConfig,
    AggregatorConfig,
    MetricConfig,
    ModelConfig,
)

class PHYSICS_REASONING_PIPELINE(ExperimentConfig):
    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None, **kwargs,) -> PipelineConfig:
        
        # prompt processing:
        self.data_proc = PromptProcessingConfig(
            component_type = PromptProcessing,
            data_reader_config = DataSetConfig(
                HFDataReader, {
                    "path": "ievabagdonaviciute/physics-reasoning-dataset", 
                    "split": "train",
                    "tasks": ["default"],
                    "transform": SequenceTransform([
                        ColumnRename(name_mapping={
                            "question": "prompt",
                            "ground_truth": "ground_truth",
                        }),
                    ]),
                },
            ),
            prompt_template_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "prompt_templates",
                "physics_reasoning_templates",
                "physics_reasoning_eval.jinja",
            ),          
            output_dir = os.path.join(self.log_dir, "01_data_processing"),
        )

        # inference:
        self.inference = InferenceConfig(
            component_type = Inference,
            model_config = model_config, # model_config is whatever you pass in at runtime
            data_loader_config = DataSetConfig(
                MMDataLoader, {
                    "path": os.path.join(self.data_proc.output_dir, "transformed_data.jsonl"),
                    "image_column_names": ["image"],
                }
            ),
            output_dir=os.path.join(self.log_dir, "02_inference"),
            resume_from=resume_from,
        )

        # post processing:
        self.post_proc = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform([
                        ColumnRename(name_mapping={"model_output": "raw_output"}),
                        AddColumn("model_output"),
                        ExtractYesNoTransform(),
                    ]),
                },
            ),
            output_dir=os.path.join(self.log_dir, "03_post_processing"),
        )

        self.eval_reporting = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.post_proc.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config = MetricConfig(ExactMatch),
            aggregator_configs=[
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["ExactMatch_result"],
                        "normalize": True,
                        "filename_base": "accuracy",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "04_eval_report"),
        )

        
        return PipelineConfig(
            [self.data_proc, self.inference, self.post_proc], self.log_dir)
        