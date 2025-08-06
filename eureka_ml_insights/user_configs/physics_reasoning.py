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
from eureka_ml_insights.metrics.physics_reasoning_metrics import PhysicsReasoningMetric
from eureka_ml_insights.metrics.reports import CountAggregator

from eureka_ml_insights.data_utils import (
    CopyColumn,
    ColumnRename,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
    AddColumn,
    RegexTransform,
    ImputeNA,
)

from ..data_utils.physics_reasoning_utils import StripAfterAssistantTransform, SortByClassAndUIDTransform

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

from .llm_extraction import LLM_EXTRACTION_SUBPIPELINE_MIXIN
from eureka_ml_insights.configs.model_configs import TRAPI_GPT4O_2024_11_20_CONFIG, TRAPI_O1_CONFIG

###
class FilterByClassTransform:
    def __init__(self, class_prefix):
        self.class_prefix = class_prefix

    def transform(self, df):
        return df[df["class"].str.startswith(self.class_prefix)].reset_index(drop=True)
##

###################### WITH LLM EVALUATION

class PHYSICS_REASONING_PIPELINE(ExperimentConfig):
    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None, **kwargs,) -> PipelineConfig:
        
        # prompt processing:
        self.data_proc = PromptProcessingConfig(
            component_type = PromptProcessing,
            data_reader_config = DataSetConfig(
                DataReader, {
                    "path": "/home/t-ievab/project/local_dataset/physics_reasoning.jsonl",  # ✅ your local JSONL file
                    "format": ".jsonl",
                    "transform": SequenceTransform([
                        ColumnRename(name_mapping={
                            "question": "prompt",
                            "ground_truth": "ground_truth",
                            "image": "frames",
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
        image_column_name = "video" if getattr(model_config.class_name, "requires_video", False) else "frames" # e.g. LLaVAVideo

        self.inference = InferenceConfig(
            component_type = Inference,
            model_config = model_config, # model_config is whatever you pass in at runtime
            data_loader_config = DataSetConfig(
                MMDataLoader, {
                    "path": os.path.join(self.data_proc.output_dir, "transformed_data.jsonl"),
                    "image_column_names": [image_column_name],
                }
            ),
            output_dir=os.path.join(self.log_dir, "02_inference"),
            resume_from=resume_from,
            
        )

        # post processing:
        self.preeval_data_post_processing_comp = DataProcessingConfig(
            component_type = DataProcessing,
            data_reader_config = DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform([
                        ColumnRename(name_mapping={"model_output": "raw_output"}), # the model output from inference becomes raw output
                        CopyColumn(column_name_src="raw_output", column_name_dst="raw_output_copy"),  # back it up
                        StripAfterAssistantTransform(column="raw_output"), # if it is llava or llava next
                        ImputeNA(columns="raw_output", value=""), # in case there are some "nones" before
                        # RegexTransform(
                        #     columns="raw_output",
                        #     prompt_pattern=r"\b(true|false)\b", #r"\b(true|false|yes|no)\b",
                        #     ignore_case=True,
                        # ),
                        ColumnRename(name_mapping={"raw_output": "model_output"}),
                        # ImputeNA(columns="model_output", value=""),
                        ColumnRename(name_mapping={"raw_output_copy": "raw_output"}),
                    ])
                },
            ),
            output_dir=os.path.join(self.log_dir, "03_post_processing"),
        )


        # prepare LLM evaluation prompts that include both the model’s answer and the ground truth
        self.eval_prompt_proc = PromptProcessingConfig(
            component_type = PromptProcessing,
            data_reader_config = DataSetConfig(
                DataReader, {
                    "path": os.path.join(self.preeval_data_post_processing_comp.output_dir,
                                         "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform([
                        # rename the model’s free‐text answer
                        ColumnRename(name_mapping={"model_output": "model_answer"}),
                    ]),
                },
            ),
            prompt_template_path = os.path.join(
                os.path.dirname(__file__),
                "..","prompt_templates",
                "physics_reasoning_templates",
                "llm_evaluate.jinja", 
            ),
            output_dir = os.path.join(self.log_dir, "03_eval_prompt_processing"),
        )

        # 3) run the LLM on those eval prompts to get “correct”/“incorrect”
        self.eval_inference = InferenceConfig(
            component_type = Inference,
            model_config = TRAPI_O1_CONFIG,    # or whichever evaluator model you choose
            data_loader_config = DataSetConfig(
                MMDataLoader, {
                    "path": os.path.join(self.eval_prompt_proc.output_dir,
                                         "transformed_data.jsonl"),
                }
            ),
            output_dir = os.path.join(self.log_dir, "04_eval_inference"),
        )

        # 4) post‐process the LLM’s judgment into a boolean column
        self.eval_post_proc = DataProcessingConfig(
            component_type = DataProcessing,
            data_reader_config = DataSetConfig(
                DataReader, {
                    "path": os.path.join(self.eval_inference.output_dir,
                                         "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform([
                        # keep the raw eval answer:
                        ColumnRename(name_mapping={"model_output": "raw_evaluation"}),
                        # pull out only “correct” or “incorrect”
                        RegexTransform(
                            columns="raw_evaluation",
                            prompt_pattern=r"\b(correct|incorrect)\b",
                            ignore_case=True,
                        ),
                        ColumnRename(name_mapping={"raw_evaluation": "PhysicsReasoning_results"}),
                    ]),
                },
            ),
            output_dir = os.path.join(self.log_dir, "05_eval_post_processing"),
        )

        # report accuracy over new boolean column
        self.eval_reporting = EvalReportingConfig(
            component_type = EvalReporting,
            data_reader_config = DataSetConfig(
                DataReader, {
                    "path": os.path.join(self.eval_post_proc.output_dir,
                                         "transformed_data.jsonl"),
                    "format": ".jsonl",
                }
            ),
            #metric_config = MetricConfig(PhysicsReasoningMetric),
            aggregator_configs = [
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["PhysicsReasoning_results"],
                        "normalize": True,
                        "filename_base": "accuracy",
                    },
                ),
            ],
            output_dir = os.path.join(self.log_dir, "06_eval_report"),
        )

        return PipelineConfig(
            [
                self.data_proc,
                self.inference,
                self.preeval_data_post_processing_comp,

                self.eval_prompt_proc,
                self.eval_inference,
                self.eval_post_proc,
                self.eval_reporting,
            ],
            self.log_dir,
        )
    
########################## WITH LLM EXTRACTION ONLY

# class PHYSICS_REASONING_PIPELINE(ExperimentConfig):
#     def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None, **kwargs,) -> PipelineConfig:
        
#         # prompt processing:
#         self.data_proc = PromptProcessingConfig(
#             component_type = PromptProcessing,
#             data_reader_config = DataSetConfig(
#                 #HFDataReader, {
#                 DataReader, {
#                     #"path": "ievabagdonaviciute/physics-reasoning", 
#                     #"split": "train",
#                     #"tasks": ["default"],
#                     "path": "/home/t-ievab/project/local_dataset/physics_reasoning.jsonl",  # ✅ your local JSONL file
#                     "format": ".jsonl",
#                     "transform": SequenceTransform([
#                         ColumnRename(name_mapping={
#                             "question": "prompt",
#                             "ground_truth": "ground_truth",
#                             "image": "frames", # added?
#                         }),
#                         #AddColumn("wproperty"),
#                         #AddColumn("end_frame_idx"),
#                     ]),
#                 },
#             ),
#             prompt_template_path = os.path.join(
#                 os.path.dirname(__file__),
#                 "..",
#                 "prompt_templates",
#                 "physics_reasoning_templates",
#                 "physics_reasoning_eval.jinja",
#             ),          
#             output_dir = os.path.join(self.log_dir, "01_data_processing"),
#         )

#         # inference:

#         image_column_name = "video" if getattr(model_config.class_name, "requires_video", False) else "frames" # e.g. LLaVAVideo

#         self.inference = InferenceConfig(
#             component_type = Inference,
#             model_config = model_config, # model_config is whatever you pass in at runtime
            
#             data_loader_config = DataSetConfig(
#                 MMDataLoader, {
#                     "path": os.path.join(self.data_proc.output_dir, "transformed_data.jsonl"),
#                     "image_column_names": [image_column_name],
#                 }
#             ),
#             output_dir=os.path.join(self.log_dir, "02_inference"),
#             resume_from=resume_from,
            
#         )

#         # post processing:
#         self.preeval_data_post_processing_comp = DataProcessingConfig(
#             component_type = DataProcessing,
#             data_reader_config = DataSetConfig(
#                 DataReader,
#                 {
#                     "path": os.path.join(self.inference.output_dir, "inference_result.jsonl"),
#                     #"path": "/home/t-ievab/project/EUREKA/eureka-ml-insights/logs/PHYSICS_REASONING_PIPELINE/logs/goal_perception_analysis/physion_QWEN25_VL_78_INSTRUCT/2025-07-10-04-47-34.093546/02_inference/inference_result.jsonl",
#                     "format": ".jsonl",
#                     "transform": SequenceTransform([
#                         ColumnRename(name_mapping={"model_output": "raw_output"}), # the model output from inference becomes raw output
#                         CopyColumn(column_name_src="raw_output", column_name_dst="raw_output_copy"),  # back it up
#                         StripAfterAssistantTransform(column="raw_output"), # if it is llava or llava next
#                         ImputeNA(columns="raw_output", value=""), # in case there are some "nones" before
#                         RegexTransform(
#                             columns="raw_output",
#                             prompt_pattern=r"\b(true|false)\b", #r"\b(true|false|yes|no)\b",
#                             ignore_case=True,
#                         ),
#                         ColumnRename(name_mapping={"raw_output": "model_output"}),
#                         ImputeNA(columns="model_output", value=""),
#                         ColumnRename(name_mapping={"raw_output_copy": "raw_output"}),
#                     ])
#                 },
#             ),
#             output_dir=os.path.join(self.log_dir, "03_post_processing"),
#         )


#         self.llm_extraction_subpipeline_conf = LLM_EXTRACTION_SUBPIPELINE_MIXIN()
#         self.llm_extraction_subpipeline = self.llm_extraction_subpipeline_conf.configure_subpipeline(
#             extraction_attempt_component=self.preeval_data_post_processing_comp,
#             extracted_answer_col="model_output",
#             llm_extraction_prompt_template=os.path.join(
#                 os.path.dirname(__file__),
#                 "..",
#                 "prompt_templates",
#                 "physics_reasoning_templates",
#                 "llm_extract_answer.jinja",
#             ),
#             #llm_extractor_model_config=TRAPI_GPT4O_2024_11_20_CONFIG,
#             llm_extractor_model_config=TRAPI_O1_CONFIG,
#             log_dir=self.log_dir,
#             llm_extractor_max_concurrent=1,
#             llm_extractor_answer_transforms=[
#                 RegexTransform(
#                     columns="model_output",
#                     prompt_pattern=r"\b(true|false|NA)\b", #r"\b(true|false|yes|no)\b",
#                     ignore_case=True,
#                 ),
#             ],
#         )

            
#         # evaluation reporting:
#         self.eval_reporting = EvalReportingConfig(
#             component_type = EvalReporting,
#             data_reader_config=DataSetConfig(
#                 DataReader,
#                 {
#                     "path": os.path.join(self.llm_extraction_subpipeline[-1].output_dir, "transformed_data.jsonl"),
#                     "format": ".jsonl",
#                     "transform": SequenceTransform([
#                         SortByClassAndUIDTransform(),
#                     ])
#                 },
#             ),
#             metric_config = MetricConfig(PhysicsReasoningMetric),
#             aggregator_configs = [
#                 AggregatorConfig(
#                     CountAggregator,
#                     {
#                         "column_names": ["PhysicsReasoning_results"],
#                         "normalize": True,
#                         "filename_base": "accuracy",
#                     },
#                 ),
#             ],
#             output_dir=os.path.join(self.log_dir, "04_eval_report"),
#         )

        
#         # return PipelineConfig(
#         #     [self.data_proc, self.inference, self.post_proc, self.eval_reporting], self.log_dir)
        
#         return PipelineConfig(
#             [
#                 self.data_proc,
#                 self.inference,
#                 self.preeval_data_post_processing_comp,
#             ] + self.llm_extraction_subpipeline + [
#                 self.eval_reporting,
#             ],
#             self.log_dir,
#         )


################################################# WITHOUT LLM EXTRACTION:

# """This file contains an implementation of the Physion++ eval: https://dingmyu.github.io/physion_v2/.
# """

# import os
# from typing import Any

# from eureka_ml_insights.core import (
#     PromptProcessing,
#     Inference,
#     DataProcessing,
#     EvalReporting,
# )
# from eureka_ml_insights.metrics.physics_reasoning_metrics import PhysicsReasoningMetric
# from eureka_ml_insights.metrics.reports import CountAggregator

# from eureka_ml_insights.data_utils import (
#     ColumnRename,
#     DataReader,
#     HFDataReader,
#     MMDataLoader,
#     SequenceTransform,
#     AddColumn,
# )

# from ..data_utils.physics_reasoning_utils import ExtractYesNoTransform

# from eureka_ml_insights.configs import (
#     ExperimentConfig,
#     PromptProcessingConfig,
#     InferenceConfig,
#     DataProcessingConfig,
#     EvalReportingConfig,
#     DataSetConfig,
#     PipelineConfig,
#     AggregatorConfig,
#     MetricConfig,
#     ModelConfig,
# )
# ###
# class FilterByClassTransform:
#     def __init__(self, class_prefix):
#         self.class_prefix = class_prefix

#     def transform(self, df):
#         return df[df["class"].str.startswith(self.class_prefix)].reset_index(drop=True)
# ##

# class PHYSICS_REASONING_PIPELINE(ExperimentConfig):
#     def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None, **kwargs,) -> PipelineConfig:
        
#         # prompt processing:
#         self.data_proc = PromptProcessingConfig(
#             component_type = PromptProcessing,
#             data_reader_config = DataSetConfig(
#                 HFDataReader, {
#                     "path": "ievabagdonaviciute/physics-reasoning", 
#                     "split": "train",
#                     "tasks": ["default"],
#                     "transform": SequenceTransform([
#                         ColumnRename(name_mapping={
#                             "question": "prompt",
#                             "ground_truth": "ground_truth",
#                             "image": "frames", 
#                         }),
#                         #AddColumn("wproperty"),
#                         #AddColumn("end_frame_idx"),
#                     ]),
#                 },
#             ),
#             prompt_template_path = os.path.join(
#                 os.path.dirname(__file__),
#                 "..",
#                 "prompt_templates",
#                 "physics_reasoning_templates",
#                 "physics_reasoning_eval.jinja",
#             ),          
#             output_dir = os.path.join(self.log_dir, "01_data_processing"),
#         )

#         # inference:
#         image_column_name = "video" if getattr(model_config.class_name, "requires_video", False) else "frames" # e.g. LLaVAVideo
    

#         self.inference = InferenceConfig(
#             component_type = Inference,
#             model_config = model_config, # model_config is whatever you pass in at runtime
            
#             data_loader_config = DataSetConfig(
#                 MMDataLoader, {
#                     "path": os.path.join(self.data_proc.output_dir, "transformed_data.jsonl"),
#                     "image_column_names": [image_column_name],
#                 }
#             ),
#             output_dir=os.path.join(self.log_dir, "02_inference"),
#             resume_from=resume_from,
            
#         )

#         # post processing:
#         self.post_proc = DataProcessingConfig(
#             component_type = DataProcessing,
#             data_reader_config = DataSetConfig(
#                 DataReader,
#                 {
#                     "path": os.path.join(self.inference.output_dir, "inference_result.jsonl"),
#                     "format": ".jsonl",
#                     "transform": SequenceTransform([
#                         ColumnRename(name_mapping={"model_output": "raw_output"}),
#                         AddColumn("model_output"),
#                         ExtractYesNoTransform(),
#                     ]),
#                 },
#             ),
#             output_dir=os.path.join(self.log_dir, "03_post_processing"),
#         )

#         # evaluation reporting:
#         self.eval_reporting = EvalReportingConfig(


# # have separate answer extraction and evaluation

#             component_type = EvalReporting,
#             data_reader_config = DataSetConfig(
#                 DataReader,
#                 {
#                     "path": os.path.join(self.post_proc.output_dir, "transformed_data.jsonl"),
#                     "format": ".jsonl",
#                 },
#             ),
#             metric_config = MetricConfig(PhysicsReasoningMetric),
#             aggregator_configs = [
#                 AggregatorConfig(
#                     CountAggregator,
#                     {
#                         "column_names": ["PhysicsReasoning_results"],
#                         "normalize": True,
#                         "filename_base": "accuracy",
#                     },
#                 ),
#             ],
#             output_dir=os.path.join(self.log_dir, "04_eval_report"),
#         )

        
#         return PipelineConfig(
#             [self.data_proc, self.inference, self.post_proc, self.eval_reporting], self.log_dir)
