from eureka_ml_insights.configs.config import ModelConfig
from eureka_ml_insights.models import (
    TogetherModel,
    ClaudeModel,
    ClaudeReasoningModel,
    GeminiModel,
    LlamaServerlessAzureRestEndpointModel,
    LLaVAModel,
    LLaVAHuggingFaceModel,
    VideoLLaVAModel,
    VideoLLaVANeXTModel,
    MistralServerlessAzureRestEndpointModel,
    DeepseekR1ServerlessAzureRestEndpointModel,
    Phi4HFModel,
    DirectOpenAIModel,
    DirectOpenAIOModel,
    AzureOpenAIModel,
    AzureOpenAIOModel,
    RestEndpointModel,
    InternVLChatModel,
    QwenVLModel,
)

# Together models
TOGETHER_SECRET_KEY_PARAMS = {
    # we do not have a key yet for together models
    "key_name": "your_togetherai_secret_key_name",
    "local_keys_path": "keys/aifeval-vault-azure-net.json",
    "key_vault_url": None,
}

TOGETHER_DEEPSEEK_R1_CONFIG = ModelConfig(
    TogetherModel,
    {
        "model_name": "deepseek-ai/DeepSeek-R1",
        "secret_key_params": TOGETHER_SECRET_KEY_PARAMS,
        "temperature": 0.6,
        # high max token limit for deep seek
        # otherwise the answers may be cut in the middle
        "max_tokens": 65536
    },
)

TOGETHER_DEEPSEEK_R1_Distill_Llama_70B_CONFIG = ModelConfig(
    TogetherModel,
    {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "secret_key_params": TOGETHER_SECRET_KEY_PARAMS,
        "temperature": 0.6,
        # high max token limit for deep seek
        # otherwise the answers may be cut in the middle
        "max_tokens": 65536
    },
)

# OpenAI models
OPENAI_SECRET_KEY_PARAMS = {
    "key_name": "openai",
    "local_keys_path": "keys/aifeval-vault-azure-net.json",
    "key_vault_url": "https://aifeval.vault.azure.net",
}

OAI_GPT45_PREVIEW_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4.5-preview-2025-02-27",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
        "temperature": 1.0,
        "max_tokens": 4096
    },
)

OAI_O1_CONFIG = ModelConfig(
    DirectOpenAIOModel,
    {
        "model_name": "o1-2024-12-17",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_O3_MINI_HIGH_CONFIG = ModelConfig(
    DirectOpenAIOModel,
    {
        "model_name": "o3-mini-2025-01-31",
        "reasoning_effort": "high",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_O3_MINI_CONFIG = ModelConfig(
    DirectOpenAIOModel,
    {
        "model_name": "o3-mini-2025-01-31",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_O1_PREVIEW_CONFIG = ModelConfig(
    DirectOpenAIOModel,
    {
        "model_name": "o1-preview",
        "num_retries": 1,
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_GPT4_1106_PREVIEW_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4-1106-preview",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_GPT4V_1106_VISION_PREVIEW_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4-1106-vision-preview",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_GPT4V_TURBO_2024_04_09_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4-turbo-2024-04-09",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_GPT4O_2024_05_13_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4o-2024-05-13",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

OAI_GPT4O_2024_11_20_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "model_name": "gpt-4o-2024-11-20",
        "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
    },
)

################# MSR LIT Models #################

MSR_LIT_O1_reasoning_1_CONFIG = ModelConfig(
    AzureOpenAIOModel,
    {
        "url": "https://reasoning-eastus2.openai.azure.com/",
        "api_version": '2024-12-01-preview',
        ## o1: o1-reasoning-1, o1-reasoning-2
        "model_name": "o1-reasoning-1",
        "auth_scope": "https://cognitiveservices.azure.com/.default",        
    },
)

MSR_LIT_O3_mini_reasoning_1_CONFIG = ModelConfig(
    AzureOpenAIOModel,
    {
        "url": "https://reasoning-eastus2.openai.azure.com/",
        "api_version": '2024-12-01-preview',
        ## o3-mini: o3-mini-reasoning-1, o3-mini-reasoning-2
        "model_name": "o3-mini-reasoning-1",
        "auth_scope": "https://cognitiveservices.azure.com/.default",
        "reasoning_effort": "high"
    },
)

####


# Azure OAI models
## Azure OAI models -- TRAPI Models 
## https://dev.azure.com/msresearch/MSR%20Engineering/_wiki/wikis/MSR-Engineering.wiki/13498/Deployment-Model-Information

# this does not work yet, adding as provisionary
TRAPI_GPT45_PREVIEW_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        "api_version": '2024-12-01-preview',
        "model_name": "gpt-4.5-preview_2025-02-27",
        "auth_scope": "api://trapi/.default",
        "temperature": 1.0,
        "max_tokens": 4096
    },
)

TRAPI_GCR_GPT45_PREVIEW_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        "url": "https://trapi.research.microsoft.com/gcr/shared",
        "api_version": '2024-12-01-preview',
        "model_name": "gpt-4.5-preview_2025-02-27",
        "auth_scope": "api://trapi/.default",
        "temperature": 1.0,
        "max_tokens": 4096
    },
)

TRAPI_O1_CONFIG = ModelConfig(
    AzureOpenAIOModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        # o1 models only work with 2024-12-01-preview api version
        "api_version": '2024-12-01-preview',
        "model_name": "o1_2024-12-17",
        "auth_scope": "api://trapi/.default"
    },
)

# this endpoint is used by all GCR members, only use when the AIF endpoints are down or not available
TRAPI_GCR_SHARED_O1_CONFIG = ModelConfig(
    AzureOpenAIOModel,
    {
        "url": "https://trapi.research.microsoft.com/gcr/shared",
        # o1 models only work with 2024-12-01-preview api version
        "api_version": '2024-12-01-preview',
        "model_name": "o1_2024-12-17",
        "auth_scope": "api://trapi/.default"
    },
)

TRAPI_O1_PREVIEW_CONFIG = ModelConfig(
    AzureOpenAIOModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        "api_version": '2024-10-21',
        "model_name": "o1-preview_2024-09-12",
        "auth_scope": "api://trapi/.default"
    },
)

TRAPI_GPT4O_2024_08_06_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        "api_version": '2024-10-21',
        "model_name": "gpt-4o_2024-08-06",
        "auth_scope": "api://trapi/.default"
    },
)

TRAPI_GPT4O_2024_11_20_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        #"url": "https://trapi.research.microsoft.com/msraif/shared",
        "url": "https://trapi.research.microsoft.com/gcr/preview/openai",
        "api_version": '2024-10-21',
        "model_name": "gpt-4o_2024-11-20",
        "auth_scope": "api://trapi/.default",
		"temperature": 1.0,
        "max_tokens": 4096
    },
)

TRAPI_GPT4_VISION_PREVIEW_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        "model_name": "gpt-4_vision-preview",
        "auth_scope": "api://trapi/.default"
    },
)

TRAPI_GPT4V_TURBO_2024_04_09_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        "model_name": "gpt-4_turbo-2024-04-09",
        "api_version": '2024-10-21',
        "auth_scope": "api://trapi/.default"
    },
)

TRAPI_GPT4_1106_PREVIEW_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        "model_name": "gpt-4_1106-Preview",
        "auth_scope": "api://trapi/.default"
    },
)

TRAPI_AIF_O3_MINI_HIGH_CONFIG = ModelConfig(
    AzureOpenAIOModel,
    {
        "url": "https://trapi.research.microsoft.com/msraif/shared",
        # o1 models only work with > 2024-12-01-preview api version
        "api_version": '2025-01-01-preview',
        "model_name": "o3-mini_2025-01-31",
        "reasoning_effort": "high",
        "auth_scope": "api://trapi/.default"
    },
)
# Azure OAI models -- AIF Endpoints 
# These endpoints currently issue policy violation errors
# AIF_WUS3_GPT4_1106_PREVIEW_CONFIG = ModelConfig(
#     AzureOpenAIModel,
#     {
#         "url": "https://openai-models-west-us3.openai.azure.com/",
#         "model_name": "gpt-4-july",
#         "auth_scope": "https://cognitiveservices.azure.com/.default"
#     },
# )

# AIF_WUS3_GPT4O_2024_05_13_450K_CONFIG = ModelConfig(
#     AzureOpenAIModel,
#     {
#         "url": "https://openai-models-west-us3.openai.azure.com/",
#         "model_name": "gpt-4o-450K",
#         "auth_scope": "https://cognitiveservices.azure.com/.default"
#     },
# )

# AIF_WUS3_GPT4O_2024_05_13_150K_CONFIG = ModelConfig(
#     AzureOpenAIModel,
#     {
#         "url": "https://openai-models-west-us3.openai.azure.com/",
#         "model_name": "gpt-4o",
#         "auth_scope": "https://cognitiveservices.azure.com/.default"
#     },
# )

# AZURE_OAI_O1_PREVIEW_CONFIG = ModelConfig(
#     AzureOpenAIOModel,
#     {
#         "url": "https://crescoeastus2.openai.azure.com",
#         "model_name": "o1-preview",
#         "auth_scope": "https://cognitiveservices.azure.com/.default"
#     }
# )


# Gemini models
GEMINI_SECRET_KEY_PARAMS = {
    "key_name": "aif-eval-gemini",
    # currently we have three keys: "aif-eval-gemini-firstproject", "aif-eval-gemini", "aif-eval-gemini-aifevalunderstandproject"
    # New keys for Gemini 2.0: "aif-eval-gemini-aime", "aif-eval-gemini-aifevalunderstandproject", "aif-eval-gemini": "aif-eval-gemini-vl","aif-eval-gemini-benchagents","aif-eval-gemini-nphard" 
    # rotate between these if you get '429 Resource has been exhausted (e.g. check quota)'
    "local_keys_path": "keys/aifeval-vault-azure-net.json",
    "key_vault_url": "https://aifeval.vault.azure.net",
}

GEMINI_V2_PRO_T1_M4096_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-2.0-pro-exp-02-05",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature":1.0,
        "max_tokens":4096,

    },
)

GEMINI_V2_FLASH_THINKING_EXP_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-2.0-flash-thinking-exp-01-21",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        "temperature": 1.0,
        "max_tokens": 32768
    },
)

GEMINI_EXP_1206_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-exp-1206",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
		"temperature": 1.0,
        "max_tokens": 4096
    },
)

GEMINI_V15_PRO_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-1.5-pro",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
    },
)

GEMINI_V1_PRO_CONFIG = ModelConfig(
    GeminiModel,
    {
        "model_name": "gemini-1.0-pro",
        "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
    },
)

# Claude models
CLAUDE_SECRET_KEY_PARAMS = {
    "key_name": "aif-eval-claude",
    "local_keys_path": "keys/aifeval-vault-azure-net.json",
    "key_vault_url": "https://aifeval.vault.azure.net",
}

CLAUDE_3_OPUS_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "model_name": "claude-3-opus-20240229",
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
    },
)

CLAUDE_3_5_SONNET_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-5-sonnet-20240620",
    },
)

CLAUDE_3_7_SONNET_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-7-sonnet-20250219",
    },
)

CLAUDE_3_7_SONNET_THINKING_CONFIG = ModelConfig(
    ClaudeReasoningModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-7-sonnet-20250219",
        "thinking_enabled": True,
        "thinking_budget": 30720,
        "max_tokens": 32768, # This number should always be higher than the thinking budget
        "temperature": 1.0, # As of 03/08/2025, thinking only works with temperature 1.0
        "timeout": 600, # We set a timeout of 10 minutes for thinking
    },
)

CLAUDE_3_5_SONNET_20241022_CONFIG = ModelConfig(
    ClaudeModel,
    {
        "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
        "model_name": "claude-3-5-sonnet-20241022",
		"temperature": 1.0,
        "max_tokens": 4096
    },
)

# Agent Parallelism Endpoints
# Sahaj is an owner of the subscription. Contact him for any issues with the endpoints or for being added to the subscription.
AGENT_PAR_GPT4O_2024_11_20_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        "url": "https://aoai-eval-ncus.openai.azure.com",
        "model_name": "gpt-4o_2024-11-20",
        "api_version": "2024-08-01-preview",
        "temperature": 1.0,
        "max_tokens": 4096,
        "auth_scope": "https://cognitiveservices.azure.com/.default"
    },
)

AGENT_PAR_GPT4O_2024_08_06_CONFIG = ModelConfig(
    AzureOpenAIModel,
    {
        "url": "https://aoai-eval-ncus.openai.azure.com",
        "model_name": "gpt-4o_2024-08-06",
        "api_version": "2024-08-01-preview",
        "temperature": 1.0,
        "auth_scope": "https://cognitiveservices.azure.com/.default"
    },
)

AGENT_PAR_O1_2024_12_17_CONFIG = ModelConfig(
    AzureOpenAIOModel,
    {
        "url": "https://ap-orca-eus2.openai.azure.com/",
        "model_name": "o1-2",
        "api_version": '2024-12-01-preview',
        "auth_scope": "https://cognitiveservices.azure.com/.default"
    },
)


# LLAVA models
LLAVAHF_V16_34B_CONFIG = ModelConfig(
    LLaVAHuggingFaceModel,
    {"model_name": "llava-hf/llava-v1.6-34b-hf", "use_flash_attn": True},
)

LLAVAHF_V15_7B_CONFIG = ModelConfig(
    LLaVAHuggingFaceModel,
    {"model_name": "llava-hf/llava-1.5-7b-hf", "use_flash_attn": True},
)

LLAVA_V16_34B_CONFIG = ModelConfig(
    LLaVAModel,
    {"model_name": "liuhaotian/llava-v1.6-34b", "use_flash_attn": True},
)

LLAVA_V15_7B_CONFIG = ModelConfig(
    LLaVAModel,
    {"model_name": "liuhaotian/llava-v1.5-7b", "use_flash_attn": True},
)
 
VIDEOLLAVA_7B_CONFIG = ModelConfig(
    VideoLLaVAModel,
    {"model_name": "LanguageBind/Video-LLaVA-7B-hf", "use_flash_attn": True},
)

VIDEOLLAVA_NEXT_7B_CONFIG = ModelConfig(
    VideoLLaVANeXTModel,
    {"model_name": "LanguageBind/LLaVA-NeXT-Video-7B-hf", "use_flash_attn": True},
)

# InternVL models

MINI_INTERNVL_CHAT_4B_V1_5_CONFIG = ModelConfig(
    InternVLChatModel,
    {"model_name": "OpenGVLab/Mini-InternVL-Chat-4B-V1-5", "use_flash_attn": True},
)

# Qwen models

QWEN25_VL_78_INSTRUCT = ModelConfig(
    QwenVLModel,
    {"model_name": "Qwen/Qwen2.5-VL-7B-Instruct", "use_flash_attn": True},
)

# Phi models

PHI4_HF_CONFIG = ModelConfig(
    Phi4HFModel,
    {
        "model_name": "microsoft/phi-4",
        "temperature": 1.0,
        "max_tokens": 4096,
        "use_flash_attn": True,
    },
)

AIF_PHI4_CONFIG_1 = ModelConfig(
    RestEndpointModel,
    {
        "url": "https://aif-phi4.eastus2.inference.ml.azure.com/score",
        "secret_key_params": {
            "key_name": "phi-4-aif",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
    },
)

AIF_PHI4_CONFIG_2 = ModelConfig(
    RestEndpointModel,
    {
        "url": "https://aif-phi4-2.eastus2.inference.ml.azure.com/score",
        "secret_key_params": {
            "key_name": "phi-4-aif-2",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
    },
)

AIF_PHI4_CONFIG_3 = ModelConfig(
    RestEndpointModel,
    {
        "url": "https://aif-phi4-3.eastus2.inference.ml.azure.com/score",
        "secret_key_params": {
            "key_name": "phi-4-aif-3",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
    },
)

GCR_PHI4_CONFIG = ModelConfig(
    RestEndpointModel,
    {
        "url": "https://gcr-phi-4.westus3.inference.ml.azure.com/score",
        "secret_key_params": {
            "key_name": "gcr-phi-4",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
    },
)

GCR_PHI3_MINI_128K_INSTRUCT_CONFIG = ModelConfig(
    RestEndpointModel,
    {
        "url": "https://gcr-phi3-mini-128k-instruct.westus3.inference.ml.azure.com/score",
        "secret_key_params": {
            "key_name": "phi-3-mini-128k-instruct-7",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
    },
)

# Llama models
GCR_LLAMA3_70B_INSTRUCT_CONFIG = ModelConfig(
    RestEndpointModel,
    {
        "url": "https://gcr-llama3-70b-instruct.westus3.inference.ml.azure.com/score",
        "secret_key_params": {
            "key_name": "meta-llama-3-70b-instruct-4",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
    },
)

GCR_LLAMA3_1_70B_INSTRUCT_CONFIG = ModelConfig(
    RestEndpointModel,
    {
        "url": "https://gcr-llama31-70b-instruct.westus3.inference.ml.azure.com/score",
        "secret_key_params": {
            "key_name": "meta-llama-3-1-70b-instruct-1",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
    },
)

AIF_NT_LLAMA3_1_405B_INSTRUCT_EASTUS_OSS_CONFIG = ModelConfig(
    LlamaServerlessAzureRestEndpointModel,
    {
        "url": "https://Meta-Llama-3-1-405B-Instruct-aif.eastus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-meta-llama-3-1-405b-instruct-1",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
    },
)

AIF_NT_LLAMA3_1_405B_INSTRUCT_EASTUS_OSS_CONFIG_2 = ModelConfig(
    LlamaServerlessAzureRestEndpointModel,
    {
        "url": "https://Meta-Llama-3-1-405B-Instruct-2.eastus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-meta-llama-3-1-405b-instruct-2-oss",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
		"temperature": 1.0,
        "max_tokens": 4096
    },
)

AIF_NT_LLAMA3_1_405B_INSTRUCT_WESTUS_CONFIG = ModelConfig(
    LlamaServerlessAzureRestEndpointModel,
    {
        "url": "https://Meta-Llama-3-1-405B-Instruct-4.westus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-meta-llama-3-1-405b-instruct-4-westus",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
		"temperature": 1.0,
        "max_tokens": 4096
    },
)

AIF_NT_LLAMA3_1_405B_INSTRUCT_WESTUS3_CONFIG = ModelConfig(
    LlamaServerlessAzureRestEndpointModel,
    {
        "url": "https://Meta-Llama-3-1-405B-Instruct-2.westus3.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-meta-llama-3-1-405b-instruct-westus3",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
		"temperature": 1.0,
        "max_tokens": 4096
    },
)


AIF_NT_LLAMA3_2_90B_VISION_INSTRUCT_CONFIG = ModelConfig(
    LlamaServerlessAzureRestEndpointModel,
    {
        "url": "https://Llama-3-2-90B-Vision-Instruct-ev.eastus2.models.ai.azure.com/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-meta-llama-3-2-90b-Instruct-1",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "model_name": "llama32",
    },
)

AIF_NT_LLAMA3_2_90B_VISION_INSTRUCT_CONFIG_2 = ModelConfig(
    LlamaServerlessAzureRestEndpointModel,
    {
        "url": "https://Llama-3-2-90B-Vision-Instruct-2.eastus2.models.ai.azure.com/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-meta-llama-3-2-90b-Instruct-2",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
        },
    },
)

# Mistral Endpoints
AIF_NT_MISTRAL_LARGE_2_2407_CONFIG = ModelConfig(
    MistralServerlessAzureRestEndpointModel,
    {
        "url": "https://Mistral-large-2407-aifeval.eastus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-mistral-large-2-2407",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
    },
)

# DeepSeek R1 Endpoints on Azure
MSR_LIT_DEEPSEEK_R1_CONFIG = ModelConfig(
    DeepseekR1ServerlessAzureRestEndpointModel,
    {
        "url": "https://deepseek-r1-reasoning.westus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "lit-deepseek-r1",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "max_tokens": 32768,
        "timeout": 1200
    },
)

MSR_LIT_DEEPSEEK_R1_CONFIG_2 = ModelConfig(
    DeepseekR1ServerlessAzureRestEndpointModel,
    {
        "url": "https://deepseek-r1-reasoning-2.westus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "lit-deepseek-r1-2",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "max_tokens": 32768,
        "timeout": 1200
    },
)

MSR_LIT_DEEPSEEK_R1_CONFIG_3 = ModelConfig(
    DeepseekR1ServerlessAzureRestEndpointModel,
    {
        "url": "https://deepseek-r1-reasoning-3.westus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "lit-deepseek-r1-3",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "max_tokens": 32768,
        "timeout": 1200
    },
)

SHARED_LIT_DEEPSEEK_R1_CONFIG_3 = ModelConfig(
    DeepseekR1ServerlessAzureRestEndpointModel,
    {
        "url": "https://DeepSeek-R1-zrvlt.westus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "shared-lit-r1",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "max_tokens": 32768,
        "timeout": 1200
    },
)



AIF_NT_DEEPSEEK_R1_CONFIG = ModelConfig(
    DeepseekR1ServerlessAzureRestEndpointModel,
    {
        "url": "https://DeepSeek-R1-aif-nt.eastus.models.ai.azure.com/v1/chat/completions",
        "secret_key_params": {
            "key_name": "aif-nt-deepseek-r1",
            "local_keys_path": "keys/aifeval-vault-azure-net.json",
            "key_vault_url": "https://aifeval.vault.azure.net",
        },
        "max_tokens": 32768,
        "timeout": 1200
    },
)

### Gateway models
GATEWAY_SECRET_KEY_PARAMS = {
    "key_name": "your_gateway_key",
    "local_keys_path": "keys/keys.json",
    "key_vault_url": None,
}

GATEWAY_GPT_4O_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "base_url": "https://gateway.phyagi.net/api/",
        "model_name": "gpt-4o-impact",
        "secret_key_params": GATEWAY_SECRET_KEY_PARAMS,
        "extra_body":{"tier": "impact", "cache_ttl": 0},
        #"temperature": 1,
        #"max_tokens": 4096
    },
)

GATEWAY_O1_CONFIG = ModelConfig(
    DirectOpenAIOModel,
    {
        "base_url": "https://gateway.phyagi.net/api/",
        "model_name": "o1-impact",
        "secret_key_params": GATEWAY_SECRET_KEY_PARAMS,
        "extra_body":{"tier": "impact", "cache_ttl": 0},
    },
)

GATEWAY_O3_MINI_CONFIG = ModelConfig(
    DirectOpenAIOModel,
    {
        "base_url": "https://gateway.phyagi.net/api/",
        "model_name": "o3-mini-impact",
        "reasoning_effort": "high",
        "secret_key_params": GATEWAY_SECRET_KEY_PARAMS,
        "extra_body":{"tier": "impact", "cache_ttl": 0},
    },
)

GATEWAY_PHI_4_sft_14b_allmathv2_code_2e_coco_5e_high32k_10xlr = ModelConfig(
    DirectOpenAIModel,
    {
        "base_url": "https://gateway.phyagi.net/api/",
        "model_name": "sft_14b_allmathv2_code_2e_coco_5e_high32k_10xlr",
        "secret_key_params": GATEWAY_SECRET_KEY_PARAMS,
        "extra_body":{"tier": "impact", "cache_ttl": 0},
        "temperature": 0.8,
         "max_tokens": 30000
    },
)

GATEWAY_PHI_4_CONFIG = ModelConfig(
    DirectOpenAIModel,
    {
        "base_url": "https://gateway.phyagi.net/api/",
        "model_name": "phi-4",
        "secret_key_params": GATEWAY_SECRET_KEY_PARAMS,
        "extra_body":{"tier": "impact", "cache_ttl": 0},
        "temperature": 0.8,
        "max_tokens": 4096
    },
)

GATEWAY_PHI_4_CONFIG_TEMP_ZERO = ModelConfig(
    DirectOpenAIModel,
    {
        "base_url": "https://gateway.phyagi.net/api/",
        "model_name": "phi-4",
        "secret_key_params": GATEWAY_SECRET_KEY_PARAMS,
        "extra_body":{"tier": "impact", "cache_ttl": 0},
        "temperature": 0.0,
        "max_tokens": 4096
    },
)

# The following dictionaties are used to better organize and access the GPT model configurations
# since there are so many GPT models to keep track of:

# A dictionary of all TRAPI model configurations for easy access
TRAPI_MODEL_CONFIGS = {
    "TRAPI_GPT45_PREVIEW_CONFIG": TRAPI_GPT45_PREVIEW_CONFIG,
    "TRAPI_GCR_GPT45_PREVIEW_CONFIG": TRAPI_GCR_GPT45_PREVIEW_CONFIG,
    "TRAPI_O3_MINI_HIGH_CONFIG": TRAPI_AIF_O3_MINI_HIGH_CONFIG,
    "TRAPI_O1_CONFIG": TRAPI_O1_CONFIG,
    "TRAPI_GCR_SHARED_O1_CONFIG": TRAPI_GCR_SHARED_O1_CONFIG,
    "TRAPI_O1_PREVIEW_CONFIG": TRAPI_O1_PREVIEW_CONFIG,
    "TRAPI_GPT4O_2024_08_06_CONFIG": TRAPI_GPT4O_2024_08_06_CONFIG,
    "TRAPI_GPT4O_2024_11_20_CONFIG": TRAPI_GPT4O_2024_11_20_CONFIG,
    "TRAPI_GPT4_VISION_PREVIEW_CONFIG": TRAPI_GPT4_VISION_PREVIEW_CONFIG,
    "TRAPI_GPT4V_TURBO_2024_04_09_CONFIG": TRAPI_GPT4V_TURBO_2024_04_09_CONFIG,
    "TRAPI_GPT4_1106_PREVIEW_CONFIG": TRAPI_GPT4_1106_PREVIEW_CONFIG,
}

# A dictionary of all AIF model configurations for easy access
# AIF_MODEL_CONFIGS = {
#     "AIF_WUS3_GPT4_1106_PREVIEW_CONFIG": AIF_WUS3_GPT4_1106_PREVIEW_CONFIG,
#     "AIF_WUS3_GPT4O_2024_05_13_450K_CONFIG": AIF_WUS3_GPT4O_2024_05_13_450K_CONFIG,
#     "AIF_WUS3_GPT4O_2024_05_13_150K_CONFIG": AIF_WUS3_GPT4O_2024_05_13_150K_CONFIG,
#     "AZURE_OAI_O1_PREVIEW_CONFIG": AZURE_OAI_O1_PREVIEW_CONFIG,
# }

# A dictionary of all OpenAI model configurations for easy access
OPENAI_MODEL_CONFIGS = {
    "OAI_GPT45_PREVIEW_CONFIG": OAI_GPT45_PREVIEW_CONFIG,
    "OAI_O1_CONFIG": OAI_O1_CONFIG,
    "OAI_O3_MINI_HIGH_CONFIG": OAI_O3_MINI_HIGH_CONFIG,
    "OAI_O3_MINI_CONFIG": OAI_O3_MINI_CONFIG,
    "OAI_O1_PREVIEW_CONFIG": OAI_O1_PREVIEW_CONFIG,
    "OAI_GPT4_1106_PREVIEW_CONFIG": OAI_GPT4_1106_PREVIEW_CONFIG,
    "OAI_GPT4V_1106_VISION_PREVIEW_CONFIG": OAI_GPT4V_1106_VISION_PREVIEW_CONFIG,
    "OAI_GPT4V_TURBO_2024_04_09_CONFIG": OAI_GPT4V_TURBO_2024_04_09_CONFIG,
    "OAI_GPT4O_2024_05_13_CONFIG": OAI_GPT4O_2024_05_13_CONFIG,
    "OAI_GPT4O_2024_11_20_CONFIG": OAI_GPT4O_2024_11_20_CONFIG
}
