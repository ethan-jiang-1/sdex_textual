# %% [markdown]
# # Textual-inversion fine-tuning for Stable Diffusion using d🧨ffusers 
# 
# This notebook shows how to "teach" Stable Diffusion a new concept via textual-inversion using 🤗 Hugging Face [🧨 Diffusers library](https://github.com/huggingface/diffusers). 
# 
# _By using just 3-5 images you can teach new concepts to Stable Diffusion and personalize the model on your own images_ 
# 
# For a general introduction to the Stable Diffusion model please refer to this [colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb).
# 
# 

# %% [markdown]
# ## Initial setup

# %%
#@title Install the required libs
#!pip install -qq diffusers["training"]==0.4.1 
#!pip install transformers ftfy
#!pip install -qq "ipywidgets>=7,<8"

# %%
#@title Import required libraries
#import argparse
#import itertools
#import math
import os
#import random

#import numpy as np
import torch
#import torch.nn.functional as F
import torch.utils.checkpoint
#from torch.utils.data import Dataset

#import PIL
#from accelerate import Accelerator
#from accelerate.logging import get_logger
#from accelerate.utils import set_seed
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel #,  DDPMScheduler, PNDMScheduler,
#from diffusers.hub_utils import init_git_repo, push_to_hub
#from diffusers.optimization import get_scheduler
#from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
#from PIL import Image
#from torchvision import transforms
#from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer # CLIPFeatureExtractor, 

# %%
#@title Login to the Hugging Face Hub
#@markdown Add a token with the "Write Access" role to be able to add your trained concept to the [Library of Concepts](https://huggingface.co/sd-concepts-library)
from huggingface_hub import notebook_login
notebook_login()

# %% [markdown]
# ## Settings for teaching your new concept

# %%
#@markdown `pretrained_model_name_or_path` which Stable Diffusion checkpoint you want to use
pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4" #@param {type:"string"}
#pretrained_model_name_or_path = "/root/autodl-tmp/sd-concept-output"

KEEP_TRAINING = True
DRY_RUN = True

# %%


# %% [markdown]
# ### Data for teapot, pizza and snake

# %%
from collections import namedtuple
from ti_utils import prepare_images

TiInfo = namedtuple("TiInfo", "what_to_teach, placeholder_token, initializer_token, save_path")

def create_ti_infos():
    ti_teapot = TiInfo(what_to_teach="object", placeholder_token="|teapot|", initializer_token="teapot", save_path="./my_concept_teapot")
    prepare_images(ti_teapot.save_path)

    ti_pizza = TiInfo(what_to_teach="object", placeholder_token="|pizza|", initializer_token="pizza", save_path="./my_concept_pizza")
    prepare_images(ti_pizza.save_path)

    ti_snake = TiInfo(what_to_teach="object", placeholder_token="|snake|", initializer_token="snake", save_path="./my_concept_snake")
    prepare_images(ti_snake.save_path)

    ti_infos = [ti_teapot, ti_pizza, ti_snake]
    return ti_infos

ti_infos = create_ti_infos()

# %% [markdown]
# ## Teach the model a new concept (fine-tuning with textual inversion)
# Execute this this sequence of cells to run the training process. The whole process may take from 1-4 hours. (Open this block if you are interested in how this process works under the hood or if you want to change advanced training settings or hyperparameters)

# %% [markdown]
# ### Create Dataset

# %% [markdown]
# ### Setting up the model

# %%
#title Load the tokenizer and add the placeholder token as a additional special token.
#@markdown Please read and if you agree accept the LICENSE [here](https://huggingface.co/CompVis/stable-diffusion-v1-4) if you see an error

tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
)

# Load models and create wrapper for stable diffusion
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae"
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
)


# %%
from ti_encoder import add_new_tokens_to_tokenizer, list_new_tokens

initializer_tokens = []
placeholder_tokens = []
if KEEP_TRAINING:
    #ethan: add new tokens to tokenizer

    for ti_info in ti_infos:
        initializer_tokens.append(ti_info.initializer_token)
        placeholder_tokens.append(ti_info.placeholder_token)

    add_new_tokens_to_tokenizer(tokenizer, initializer_tokens, placeholder_tokens)

    list_new_tokens(tokenizer)


# %% [markdown]
# We have added the `placeholder_token` in the `tokenizer` so we resize the token embeddings here, this will a new embedding vector in the token embeddings for our `placeholder_token`
# 
#  Initialise the newly added placeholder token with the embeddings of the initializer token

# %%
from ti_encoder import add_new_token_to_encoder_with_init_embedding

if KEEP_TRAINING:
    #ethan: add new tokens to encoder: the new token should inherit existing token's embedding, by copying it 
    add_new_token_to_encoder_with_init_embedding(tokenizer,
                                                text_encoder,
                                                initializer_tokens, 
                                                placeholder_tokens)

# %% [markdown]
# In Textual-Inversion we only train the newly added embedding vector, so lets freeze rest of the model parameters here

# %%
from ti_train import freeze_network
freeze_network(vae, unet, text_encoder)

# %% [markdown]
# ### Creating our training data

# %% [markdown]
# #### dataset_bpt 
# Let's create the Dataset and Dataloader

# %%
from ti_dataset import TextualInversionDataset, inspect_dataset

def create_train_datasets(ti_infos):
    train_datasets = []
    for ti_info in ti_infos:
        train_dataset = TextualInversionDataset(
            data_root=ti_info.save_path,
            tokenizer=tokenizer,
            size=512,
            placeholder_token=ti_info.placeholder_token,
            repeats=100,
            learnable_property=ti_info.what_to_teach, #Option selected above between object and style
            center_crop=False,
            set="train",
        )
        print(ti_info)
        inspect_dataset(train_dataset)
        train_datasets.append(train_dataset)
    return train_datasets

# %%
#ethan concat dataset all together
train_datasets = create_train_datasets(ti_infos)
train_dataset_all = torch.utils.data.ConcatDataset(train_datasets)
print("len(train_dataset_all)", len(train_dataset_all))

# %% [markdown]
# Define hyperparameters for our training
# If you are not happy with your results, you can tune the `learning_rate` and the `max_train_steps`

# %%
output_dir = "sd-concept-output"
if os.path.isdir("/root/autodl-tmp"):
    output_dir = "/root/autodl-tmp/sd-concept-output"

hyperparameters = {
    "learning_rate": 5e-04,
    "scale_lr": True,
    "max_train_steps": 3000,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "seed": 42,
    "output_dir": output_dir
}

#hyperparameters["max_train_steps"] = 100
hyperparameters["max_train_steps"] = 3000 * 3

# %% [markdown]
# Train!

# %%
from ti_train import get_new_placeholder_token_ids, training_function
#import accelerate

if KEEP_TRAINING:
    new_placeholder_tokens = placeholder_tokens
    _ids = get_new_placeholder_token_ids(tokenizer, new_placeholder_tokens)
    print(_ids)

    # accelerate.notebook_launcher(training_function, 
    #                              args=(hyperparameters, train_dataset_all, text_encoder, vae, unet, tokenizer, new_placeholder_tokens, DRY_RUN),
    #                              num_processes=1)

    # accelerate.debug_launcher(training_function, 
    #                           args=(hyperparameters, train_dataset_all, text_encoder, vae, unet, tokenizer, new_placeholder_tokens, DRY_RUN),
    #                           num_processes=1)
    training_function(hyperparameters, train_dataset_all, text_encoder, vae, unet, tokenizer, new_placeholder_tokens, DRY_RUN)

# %% [markdown]
# ## Run the code with your newly trained model
# If you have just trained your model with the code above, use the block below to run it
# 
# To save this concept for re-using, download the `learned_embeds.bin` file or save it on the library of concepts.
# 
# Use the [Stable Conceptualizer notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_conceptualizer_inference.ipynb) for inference with persistently saved pre-trained concepts
# 
#  upload to
# https://huggingface.co/sd-concepts-library/ettblackteapot

# %%
#@title Set up the pipeline 
def load_new_pipe(model_path_or_name):
    new_pipe = StableDiffusionPipeline.from_pretrained(
        model_path_or_name,
        torch_dtype=torch.float16,
    ).to("cuda")

    new_tokenizer = CLIPTokenizer.from_pretrained(
        model_path_or_name,
        subfolder="tokenizer",
    )
    print(model_path_or_name)
    print(new_tokenizer)
    print(new_tokenizer.vocab_size)
    #text_encoder.resize_token_embeddings(len(tokenizer_new))
    list_new_tokens(new_tokenizer)

    return new_pipe

new_pipe = load_new_pipe(hyperparameters["output_dir"])

# %%
def draw_result_grid(prompt, pipe, num_samples=2, num_rows=2):
    from ti_utils import image_grid
    all_images = [] 
    for _ in range(num_rows):
        images = pipe([prompt] * num_samples, num_inference_steps=50, guidance_scale=7.5).images
        all_images.extend(images)

    grid = image_grid(all_images, num_samples, num_rows)
    return grid

# %%
prompt = "a grafitti in a wall with a |teapot| on it" #@param {type:"string"}

num_samples = 2 #@param {type:"number"}
num_rows = 2 #@param {type:"number"}

draw_result_grid(prompt, new_pipe, num_samples=num_samples, num_rows=num_rows)

# %%
prompt = "a grafitti in a wall with a teapot on it" #@param {type:"string"}

num_samples = 2 #@param {type:"number"}
num_rows = 2 #@param {type:"number"}

draw_result_grid(prompt, new_pipe, num_samples=num_samples, num_rows=num_rows)

# %%
prompt = "a grafitti in a wall with a |snake| on it" #@param {type:"string"}

num_samples = 2 #@param {type:"number"}
num_rows = 2 #@param {type:"number"}

draw_result_grid(prompt, new_pipe, num_samples=num_samples, num_rows=num_rows)

# %%
prompt = "a grafitti in a wall with a snake on it" #@param {type:"string"}

num_samples = 2 #@param {type:"number"}
num_rows = 2 #@param {type:"number"}

draw_result_grid(prompt, new_pipe, num_samples=num_samples, num_rows=num_rows)

# %%
prompt = "a grafitti in a wall with a |pizza| on it" #@param {type:"string"}

num_samples = 2 #@param {type:"number"}
num_rows = 2 #@param {type:"number"}

draw_result_grid(prompt, new_pipe, num_samples=num_samples, num_rows=num_rows)

# %%
prompt = "a grafitti in a wall with a pizza on it" #@param {type:"string"}

num_samples = 2 #@param {type:"number"}
num_rows = 2 #@param {type:"number"}

draw_result_grid(prompt, new_pipe, num_samples=num_samples, num_rows=num_rows)

# %%
#@title Run the Stable Diffusion pipeline
#@markdown Don't forget to use the placeholder token in your prompt

prompt = " a picture of |teapot|, |snake|" #@param {type:"string"}

num_samples = 2 #@param {type:"number"}
num_rows = 2 #@param {type:"number"}

draw_result_grid(prompt, new_pipe, num_samples=num_samples, num_rows=num_rows)

# %%
#@title Run the Stable Diffusion pipeline
#@markdown Don't forget to use the placeholder token in your prompt

prompt = " a picture of |snake|, |teapot|" #@param {type:"string"}

num_samples = 2 #@param {type:"number"}
num_rows = 2 #@param {type:"number"}

draw_result_grid(prompt, new_pipe, num_samples=num_samples, num_rows=num_rows)

# %%
#@title Run the Stable Diffusion pipeline
#@markdown Don't forget to use the placeholder token in your prompt

prompt = " a picture of |pizza|, |teapot|" #@param {type:"string"}

num_samples = 2 #@param {type:"number"}
num_rows = 2 #@param {type:"number"}

draw_result_grid(prompt, new_pipe, num_samples=num_samples, num_rows=num_rows)

# %%
#@title Run the Stable Diffusion pipeline
#@markdown Don't forget to use the placeholder token in your prompt

prompt = " a |pizza| is on grass, and a |btp| and  a |snake| are nearby" #@param {type:"string"}

num_samples = 2 #@param {type:"number"}
num_rows = 2 #@param {type:"number"}

draw_result_grid(prompt, new_pipe, num_samples=num_samples, num_rows=num_rows)


