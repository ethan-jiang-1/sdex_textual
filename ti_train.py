import itertools
import torch 
import math
import os
from tqdm.auto import tqdm
#from diffusers import DDPMScheduler
import logging
from accelerate import Accelerator
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers import DDPMScheduler, PNDMScheduler, StableDiffusionPipeline # AutoencoderKL, , UNet2DConditionModel
from transformers import CLIPFeatureExtractor #, CLIPTextModel, CLIPTokenizer
#from torch._C import NoneType
import torch.nn.functional as F

def _freeze_params(params):
    for param in params:
        param.requires_grad = False


def freeze_network(vae, unet, text_encoder):
    # Freeze vae and unet
    _freeze_params(vae.parameters())
    _freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    _freeze_params(params_to_freeze)


def get_new_placeholder_token_ids(tokenizer, new_placeholder_tokens):
    new_placeholder_token_ids = []

    for new_placeholder_token in new_placeholder_tokens:
        placeholder_token_id = tokenizer.encode(new_placeholder_token, add_special_tokens=False)[0]
        new_placeholder_token_ids.append(placeholder_token_id)
    return new_placeholder_token_ids

def create_dataloader(train_dataset_all, train_batch_size=1):
    return torch.utils.data.DataLoader(train_dataset_all, batch_size=train_batch_size, shuffle=True)


def training_function(hyperparameters,train_dataset_all, text_encoder, vae, unet, tokenizer, new_placeholder_tokens, dry_run):
    logger = logging.getLogger(__name__)

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear", 
        num_train_timesteps=1000, 
        tensor_format="pt")

    train_batch_size = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    learning_rate = hyperparameters["learning_rate"]
    max_train_steps = hyperparameters["max_train_steps"]
    output_dir = hyperparameters["output_dir"]

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    train_dataloader = create_dataloader(train_dataset_all, train_batch_size)

    if hyperparameters["scale_lr"]:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,)

    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader)

    # Move vae and unet to device
    vae.to(accelerator.device)
    unet.to(accelerator.device)

    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset_all)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    new_placeholder_token_ids = get_new_placeholder_token_ids(tokenizer, new_placeholder_tokens)

    for epoch in range(num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    grads = text_encoder.module.get_input_embeddings().weight.grad
                else:
                    grads = text_encoder.get_input_embeddings().weight.grad
                
                # Get the index for tokens that we want to zero the grads for
                index_all_token = torch.arange(len(tokenizer))
                for ndx, new_placeholder_token_id in enumerate(new_placeholder_token_ids):
                    if ndx == 0:
                        index_grads_to_zero = index_all_token != new_placeholder_token_id
                    else:
                        index_grads_to_zero &= index_all_token != new_placeholder_token_id
                grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            loss_val = loss.detach().item()
            logs = {"loss": loss_val}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if dry_run:
        return 

    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline(
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=PNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
            ),
            safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        )
        pipeline.save_pretrained(output_dir)

        # Also save the newly trained embeddings
        learned_embeds_dict = {}
        for _id, _token in zip(new_placeholder_token_ids,new_placeholder_tokens):
            learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[_id]
            learned_embeds_dict = {_token: learned_embeds.detach().cpu()}
        torch.save(learned_embeds_dict, os.path.join(output_dir, "learned_embeds_multiple.bin"))