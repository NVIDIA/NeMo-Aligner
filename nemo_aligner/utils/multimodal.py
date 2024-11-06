import os
from transformers import AutoConfig, PixtralImageProcessor
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import from_parallel_logits_to_logprobs
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo_aligner.models.mm.vision.vit import VisionTransformer, VisionLanguageAdapter
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

def pad_and_batch_samples(samples, padding_value=-1):
    # Determine the maximum dimensions for padding
    max_num_images = max(sample.shape[0] for sample in samples)
    max_channels = max(sample.shape[1] for sample in samples)  # Should be the same for all if constant
    max_height = max(sample.shape[2] for sample in samples)
    max_width = max(sample.shape[3] for sample in samples)
    
    # Pad each sample to match the maximum dimensions
    padded_samples = []
    for sample in samples:
        num_images, num_channels, height, width = sample.shape
        # Calculate padding for each dimension
        pad_num_images = max_num_images - num_images
        pad_height = max_height - height
        pad_width = max_width - width
        
        # Padding format: (W_left, W_right, H_top, H_bottom, Depth_front, Depth_back)
        padding = (0, pad_width, 0, pad_height, 0, 0, 0, pad_num_images)
        padded_sample = F.pad(sample, padding, value=padding_value)
        padded_samples.append(padded_sample)
    
    # Stack padded samples along the batch dimension
    batched_tensor = torch.stack(padded_samples)
    return batched_tensor


def remove_padding(batched_tensor, padding_val=-1):
    result = []
    # Iterate over each sample in the batch
    for sample in batched_tensor:
        unpadded_images = []
        # Iterate over each image in the num_images dimension
        for image in sample:
            # Remove padding in height and width dimensions
            non_padded_image = image[
                :, 
                ~(image == padding_val).all(dim=(0, 2)),  # Filter out padded rows (height)
                :][:, :, ~(image == padding_val).all(dim=(0, 1))]  # Filter out padded columns (width)
            unpadded_images.append(non_padded_image)
        result.append(unpadded_images)
    return result

class MultimodalMixin:
    vit_safetensor_file = "model.safetensors"
    def __init__(self, cfg, trainer):
        # Pass cfg and trainer to MegatronGPTModel
        super().__init__(cfg, trainer=trainer)
        
        self.is_multimodal = cfg.get("mm_cfg") is not None  # Check if multimodal config exists

        if self.is_multimodal:
            # Media tokens
            self.image_token = cfg.mm_cfg.image_token
            self.image_token_id = self.tokenizer.token_to_id(self.image_token)
            self.image_patch_token = cfg.mm_cfg.image_patch_token
            self.image_break_token = cfg.mm_cfg.image_break_token
            self.image_end_token = cfg.mm_cfg.image_end_token
            
            vision_encoder_name = cfg.mm_cfg.vision_encoder.from_pretrained
            llm_hidden_size = cfg.hidden_size
            # Create the vision encoder
            config = AutoConfig.from_pretrained(vision_encoder_name)
            self.vision_encoder = VisionTransformer(config).to(torch.bfloat16).eval()
            # Load vision encoder weights
            vit_ckpt_file = os.path.join(vision_encoder_name, self.vit_safetensor_file)
            self.vision_encoder.load_state_dict(load_file(vit_ckpt_file))
            # Freeze vision encoder weights
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
                    
            # Create the multimodal adapter
            vit_hidden_size = config.hidden_size
            if "bf16" in cfg.precision:
                self.vision_langauge_adapter = VisionLanguageAdapter(vit_hidden_size, llm_hidden_size).to(torch.bfloat16)
            elif "fp16" in cfg.precision:
                self.vision_langauge_adapter = VisionLanguageAdapter(vit_hidden_size, llm_hidden_size).to(torch.float16)

            # Create image processor
            self.image_processor = PixtralImageProcessor.from_pretrained(
                    cfg.mm_cfg.vision_encoder.from_pretrained, torch_dtype=torch.bfloat16
                )
        
    def encode_vision(self, images):
        with torch.no_grad():
            image_outputs = self.vision_encoder(images, output_hidden_states=True)
            image_features = image_outputs.hidden_states[-1] # hardcoded for now
        image_features = self.vision_langauge_adapter(image_features)
        return image_features
    
    def replace_media_embeddings(self, input_ids, inputs_embeds, media):
        if media is None:
            return inputs_embeds
        print(f"{input_ids.shape = }")
        # Create mask for media tokens
        special_image_mask = (input_ids == self.image_token_id)  # Shape: (batch_size, sequence_length)

        # Find indices of media tokens
        media_indices = special_image_mask.nonzero(as_tuple=False)  # Shape: (num_media_tokens, 2)

        # Encode media features with gradients
        media_features = self.encode_vision(media).squeeze(0)  # Shape: (batch_size, num_media_tokens, hidden_size)
        print(f"{media_features.shape = }")
        # Clone inputs_embeds to maintain gradient flow
        updated_embeds = inputs_embeds.clone()

        if media_indices.size(0) > 0:
            # Replace the embeddings at media token positions
            print(f"{updated_embeds.shape = }")
            updated_embeds[media_indices[:, 0], media_indices[:, 1], :] = media_features
        else:
            # If there are no media tokens, add a dummy computation to ensure media_features is used in the computation graph
            updated_embeds = updated_embeds + media_features.sum(dim=(1, 2), keepdim=True) * 0

        return updated_embeds
    
    def old_forward(self, input_ids, media=None, **kwargs):
        # Only apply multimodal processing if multimodal is enabled
        if self.is_multimodal and media is not None:
            # Check if reduce_scatter_embeddings is enabled in the embedding forward function
            apply_reduce_scatter = getattr(self.model.language_model.embedding.word_embeddings, 'reduce_scatter_embeddings', False)
            word_embeddings = self.model.language_model.embedding.word_embeddings.forward(input_ids, **kwargs)
            
            # Replace embeddings with vision features
            word_embeddings = self.replace_media_embeddings(input_ids, word_embeddings, media)

            # Scatter embeddings back to each TP rank if needed
            if apply_reduce_scatter:
                word_embeddings = self.model.language_model.embedding.word_embeddings._apply_reduce_scatter(word_embeddings)
                self.model.language_model.embedding.word_embeddings.reduce_scatter_embeddings = True
            
            return super().forward(decoder_input=word_embeddings, **kwargs)
        else:
            # Standard forward pass when multimodal is not used
            return super().forward(input_ids=input_ids, **kwargs)

    def forward(self, model_, **kwargs):
        """
        Forward pass for multimodal processing.

        Args:
            input_ids (torch.LongTensor): Tensor of token IDs with shape [batch_size, seq_length].
            media (torch.FloatTensor, optional): Tensor of media inputs with shape [batch_size, n_img, 3, H, W].
                Padded with -1 for varying numbers and sizes of images.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output logits from the language model.
        """
        media = kwargs.pop("media")
        input_ids = kwargs.pop("input_ids")
        model = model_.module if hasattr(model_, "module") else model_
        if self.is_multimodal and media is not None:            
            # Step 1: Remove padding from media tensor to get a list of images per sample
            # media: [batch_size, n_img, 3, H, W]
            media = media.to(input_ids.dtype)
            unpadded_media = remove_padding(media, padding_val=-1)  # List[List[torch.Tensor]]
            # unpadded_media[i] -> List of images for sample i

            # Step 2: Get word embeddings from the language model
            # Assuming self.language_model.embedding.word_embeddings is accessible
            word_embeddings = model.embedding(input_ids=input_ids, position_ids=kwargs["position_ids"])  # [batch_size, seq_length, hidden_dim]

            # Step 3: Iterate over each sample in the batch to replace media embeddings
            for batch_idx in range(input_ids.size(0)):
                images = unpadded_media[batch_idx]  # List[torch.Tensor]
                print(f"{images[0].shape = }")
                # Step 3a: Process images with the vision encoder to get image features
                #image_features = self.encode_vision(images)  # [n_img, hidden_dim]

                # Step 3b: Slice to retain batch dimension
                # Retain batch dimension by slicing [batch_idx:batch_idx+1]
                sliced_input_ids = input_ids[batch_idx:batch_idx+1]  # [1, seq_length]
                sliced_word_embeddings = word_embeddings[batch_idx:batch_idx+1]  # [1, seq_length, hidden_dim]

                # Step 3c: Replace media embeddings
                # Assuming replace_media_embeddings expects [1, seq_length, hidden_dim]
                updated_embeddings = self.replace_media_embeddings(
                    input_ids=sliced_input_ids,
                    inputs_embeds=sliced_word_embeddings,
                    media=images
                )  # [1, seq_length, hidden_dim]

                # Step 3d: Assign the updated embeddings back to word_embeddings
                word_embeddings[batch_idx:batch_idx+1] = updated_embeddings  # [batch_size, seq_length, hidden_dim]

            # Step 4: Check if reduce_scatter_embeddings is enabled
            apply_reduce_scatter = getattr(model.embedding.word_embeddings, 'reduce_scatter_embeddings', False)
            if apply_reduce_scatter:
                word_embeddings = model.embedding.word_embeddings._apply_reduce_scatter(word_embeddings)
                model.embedding.word_embeddings.reduce_scatter_embeddings = True

            # Step 5: Pass the modified embeddings to the language model
            # Assuming the language model's forward can accept decoder_input
            return model.forward(decoder_input=word_embeddings, **kwargs)
        else:
            # Standard forward pass when multimodal is not used
            return model.forward(input_ids=input_ids, **kwargs)

class DPOMixin:
    """
    Mixin class for Direct Preference Optimization (DPO) alignment.
    Overrides the get_forward_output_and_loss_func method to incorporate multimodal inputs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override the get_forward_output_and_loss_func method
        self.get_forward_output_and_loss_func = self.dpo_get_forward_output_and_loss_func

    def dpo_get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            batch = next(dataloader_iter)
            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                # Always give the attention mask
                required_keys.add("attention_mask")

                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(("chosen", "rejected", "position_ids"))
                    if batch.get("media") is not None:
                        required_keys.add("media")

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(
                        (
                            "ref_policy_log_probs_chosen",
                            "ref_policy_log_probs_rejected",
                            "chosen_labels",
                            "rejected_labels",
                            "chosen_rewards",
                            "rejected_rewards",
                        )
                    )

            # Move required keys to GPU, others set to None
            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            tokens, labels, ref_logprobs, gt_rewards = None, None, None, None
            if batch["chosen"] is not None and batch["rejected"] is not None:
                tokens = torch.cat((batch["chosen"], batch["rejected"]), dim=0)

            if batch["chosen_labels"] is not None and batch["rejected_labels"] is not None:
                labels = torch.cat((batch["chosen_labels"], batch["rejected_labels"]), dim=0)

            if batch["ref_policy_log_probs_chosen"] is not None and batch["ref_policy_log_probs_rejected"] is not None:
                ref_logprobs = torch.cat(
                    (batch["ref_policy_log_probs_chosen"], batch["ref_policy_log_probs_rejected"]), dim=0
                )

            if batch["chosen_rewards"] is not None and batch["rejected_rewards"] is not None:
                gt_rewards = torch.cat((batch["chosen_rewards"], batch["rejected_rewards"]), dim=0)

            if batch.get("media") is not None:
                media = torch.cat((batch["media"], batch["media"]), dim=0)  # Adjust concatenation as needed
            else:
                media = None

            # Handle position_ids and attention_mask
            attention_mask = batch["attention_mask"][0:1]

            # Model forward pass using the shared forward method from MultimodalMixin
            forward_args = {
                "input_ids": tokens,
                "position_ids": batch["position_ids"],
                "attention_mask": attention_mask,
                "labels": None,
                "loss_mask": None,
                "media": media,
            }

            # Handle checkpointing and loss_mask based on configuration
            if not self.mcore_gpt:
                forward_args["checkpoint_activations_all_layers"] = checkpoint_activations_all_layers
                if not self.use_loss_mask:
                    forward_args.pop("loss_mask")
            else:
                forward_args.pop("loss_mask")

            # Call the overridden forward method from MultimodalMixin
            #output_tensor = model(**forward_args)
            output_tensor = self.forward(model, **forward_args)

            # Cast to appropriate dtype if not in the last pipeline stage
            if not parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor.to(dtype=self.autocast_dtype)

            def loss_func(output_tensor_inner):
                if validation_step and not self.cfg.data.get("validation_drop_last", True):
                    raise NotImplementedError("DPO does not support validation when cfg.data.drop_last=False")

                per_token_logps = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=output_tensor_inner, target=labels, higher_stability=True
                )

                preference_loss, acc_chosen = self.loss_func(
                    per_token_logps,
                    ref_logprobs,
                    labels[:, 1:],
                    gt_rewards,
                    average_log_probs=self.preference_avg_log_probs,
                )

                sft_loss = torch.zeros_like(preference_loss)
                if self.sft_loss_weight != 0:
                    sft_loss = self.sft_loss_func(
                        per_token_logps, labels[:, 1:], average_log_probs=self.sft_avg_log_probs
                    )
                loss = self.preference_loss_weight * preference_loss + self.sft_loss_weight * sft_loss

                (
                    reduced_loss,
                    reduced_preference_loss,
                    reduced_sft_loss,
                    reduced_acc,
                ) = average_losses_across_data_parallel_group([loss, preference_loss, sft_loss, acc_chosen])

                out_chosen, out_rejected = self.gather_and_split_rewards(
                    per_token_logps, ref_logprobs, labels, average_log_probs=self.preference_avg_log_probs
                )

                return (
                    loss,
                    {
                        "avg": reduced_loss,
                        "avg_sft_loss": reduced_sft_loss,
                        "avg_preference_loss": reduced_preference_loss,
                        "acc": reduced_acc,
                        "out_chosen": out_chosen,
                        "out_rejected": out_rejected,
                    },
                )

            return output_tensor, loss_func

        return fwd_output_and_loss_func