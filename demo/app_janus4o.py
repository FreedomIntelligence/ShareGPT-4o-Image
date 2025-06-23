import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image

import numpy as np
import os
import time
# import spaces  # Import spaces for ZeroGPU compatibility


# Load model and processor
model_path = "FreedomIntelligence/Janus-4o-7B"
config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'
vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                             language_config=language_config,
                                             trust_remote_code=True)
if torch.cuda.is_available():
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
else:
    vl_gpt = vl_gpt.to(torch.float16)

vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.inference_mode()
# @spaces.GPU(duration=120) 
# Multimodal Understanding function
def multimodal_understanding(image, question, seed, top_p, temperature):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    pil_images = [Image.fromarray(image)]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
    
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer


def generate(input_ids,
             width,
             height,
             temperature: float = 1,
             parallel_size: int = 5,
             cfg_weight: float = 5,
             image_token_num_per_image: int = 576,
             patch_size: int = 16):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(cuda_device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(cuda_device)

    pkv = None
    for i in range(image_token_num_per_image):
        with torch.no_grad():
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds,
                                                use_cache=True,
                                                past_key_values=pkv)
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)

            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    

    patches = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                 shape=[parallel_size, 8, width // patch_size, height // patch_size])

    return generated_tokens.to(dtype=torch.int), patches

def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img



@torch.inference_mode()
# @spaces.GPU(duration=120)  # Specify a duration to avoid timeout
def generate_image(prompt,
                   seed=None,
                   guidance=5,
                   t2i_temperature=1.0):
    # Clear CUDA cache and avoid tracking gradients
    torch.cuda.empty_cache()
    # Set the seed for reproducible results
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    width = 384
    height = 384
    parallel_size = 5
    
    with torch.no_grad():
        messages = [{'role': '<|User|>', 'content': prompt},
                    {'role': '<|Assistant|>', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=messages,
                                                                   sft_format=vl_chat_processor.sft_format,
                                                                   system_prompt='')
        text = text + vl_chat_processor.image_start_tag
        
        input_ids = torch.LongTensor(tokenizer.encode(text))
        output, patches = generate(input_ids,
                                   width // 16 * 16,
                                   height // 16 * 16,
                                   cfg_weight=guidance,
                                   parallel_size=parallel_size,
                                   temperature=t2i_temperature)
        images = unpack(patches,
                        width // 16 * 16,
                        height // 16 * 16,
                        parallel_size=parallel_size)

        return [Image.fromarray(images[i]).resize((768, 768), Image.LANCZOS) for i in range(parallel_size)]
        

# NOTE: MOD CZY BEG
from dataclasses import dataclass

@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)

def process_image(pil_images, vl_chat_processor):
    images = pil_images
    images_outputs = vl_chat_processor.image_processor(images, return_tensors="pt")
    return images_outputs['pixel_values']

def generate_image_v2v_mask_v3(question, input_image, temperature = 1, parallel_size = 1, cfg_weight = 5, cfg_weight2 = 5):
    torch.cuda.empty_cache()

    input_img_tokens = vl_chat_processor.image_start_tag + vl_chat_processor.image_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag + vl_chat_processor.image_start_tag + vl_chat_processor.pad_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag
    output_img_tokens = vl_chat_processor.image_start_tag 

    pre_data = []
    input_image = Image.fromarray(input_image)
    input_images = [input_image]
    img_len = len(input_images)
    prompts = input_img_tokens * img_len + question
    conversation = [
                    {"role": "<|User|>", "content": prompts},
                    {"role": "<|Assistant|>", "content": ""}
                ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )

    sft_format = sft_format + output_img_tokens

    mmgpt = vl_gpt

    image_token_num_per_image = 576
    img_size = 384
    patch_size = 16

    with torch.inference_mode():
        input_image_pixel_values = process_image(input_images,vl_chat_processor).to(torch.bfloat16).cuda()
        quant_input, emb_loss_input, info_input = mmgpt.gen_vision_model.encode(input_image_pixel_values)
        image_tokens_input = info_input[2].detach().reshape(input_image_pixel_values.shape[0], -1)
        image_embeds_input = mmgpt.prepare_gen_img_embeds(image_tokens_input)

        input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))

        encoder_pixel_values = process_image(input_images,vl_chat_processor).cuda()
        tokens = torch.zeros((parallel_size*3, len(input_ids)), dtype=torch.long)
        for i in range(parallel_size*3):
            tokens[i, :] = input_ids
            if i % 3 == 2:
                tokens[i, 1:-1] = vl_chat_processor.pad_id
                pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=encoder_pixel_values, input_ids=tokens[i-2], num_image_tokens=[vl_chat_processor.num_image_tokens] * img_len))
                pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=encoder_pixel_values, input_ids=tokens[i-1], num_image_tokens=[vl_chat_processor.num_image_tokens] * img_len))
                pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=None, input_ids=tokens[i], num_image_tokens=[]))

        prepare_inputs = vl_chat_processor.batchify(pre_data)

        inputs_embeds = mmgpt.prepare_inputs_embeds(
                    input_ids=tokens.cuda(),
                    pixel_values=prepare_inputs['pixel_values'].to(torch.bfloat16).cuda(),
                    images_emb_mask=prepare_inputs['images_emb_mask'].cuda(),
                    images_seq_mask=prepare_inputs['images_seq_mask'].cuda()
                )

        image_gen_indices = (tokens == vl_chat_processor.image_end_id).nonzero()

        for ii, ind in enumerate(image_gen_indices):
            if ii % 4 == 0:
                offset = ind[1] + 2
                inputs_embeds[ind[0],offset: offset+image_embeds_input.shape[1],:] = image_embeds_input[(ii // 2) % img_len]

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        for i in range(image_token_num_per_image):
            outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state

            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond_full = logits[0::3, :]
            logit_cond_part = logits[1::3, :]
            logit_uncond = logits[2::3, :]

            logit_cond = (logit_cond_full + cfg_weight2 * (logit_cond_part)) / (1 + cfg_weight2)
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)

            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        images = [Image.fromarray(visual_img[i]) for i in range(parallel_size)]

        return images


def text_and_image_to_image(prompt, image, seed=None, guidance1=5, guidance2=5, t2i_temperature=1.0):
    torch.cuda.empty_cache()
    
    if seed is not None:
        seed = int(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    output_images = generate_image_v2v_mask_v3(
        question=prompt,
        input_image=image,
        temperature=t2i_temperature,
        parallel_size=5,
        cfg_weight=guidance1,
        cfg_weight2=guidance2
    )
    
    return output_images

# Gradio interface
with gr.Blocks() as demo:
        
    gr.Markdown(value="# Text-to-Image Generation")

    with gr.Row():
        cfg_weight_input = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight")
        t2i_temperature = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.05, label="temperature")

    prompt_input = gr.Textbox(label="Prompt. (Prompt in more detail can help produce better images!)")
    seed_input = gr.Number(label="Seed (Optional)", precision=0, value=12345)

    generation_button = gr.Button("Generate Images")

    image_output = gr.Gallery(label="Generated Images", columns=2, rows=2, height=300)
    
    generation_button.click(
        fn=generate_image,
        inputs=[prompt_input, seed_input, cfg_weight_input, t2i_temperature],
        outputs=image_output
    )

    gr.Markdown(value="# Text and Image to Image Generation")
    
    with gr.Row():
        image_input = gr.Image(label="Input Image")
        prompt_input = gr.Textbox(label="Text Prompt")

    with gr.Row():
        cfg_weight_input1 = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight 1")
        cfg_weight_input2 = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight 2")
        
    with gr.Row():
        t2i_temperature = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.05, label="temperature")
        seed_input_ti2i = gr.Number(label="Seed (Optional)", precision=0, value=12345)

    generation_button = gr.Button("Generate Image")
    image_output = gr.Gallery(label="Generated Images", columns=2, rows=2, height=300)

    generation_button.click(
        text_and_image_to_image,
        inputs=[prompt_input, image_input, seed_input_ti2i, cfg_weight_input1, cfg_weight_input2, t2i_temperature],
        outputs=image_output
    )

demo.launch(share=True)
# demo.queue(concurrency_count=1, max_size=10).launch(server_name="0.0.0.0", server_port=37906, root_path="/path")