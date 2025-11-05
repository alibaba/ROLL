import numpy as np
import torch
import types
import json
import gc
import os
import imageio
import queue

from concurrent.futures import ThreadPoolExecutor
from torchvision.io import write_video
from typing import List, Optional
from datetime import datetime
from einops import reduce
from peft import LoraConfig, inject_adapter_in_model, get_peft_model

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, TeaCache, TemporalTiler_BCTHW
from diffsynth.models.wan_video_motion_controller import WanMotionControllerModel
from diffsynth.models.wan_video_vace import VaceWanModel

from diffsynth.trainers.utils import DiffusionTrainingModule
from diffsynth.utils import ModelConfig, PipelineUnit
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d

from roll.pipeline.diffusion.reward_fl.face_tools import FaceAnalysis, Face
from roll.pipeline.diffusion.reward_fl.wan_video_vae import WanVideoVAE
from roll.pipeline.diffusion.reward_fl.euler import EulerScheduler
from roll.platforms import current_platform


def vae_output_to_videotensor(vae_output, pattern="B C T H W", min_value=-1, max_value=1):
    # process vae_output to videotensor
    if pattern != "T H W C":
        vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
    video_tensor = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
    return video_tensor


def training_loss(self, **inputs):
    self.scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)

    inputs["latents"] = self.generate_noise(inputs["latents"].shape, seed=24, rand_device=self.device)
    
    timesteps = self.scheduler.timesteps
    models = {name: getattr(self, name) for name in self.in_iteration_models}
    kl_loss_total = 0
    kl_steps = 0
    for i, timestep in enumerate(timesteps[:]):
        current_timestep = timestep.expand([1])
        current_sigma = current_timestep / self.scheduler.num_train_timesteps
        # switch dit if necessary
        if timestep.item() < 0.9 * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
            models["dit"] = self.dit2
        elif timestep.item() >= 0.9 * self.scheduler.num_train_timesteps and self.dit is not None and not models["dit"] is self.dit:
            models["dit"] = self.dit
        
        # Timestep
        timestep = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device)
        inputs["timestep"] = timestep
        inputs["less_mid_step"] = True
        
        # Inference
        if i < self.mid_timestep:
            inputs["less_mid_step"] = True
            with torch.no_grad():
                model_pred = self.model_fn(**models, **inputs)
        else:
            inputs["less_mid_step"] = False
            model_pred = self.model_fn(**models, **inputs)
            with torch.no_grad():
                inputs["less_mid_step"] = True
                models["dit"].disable_adapter_layers()
                model_pred_no_lora = self.model_fn(**models, **inputs)
            models["dit"].enable_adapter_layers()
            kl_loss_step = torch.mean(
                    (model_pred.float() - model_pred_no_lora.float()) ** 2 / (2 * current_sigma.float() ** 2)
                )
            kl_loss_total += kl_loss_step
            kl_steps += 1

        noise_pred = model_pred
        
        # Scheduler denoise
        if i < self.final_timestep:
            inputs["latents"] = self.scheduler.step(noise_pred, timestep, inputs["latents"]).to(torch.bfloat16)
        else:
            inputs["latents"] = self.scheduler.step(noise_pred, timestep, inputs["latents"]).to(torch.bfloat16)
            break
        
        if "first_frame_latents" in inputs:
            inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    kl_loss = kl_loss_total / kl_steps if kl_steps > 0 else 0.0
    video_decoded = self.vae.decode(inputs["latents"], device=self.device, tiled=True)
    return video_decoded, kl_loss


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths,
        reward_model_path,
        tokenizer_path,
        trainable_models,
        model_id_with_origin_paths=None,
        lora_base_model=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.9,
        num_inference_steps=8,
        mid_timestep=4,
        final_timestep=7,
        **kwargs
    ):
        super().__init__()
        # Load models
        model_configs : List[ModelConfig] = []
        if model_paths is not None:
            with open(model_paths, 'r', encoding='utf-8') as f:
                model_paths = json.load(f)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            tokenizer_config=ModelConfig(path=tokenizer_path),
            redirect_common_files=False
        )
        
        self.apply_patches()
        
        face_model = FaceAnalysis(root=reward_model_path, device=current_platform.device_type)
        
        # 将冻结模型存入一个普通字典中 PyTorch 不会注册普通字典中的 nn.Module
        self.frozen_dependencies = {
            'face_model': face_model,
        }

        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.pipe.num_inference_steps = num_inference_steps
        self.pipe.mid_timestep = mid_timestep
        self.pipe.final_timestep = final_timestep
        self.io_executor = ThreadPoolExecutor(max_workers=1) 
        self.io_queue = queue.Queue()
        self.output_path = './output_video/'

        self.global_step = 0

    def apply_patches(self):

        # apply patches
        self.pipe.units.append(WanVideoUnit_Face())
        self.pipe.scheduler = EulerScheduler(num_train_timesteps=1000, shift=5, device=current_platform.device_type)
        vae_state_dict = self.pipe.vae.state_dict()
        self.pipe.vae = WanVideoVAE()
        self.pipe.vae.load_state_dict(vae_state_dict, strict=True)
        self.pipe.model_fn = model_fn_wan_video
        self.pipe.training_loss = types.MethodType(training_loss, self.pipe)


    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "face_model": self.frozen_dependencies['face_model'],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)

        face_embeddings = inputs['face_embeddings'].to(device=self.pipe.device, dtype=self.pipe.torch_dtype)

        # step1: forward latents + vae decode
        video_decoded, kl_loss = self.pipe.training_loss(**inputs)

        # step2: get video_tensor
        video_tensor = vae_output_to_videotensor(video_decoded)

        # step3: true video submit
        self.vae_video2_submit(video_tensor, self.output_path)

        video_decoded = video_decoded[0].permute(1, 0, 2, 3) # (C, T, H, W) -> (T, C, H, W)
        print(f'video decode shape: {video_decoded.shape}')
        
        video_decoded = torch.clamp(video_decoded, min=-1, max=1)
        self.frozen_dependencies['face_model'].detection_model.torch_model.to(self.pipe.device)
        self.frozen_dependencies['face_model'].arcface_model.torch_model.to(self.pipe.device)

        id_embeds, id_masked = [], []
        face_num = 0
        for f in video_decoded:
            f = f.float()
            bboxes, kpss = self.frozen_dependencies['face_model'].detection_model.detect(f)
            if bboxes.shape[0] > 0:
                indexed_bboxes = [(i, x) for i, x in enumerate(bboxes)]
                sorted_bboxes = sorted(indexed_bboxes, key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]))
                max_index, max_bbox = sorted_bboxes[-1]
                kps = kpss[max_index]
                face = Face(bbox=bboxes[max_index][0:4], kps=kps, det_score=bboxes[max_index][4])
                id_embeds.append(self.frozen_dependencies['face_model'].arcface_model.get(f, face))
                id_masked.append(1)
                face_num += 1
            else:
                id_embeds.append(torch.zeros(512).to(self.pipe.device))
                id_masked.append(0)
        assert face_num > 0, f"face_num must be greater than 0"

        id_embeds = torch.stack(id_embeds).unsqueeze(0)
        id_masked = torch.tensor(id_masked).unsqueeze(0).to(self.pipe.device)

        face_score = self.frozen_dependencies['face_model'].pool_embedding_loss(id_embeds, face_embeddings, id_masked)
        print(f"{face_score=}")
        
        face_score = face_score.to(self.pipe.device)

        del video_tensor, video_decoded
        gc.collect()
        current_platform.empty_cache()

        loss = -(face_score.bfloat16()-0.54)/0.16 * 0.1
        loss += kl_loss
        
        loss = loss.to(self.pipe.device)
        
        print(f'loss: {loss.float().detach().cpu().item()}')
        self.global_step = self.global_step + 1

        gc.collect()
        current_platform.empty_cache()

        return loss, face_score, kl_loss

    def vae_video2_submit(self, video_tensor, output_path):
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            while not self.io_queue.empty():
                try:
                    future = self.io_queue.get_nowait()
                    if not future.done():
                        self.io_queue.put(future)
                        return
                except queue.Empty:
                    break

            step = self.global_step
            video_tmp = video_tensor.clone().detach().cpu().float().numpy()
            video_tmp = video_tmp.round().astype(np.uint8)
            
            
            future = self.io_executor.submit(
                self._save_video_background, video_tmp, output_path, rank, step
            )
            self.io_queue.put(future)


    def _save_video_background(self, video_data, output_path, rank, step):
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            save_oss_dir = os.path.join(output_path, 'decode_videos')
            os.makedirs(save_oss_dir, exist_ok=True)
            video_filename = f'video_rank{rank}_iter{step}_time{timestamp}.mp4'
            save_video(video_data, video_filename, save_oss_dir, save_to_oss=True)
        except Exception as e:
            print(f"Error during background video saving: {e}")
            raise e

    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = get_peft_model(model, lora_config)
        return model

class WanVideoUnit_Face(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("input_video", "face_model"))

    def process(self, pipe: WanVideoPipeline, input_video, face_model):
        input_video = pipe.preprocess_video(input_video) # (1, 3, F, H, W)
        input_video = input_video[0].transpose(0, 1)
        
        face_embeds, face_masked= [], []
        # input_video (F, 3, H, W) 数值范围(-1, 1)
        for f in input_video:
            f = f.float().to(pipe.device) #(3, h, w)
            face_model.detection_model.torch_model.to(pipe.device)
            face_model.arcface_model.torch_model.to(pipe.device)
            bboxes, kpss = face_model.detection_model.detect(f)
            if bboxes.shape[0] > 0:
                indexed_bboxes = [(i, x) for i, x in enumerate(bboxes)]
                sorted_bboxes = sorted(indexed_bboxes, key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]))
                max_index, max_bbox = sorted_bboxes[-1]
                kps = kpss[max_index]
                face = Face(bbox=bboxes[max_index][0:4], kps=kps, det_score=bboxes[max_index][4])
                embedding = face_model.arcface_model.get(f, face)
                face_embeds.append(embedding.cpu())
                face_masked.append(1)
            else:
                face_embeds.append(torch.zeros(512))
                face_masked.append(0)
        face_embeds = torch.stack(face_embeds).unsqueeze(0)
        face_masked = torch.tensor(face_masked).unsqueeze(0).to(pipe.device)
        return {"face_embeddings": face_embeds, "face_masked":face_masked}


def save_video(video_frames, save_video_basename, output_oss_dir, save_to_oss=True):
    if video_frames.shape[0] == 1:  # T=1时保存为图像
        local_output_path = f'{save_video_basename}.png' if not save_video_basename.endswith('.png') else save_video_basename
        oss_output = f'{output_oss_dir}/{local_output_path}'
        imageio.imwrite(oss_output, video_frames[0])  # 取单帧保存
    else:
        local_output_path = f'{save_video_basename}.mp4' if not save_video_basename.endswith('.mp4') else save_video_basename
        oss_output = f'{output_oss_dir}/{local_output_path}'
        write_video(local_output_path, video_frames, fps=16, options={'crf': '10'})
    if save_to_oss:
        os.system(f'cp {local_output_path} {oss_output}')
        os.system(f'rm -rf {local_output_path}')


def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    tea_cache: TeaCache = None,
    less_mid_step: bool = True,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    recompute_num_layers: Optional[int] = 1,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        with torch.amp.autocast(current_platform.device_type, dtype=torch.float32):
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
            t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
            assert t.dtype == torch.float32 and t_mod.dtype == torch.float32
        t, t_mod = t.to(dtype=torch.bfloat16), t_mod.to(dtype=torch.bfloat16)
        assert t.dtype == torch.bfloat16 and t_mod.dtype == torch.bfloat16
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    # Add camera control
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs)
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)
    
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x
