import os
import gradio as gr
import subprocess
import spaces
import ctypes
import shlex
import torch
import argparse
print(f'gradio version: {gr.__version__}')

# Add command line argument parsing
parser = argparse.ArgumentParser(description='DiMeR Demo')
parser.add_argument('--ui_only', action='store_true', help='Only load the UI interface, do not initialize models (for UI debugging)')
args = parser.parse_args()

UI_ONLY_MODE = args.ui_only
print(f"UI_ONLY_MODE: {UI_ONLY_MODE}")

# if not UI_ONLY_MODE:
#     subprocess.run(
#         shlex.split(
#             "pip install ./custom_diffusers --force-reinstall --no-deps"
#         )
#     )
#     subprocess.run(
#         shlex.split(
#             "pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html"
#         )
#     )

#     subprocess.run(
#         shlex.split(
#             "pip install ./extension/nvdiffrast-0.3.1+torch-py3-none-any.whl --force-reinstall --no-deps"
#         )
#     )

#     subprocess.run(
#         shlex.split(
#             "pip install ./extension/renderutils_plugin-0.1.0-cp310-cp310-linux_x86_64.whl --force-reinstall --no-deps"
#         )
#     )

# Status variables for tracking if detailed prompt and image have been generated
generated_detailed_prompt = False
generated_image = False

# def install_cuda_toolkit():
#     CUDA_TOOLKIT_URL = "https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run"
#     CUDA_TOOLKIT_FILE = "/tmp/%s" % os.path.basename(CUDA_TOOLKIT_URL)
#     subprocess.call(["wget", "-q", CUDA_TOOLKIT_URL, "-O", CUDA_TOOLKIT_FILE])
#     subprocess.call(["chmod", "+x", CUDA_TOOLKIT_FILE])
#     subprocess.call([CUDA_TOOLKIT_FILE, "--silent", "--toolkit"])

#     os.environ["CUDA_HOME"] = "/usr/local/cuda"
#     os.environ["PATH"] = "%s/bin:%s" % (os.environ["CUDA_HOME"], os.environ["PATH"])
#     os.environ["LD_LIBRARY_PATH"] = "%s/lib:%s" % (
#         os.environ["CUDA_HOME"],
#         "" if "LD_LIBRARY_PATH" not in os.environ else os.environ["LD_LIBRARY_PATH"],
#     )
#     # Fix: arch_list[-1] += '+PTX'; IndexError: list index out of range
#     os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
#     print("==> finished installation")

# # Only execute CUDA installation in non-UI debug mode
# if not UI_ONLY_MODE:
#     install_cuda_toolkit()

@spaces.GPU
def check_gpu():
    # if "CUDA_VISIBLE_DEVICES" in os.environ:
    #     del os.environ["CUDA_VISIBLE_DEVICES"]
    # os.environ['CUDA_HOME'] = '/usr/local/cuda-12.1'
    # os.environ['PATH'] += ':/usr/local/cuda-12.1/bin'
    # os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda-12.1/lib64:" + os.environ.get('LD_LIBRARY_PATH', '')
    subprocess.run(['nvidia-smi'])  # Test if CUDA is available
    print(f"torch.cuda.is_available:{torch.cuda.is_available()}")
    print("Device count:", torch.cuda.device_count()) 

    # test nvdiffrast
    import nvdiffrast.torch as dr
    dr.RasterizeCudaContext(device="cuda:0")
    print("nvdiffrast initialized successfully")       
    

# Only check GPU in non-UI debug mode
if not UI_ONLY_MODE:
    check_gpu()


import base64
import re
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, '../')))
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '32'

import shutil
import json
import requests
import shutil
import threading
from PIL import Image
import time
import trimesh

import random
import time
import numpy as np

# Only import video rendering module and initialize models in non-UI debug mode
if not UI_ONLY_MODE:
    from video_render import render_video_from_obj

    access_token = os.getenv("HUGGINGFACE_TOKEN")
    from pipeline.kiss3d_wrapper import init_wrapper_from_config, run_text_to_3d, run_image_to_3d, image2mesh_preprocess, image2mesh_main

# Add logo file path and hyperlinks
LOGO_PATH = "app_assets/logo_temp_.png"  # Update this to the actual path of your logo
ARXIV_LINK = "https://arxiv.org/pdf/2504.17670"
GITHUB_LINK = "https://github.com/lutao2021/DiMeR"

# Only initialize models in non-UI debug mode
if not UI_ONLY_MODE:
    k3d_wrapper = init_wrapper_from_config('./pipeline/pipeline_config/default.yaml')
    from models.ISOMER.scripts.utils import fix_vert_color_glb
    torch.backends.cuda.matmul.allow_tf32 = True

TEMP_MESH_ADDRESS=''

mesh_cache = None
preprocessed_input_image = None

def save_cached_mesh():
    global mesh_cache
    print('save_cached_mesh() called')
    return mesh_cache

def save_py3dmesh_with_trimesh_fast(meshes, save_glb_path=TEMP_MESH_ADDRESS, apply_sRGB_to_LinearRGB=True):
    from pytorch3d.structures import Meshes
    import trimesh

    # convert from pytorch3d meshes to trimesh mesh
    vertices = meshes.verts_packed().cpu().float().numpy()
    triangles = meshes.faces_packed().cpu().long().numpy()
    np_color = meshes.textures.verts_features_packed().cpu().float().numpy()
    if save_glb_path.endswith(".glb"):
        # rotate 180 along +Y
        vertices[:, [0, 2]] = -vertices[:, [0, 2]]

    def srgb_to_linear(c_srgb):
        c_linear = np.where(c_srgb <= 0.04045, c_srgb / 12.92, ((c_srgb + 0.055) / 1.055) ** 2.4)
        return c_linear.clip(0, 1.)
    if apply_sRGB_to_LinearRGB:
        np_color = srgb_to_linear(np_color)
    assert vertices.shape[0] == np_color.shape[0]
    assert np_color.shape[1] == 3
    assert 0 <= np_color.min() and np_color.max() <= 1, f"min={np_color.min()}, max={np_color.max()}"
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_colors=np_color)
    mesh.remove_unreferenced_vertices()
    # save mesh
    mesh.export(save_glb_path)
    if save_glb_path.endswith(".glb"):
        fix_vert_color_glb(save_glb_path)
    print(f"saving to {save_glb_path}")

@spaces.GPU
def text_to_detailed(prompt, seed=None):
    # test nvdiffrast
    import nvdiffrast.torch as dr
    dr.RasterizeCudaContext(device="cuda:0")
    print("nvdiffrast initialized successfully")

    print(f"torch.cuda.is_available():{torch.cuda.is_available()}")
    # print(f"Before text_to_detailed: {torch.cuda.memory_allocated() / 1024**3} GB")
    return k3d_wrapper.get_detailed_prompt(prompt, seed)

@spaces.GPU(duration=120)
def text_to_image(prompt, seed=None, strength=1.0,lora_scale=1.0, num_inference_steps=18, redux_hparam=None, init_image=None, **kwargs):
    # subprocess.run("rm -rf /data-nvme/zerogpu-offload/*", env={}, shell=True)
    # print(f"Before text_to_image: {torch.cuda.memory_allocated() / 1024**3} GB")
    # k3d_wrapper.flux_pipeline.enable_xformers_memory_efficient_attention()
    k3d_wrapper.renew_uuid()
    init_image = None
    # if init_image_path is not None:
    #     init_image = Image.open(init_image_path)
    subprocess.run(['nvidia-smi'])  # Test if CUDA is available
    with torch.no_grad():
        result = k3d_wrapper.generate_3d_bundle_image_text( 
                                    prompt,
                                    image=init_image, 
                                    strength=strength,
                                    lora_scale=lora_scale,
                                    num_inference_steps=num_inference_steps,
                                    seed=int(seed) if seed is not None else None,
                                    redux_hparam=redux_hparam,
                                    save_intermediate_results=True,
                                    **kwargs)
    return result[-1]

@spaces.GPU(duration=120)
def image2mesh_preprocess_(input_image_, seed, use_mv_rgb=True):
    global preprocessed_input_image

    seed = int(seed) if seed is not None else None

    # TODO: delete this later
    # k3d_wrapper.del_llm_model()
    
    input_image_save_path, reference_save_path, caption = image2mesh_preprocess(k3d_wrapper, input_image_, seed, use_mv_rgb)

    preprocessed_input_image = Image.open(input_image_save_path)
    return reference_save_path, caption


@spaces.GPU(duration=120)
def image2mesh_main_(reference_3d_bundle_image, caption, seed, strength1=0.5, strength2=0.95, enable_redux=True, use_controlnet=True, if_video=True):
    subprocess.run(['nvidia-smi'])  
    global mesh_cache 
    seed = int(seed) if seed is not None else None


    # TODO: delete this later
    # k3d_wrapper.del_llm_model()

    input_image = preprocessed_input_image

    reference_3d_bundle_image = torch.tensor(reference_3d_bundle_image).permute(2,0,1)/255

    gen_save_path, recon_mesh_path = image2mesh_main(k3d_wrapper, input_image, reference_3d_bundle_image, caption=caption, seed=seed, strength1=strength1, strength2=strength2, enable_redux=enable_redux, use_controlnet=use_controlnet)
    mesh_cache = recon_mesh_path


    if if_video:
        video_path = recon_mesh_path.replace('.obj','.mp4').replace('.glb','.mp4')
        render_video_from_obj(recon_mesh_path, video_path)
        print(f"After bundle_image_to_mesh: {torch.cuda.memory_allocated() / 1024**3} GB")
        return gen_save_path, video_path, mesh_cache
    else:
        return gen_save_path, recon_mesh_path, mesh_cache
    # return gen_save_path, recon_mesh_path

@spaces.GPU(duration=120)
def bundle_image_to_mesh(
        gen_3d_bundle_image, 
        camera_radius=3.5,
        lrm_radius = 3.5,
        isomer_radius = 4.2,
        reconstruction_stage1_steps = 0,
        reconstruction_stage2_steps = 50,
        save_intermediate_results=False
    ):
    global mesh_cache
    print(f"Before bundle_image_to_mesh: {torch.cuda.memory_allocated() / 1024**3} GB")
    k3d_wrapper.recon_model.init_flexicubes_geometry("cuda:0", fovy=50.0)
    print(f"init_flexicubes_geometry done")
    # TODO: delete this later
    k3d_wrapper.del_llm_model()

    print(f"Before bundle_image_to_mesh after deleting llm model: {torch.cuda.memory_allocated() / 1024**3} GB")

    gen_3d_bundle_image = torch.tensor(gen_3d_bundle_image).permute(2,0,1)/255
    
    recon_mesh_path = k3d_wrapper.reconstruct_3d_bundle_image(gen_3d_bundle_image, camera_radius=camera_radius, lrm_render_radius=lrm_radius, isomer_radius=isomer_radius, save_intermediate_results=save_intermediate_results, reconstruction_stage1_steps=int(reconstruction_stage1_steps), reconstruction_stage2_steps=int(reconstruction_stage2_steps))
    mesh_cache = recon_mesh_path
    
    print(f"Mesh generated at: {mesh_cache}")
    
    # Check if file exists
    if not os.path.exists(mesh_cache):
        print(f"Warning: Generated mesh file does not exist: {mesh_cache}")
        return None, mesh_cache
        
    return recon_mesh_path, mesh_cache

# _HEADER_=f"""
# <img src="{LOGO_PATH}">
#     <h2><b>Official 🤗 Gradio Demo</b></h2>
#     <h2><b>Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation</b></h2>
#     <h2>Try our demo:Please click the buttons in sequence. Feel free to redo some steps multiple times until you get a </h2>



# [![arXiv](https://img.shields.io/badge/arXiv-Link-red)]({ARXIV_LINK})  [![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)]({GITHUB_LINK})

# """

_STAR_ = f"""
<h2>If DiMeR is helpful, please help to ⭐ the <a href={GITHUB_LINK} target='_blank'>Github Repo</a>. Sincerely Thanks!</h2>
"""

_CITE_ = r"""

<h2>📝 Citation</h2>

<h2>If you find our work useful for your research or applications, please cite using the following papers:</h2>

```bibtex
@article{jiang2025dimer,
  title={DiMeR: Disentangled Mesh Reconstruction Model},
  author={Jiang, Lutao and Lin, Jiantao and Chen, Kanghao and Ge, Wenhang and Yang, Xin and Jiang, Yifan and Lyu, Yuanhuiyi and Zheng, Xu and Chen, Yingcong},
  journal={arXiv preprint arXiv:2504.17670},
  year={2025}
}

@article{lin2025kiss3dgenrepurposingimagediffusion,
  title={Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation},
  author={Jiantao Lin, Xin Yang, Meixi Chen, Yingjie Xu, Dongyu Yan, Leyi Wu, Xinli Xu, Lie XU, Shunsi Zhang, Ying-Cong Chen},
  journal={arXiv preprint arXiv:2503.01370},
  year={2025}
}

```

📋 **License**

Apache-2.0 LICENSE. Please refer to the [LICENSE file](https://huggingface.co/spaces/TencentARC/InstantMesh/blob/main/LICENSE) for details.

📧 **Contact**

If you have any questions, feel free to open a discussion or contact us at <b>ljiang553@connect.hkust-gz.edu.cn</b>.
"""

def image_to_base64(image_path):
    """Converts an image file to a base64-encoded string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# def main():

if not UI_ONLY_MODE:
    torch.set_grad_enabled(False)

# Convert the logo image to base64
logo_base64 = image_to_base64(LOGO_PATH)
# with gr.Blocks() as demo:
with gr.Blocks(css="""


    .orange-button {
        background-color: #FF8C00 !important;
        border-color: #FF8C00 !important;
        color: black !important;
    }


    .gradio-container {
        max-width: 1000px;
        margin: auto;
        width: 100%;
    }
    #center-align-column {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    #right-align-column {
        display: flex;
        justify-content: flex-end;
        align-items: center;
    }
    h1 {text-align: center;}
    h2 {text-align: center;}
    h3 {text-align: center;}
    p {text-align: center;}
    img {text-align: right;}
    .right {
    display: block;
    margin-left: auto;
    }
    .center {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 50%;
    }

    #content-container {
        max-width: 1200px;
        margin: 0 auto;
    }



""",elem_id="col-container") as demo:
    # Header Section
    # gr.Image(value=LOGO_PATH, width=64, height=64)
    # gr.Markdown(_HEADER_)
    with gr.Row(elem_id="content-container"):

        with gr.Column(scale=7, elem_id="center-align-column"):
            gr.Markdown(f"""
            # Official 🤗 Gradio Demo
            # DiMeR: Disentangled Mesh Reconstruction Model""")
            
            gr.HTML(f"""
            <div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
                <a href="{ARXIV_LINK}" target="_blank">
                    <img src="https://img.shields.io/badge/arXiv-Link-red" alt="arXiv">
                </a>
                <a href="{GITHUB_LINK}" target="_blank">
                    <img src="https://img.shields.io/badge/GitHub-Repo-blue" alt="GitHub">
                </a>
            </div>
            
            """)

    gr.Markdown(_STAR_)

    # Tabs Section
    with gr.Tabs() as main_tabs:
        with gr.TabItem('Text-to-3D', id='tab_text_to_3d'):
            gr.Markdown("Click the button 'One-click Generation' or click the buttons one by one.")
            with gr.Row():
                with gr.Column(scale=1):
                    prompt = gr.Textbox(value="", label="Input Prompt", lines=4, placeholder="input prompt here, english or chinese")
                    
                    # Modify the Examples section to display horizontally
                    gr.Examples(
                        examples=[
                            ["A cat"],
                            ["A person wearing a virtual reality headset, sitting position, bent legs, clasped hands."],
                            ["A battle mech in a mix of red, blue, and black color, with a cannon on the head."],
                            ["骷髅头, 邪恶的"],
                        ],
                        inputs=[prompt],
                        label="Example Prompts",
                        examples_per_page=4  # Force all examples to be on a single row
                    )
                    
                    with gr.Accordion("Advanced Parameters", open=False):
                        seed1 = gr.Number(value=666, label="Seed")
                        
                    btn_one_click_generate = gr.Button("One-click Generation", elem_id="one-click-generate-btn", elem_classes=["orange-button"])
                    
                    btn_text2detailed = gr.Button("1. Refine to detailed prompt")
                    
                    gr.Markdown("---")
                    
                    detailed_prompt = gr.Textbox(value="", label="Detailed Prompt", placeholder="detailed prompt will be generated here base on your input prompt. You can also edit this prompt", lines=10, interactive=True)

                    with gr.Accordion("Advanced Parameters", open=False):
                        with gr.Row():
                            img_gen_seed = gr.Number(value=666, label="Image Generation Seed")
                            num_inference_steps = gr.Slider(minimum=1, maximum=50, value=18, step=1, label="Inference Steps")
                        with gr.Row():
                            strength = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="Strength")
                            lora_scale = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.05, label="LoRA Scale")
                            
                    btn_text2img = gr.Button("2. Generate Images")
                    

                with gr.Column(scale=1):

                    output_image1 = gr.Image(label="Generated Image", interactive=False, width=800, height=350, container=True)
                    
                    with gr.Accordion("Advanced Parameters", open=False):
                        camera_radius = gr.Slider(minimum=3.0, maximum=6.0, value=3.5, step=0.01, label="Camera Radius")
                    
                    btn_gen_mesh = gr.Button("3. Generate Mesh")

                    # Textured mesh view
                    output_mesh_textured = gr.Model3D(label="3D Mesh Viewer", interactive=False, height=300)
                    download_1 = gr.DownloadButton(label="Download Mesh", interactive=False)
                    

        with gr.TabItem('Image-to-3D (coming soon)', id='tab_image_to_3d'):
            gr.Markdown("## Coming Soon")
            
        with gr.TabItem('Sparse-view-to-3D (coming soon)', id='tab_sparse_view_to_3d'):
            gr.Markdown("## Coming Soon")


    # Button Click Events
    # Text2
    btn_text2detailed.click(fn=text_to_detailed, inputs=[prompt, seed1], outputs=detailed_prompt)
    btn_text2img.click(fn=text_to_image, inputs=[detailed_prompt, img_gen_seed, strength, lora_scale, num_inference_steps], outputs=output_image1)
    
    # Split the mesh generation and video rendering steps
    btn_gen_mesh.click(fn=bundle_image_to_mesh, inputs=[output_image1, camera_radius], outputs=[output_mesh_textured, download_1]).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_1],
    )
    
    # Define a helper function for video rendering and correctly returning the video path
    # def render_and_return_video(mesh_path):
    #     if not mesh_path or not os.path.exists(mesh_path):
    #         print(f"Warning: Mesh file doesn't exist: {mesh_path}")
    #         return None
            
    #     video_path = mesh_path.replace('.obj','.mp4').replace('.glb','.mp4')
    #     print(f"Rendering video to: {video_path}")
    #     try:
    #         render_video_from_obj(mesh_path, video_path)
    #         print(f"Video successfully rendered to: {video_path}")
    #         if os.path.exists(video_path):
    #             return video_path
    #         else:
    #             print(f"Warning: Video file was not created: {video_path}")
    #             return None
    #     except Exception as e:
    #         print(f"Error during video rendering: {e}")
    #         return None
    
    # Add separate button for video rendering
    # btn_render_video.click(fn=render_and_return_video, 
    #                       inputs=download_1, 
    #                       outputs=output_video1)

    # Add a new function for one-click generation
    def one_click_generate(input_prompt, seed):
        return input_prompt, seed
    
    # Define functions for sequential execution steps
    def sequential_step1(input_prompt, seed):
        # Step 1: Generate detailed prompt
        detailed = text_to_detailed(input_prompt, seed)
        return detailed
    
    def sequential_step2(detailed, seed):
        # Step 2: Generate image
        image = text_to_image(detailed, seed, 1.0, 1.0, 18)
        return image
    
    def sequential_step3(image):
        # Step 3: Generate 3D mesh
        geometry_mesh_path, textured_mesh_path, mesh_path = bundle_image_to_mesh(image)
        return geometry_mesh_path, textured_mesh_path, mesh_path
    
    def enable_download_button():
        return gr.Button(interactive=True)
        
    # Modify one-click generation button's click event using chained .then() calls
    btn_one_click_generate.click(
        fn=one_click_generate, 
        inputs=[prompt, seed1], 
        outputs=[prompt, img_gen_seed]
    ).then(
        fn=sequential_step1,
        inputs=[prompt, img_gen_seed],
        outputs=detailed_prompt
    ).then(
        fn=sequential_step2,
        inputs=[detailed_prompt, img_gen_seed],
        outputs=output_image1
    ).then(
        fn=sequential_step3,
        inputs=output_image1,
        outputs=[output_mesh_textured, download_1]
    ).then(
        fn=enable_download_button,
        outputs=download_1
    )

    with gr.Row():
        pass
    with gr.Row():
        gr.Markdown(_CITE_)

# Modify launch parameters to ensure background processing can continue
demo.launch()


# if __name__ == "__main__":
#     main()
