import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../submodules/Hunyuan3D_2')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../submodules/UniRig')))

from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import trimesh
import open3d as o3d

from PIL import Image
import numpy as np
from rembg import remove as BackgroundRemover

import psutil
import pynvml
import time
import gc
import torch
import multiprocessing as mp

def log_time(save_path, attempt, log_seconds):
    """Logs CPU and GPU memory usage every second, with percentages and timestamps."""
    pynvml.nvmlInit()
    num_gpus = pynvml.nvmlDeviceGetCount()

    log_file_path = os.path.join(save_path, f"system_monitor_attempt_{attempt}.log")

    with open(log_file_path, "w") as log_file:
        log_file.write("Timestamp,time.time,CPU_Used_GB,CPU_Total_GB,CPU_Usage_Percent,"
                       "GPU_Index,GPU_Used_MB,GPU_Total_MB,GPU_Usage_Percent,GPU_Utilization\n")

        while True:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            unix_time = time.time()

            # CPU memory
            virtual_mem = psutil.virtual_memory()
            cpu_used = virtual_mem.used / (1024 ** 3)
            cpu_total = virtual_mem.total / (1024 ** 3)
            cpu_percent = virtual_mem.percent

            # GPU memory
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                used = mem.used / (1024 ** 2)  # MB
                total = mem.total / (1024 ** 2)  # MB
                percent = (used / total) * 100 if total else 0

                # Log line
                log_line = (f"{timestamp},{unix_time:.2f},{cpu_used:.2f},{cpu_total:.2f},{cpu_percent:.1f},"
                            f"{i},{used:.0f},{total:.0f},{percent:.1f},{util.gpu}\n")
                log_file.write(log_line)
                
            log_file.flush()
            time.sleep(log_seconds)



def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("GPU memory cleared.")
    else:
        print("No GPU available.")


def generate_mesh(image, save_path, attempt):
    """Runs mesh generation pipeline in a separate process"""
    mesh_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2',
        subfolder='hunyuan3d-dit-v2-0',
        use_safetensors=False,
        variant='fp16'
    )
    mesh_pipeline.enable_model_cpu_offload(device="cuda")

    print("Mesh model loaded successfully")
    tic = time.time()

    mesh = mesh_pipeline(image=image)[0]

    # Cleanup mesh with Open3D
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    simplified = o3d_mesh.simplify_quadric_decimation(50000)
    simplified.remove_unreferenced_vertices()
    o3d_mesh = simplified.remove_degenerate_triangles()
    o3d_mesh = o3d_mesh.remove_duplicated_triangles()
    o3d_mesh = o3d_mesh.remove_duplicated_vertices()
    o3d_mesh = o3d_mesh.remove_non_manifold_edges()

    decimated_mesh = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False
    )
    out_path = os.path.join(save_path, f"mesh_attempt_{attempt}.glb")
    decimated_mesh.export(out_path)

    print("Mesh generation done in", time.time() - tic, "seconds")
    clean_memory()
    return out_path


def generate_textures(mesh_path, image, save_path, attempt):
    """Runs texture painting pipeline in a separate process"""
    paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        'tencent/Hunyuan3D-2',
        subfolder='hunyuan3d-paint-v2-0-turbo',
    )
    paint_pipeline.enable_model_cpu_offload(device="cuda")

    print("Paint model loaded successfully")
    tic = time.time()

    mesh = trimesh.load(mesh_path)
    textured_mesh = paint_pipeline(mesh=mesh, image=image)

    out_path = os.path.join(save_path, f"painted_mesh_attempt_{attempt}.glb")
    textured_mesh.export(out_path)

    print("Texture generation done in", time.time() - tic, "seconds")
    clean_memory()
    return out_path


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    IMAGE_PATH = 'submodules/Hunyuan3D_2/assets/results/Octopus/Dumbo-hires_(cropped)_wikipedia.jpg'
    SAVE_PATH = 'submodules/Hunyuan3D_2/assets/results/Octopus'
    attempt = 5  # For comparison purpose only
    log_seconds = 0.5
    
    # Start system monitoring process
    time_logging = mp.Process(target=log_time, args=(SAVE_PATH, attempt, log_seconds))
    time_logging.start()
    print("System monitoring started")
    
    time.sleep(2)  # Give some time for the logger to start and get initial baseline readings
    
    # Step 0: Image background removal
    image = Image.open(IMAGE_PATH).convert("RGBA")
    image_no_bg = BackgroundRemover(image)
    arr = np.array(image_no_bg)
    alpha = arr[:, :, 3]
    threshold = 10
    arr[alpha < threshold] = (0, 0, 0, 0)
    image_no_bg_clean = Image.fromarray(arr)

    # Step 1: Mesh process
    mesh_proc = mp.Process(target=generate_mesh, args=(image_no_bg_clean, SAVE_PATH, attempt))
    mesh_proc.start()
    mesh_proc.join()
    mesh_path = os.path.join(SAVE_PATH, f"mesh_attempt_{attempt}.glb")

    # Step 2: Texture process (running in main process as both the internal models are run in separate 
    # processes anyway so no need to spawn another process here)
    paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        'tencent/Hunyuan3D-2',
        subfolder='hunyuan3d-paint-v2-0-turbo',
    )
    paint_pipeline.enable_model_cpu_offload(device="cuda")

    print("Paint model loaded successfully")
    tic = time.time()

    mesh = trimesh.load(mesh_path)
    textured_mesh = paint_pipeline(mesh=mesh, image=image_no_bg_clean)

    out_path = os.path.join(SAVE_PATH, f"painted_mesh_attempt_{attempt}.glb")
    textured_mesh.export(out_path)

    print("Texture generation done in", time.time() - tic, "seconds")
    clean_memory()

    print("✅ Mesh + Texture pipeline complete")
