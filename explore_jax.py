import jax
import jax.numpy as jnp
import numpy as np
import time
import cv2
import argparse
import sys
import platform

# Use CPU on Windows, GPU on other platforms
if platform.system() == "win32":
    jax.config.update("jax_platform_name", "cpu")

# ==========================================
# 1. Math Helpers & SDFs
# ==========================================

def smoothstep(edge0, edge1, x):
    t = jnp.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def rotate_2d(p, angle):
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    nx = c * p[0] - s * p[1]
    ny = s * p[0] + c * p[1]
    return jnp.array([nx, ny])

def sd_triangle(p, p0, p1, p2):
    e0, e1, e2 = p1 - p0, p2 - p1, p0 - p2
    v0, v1, v2 = p - p0, p - p1, p - p2
    
    pq0 = v0 - e0 * jnp.clip(jnp.dot(v0, e0) / jnp.dot(e0, e0), 0.0, 1.0)
    pq1 = v1 - e1 * jnp.clip(jnp.dot(v1, e1) / jnp.dot(e1, e1), 0.0, 1.0)
    pq2 = v2 - e2 * jnp.clip(jnp.dot(v2, e2) / jnp.dot(e2, e2), 0.0, 1.0)
    
    s = jnp.sign(e0[0]*e2[1] - e0[1]*e2[0])
    
    d = jnp.min(jnp.array([
        jnp.dot(pq0, pq0), jnp.dot(pq1, pq1), jnp.dot(pq2, pq2)
    ]))
    
    dist = -jnp.sqrt(d) * jnp.sign(jnp.min(jnp.array([
        s * (v0[0]*e0[1] - v0[1]*e0[0]),
        s * (v1[0]*e1[1] - v1[1]*e1[0]),
        s * (v2[0]*e2[1] - v2[1]*e2[0])
    ])))
    return dist

# ==========================================
# 2. Pixel Shader
# ==========================================

def render_pixel(uv, time_val):
    bg_color = jnp.array([0.1, 0.1, 0.2]) * (1.0 - uv[1] * 0.3)
    
    # Red Triangle
    ang1 = time_val
    p0_1 = rotate_2d(jnp.array([0.0, 0.4]), ang1)
    p1_1 = rotate_2d(jnp.array([0.35, -0.3]), ang1)
    p2_1 = rotate_2d(jnp.array([-0.35, -0.3]), ang1)
    d1 = sd_triangle(uv, p0_1, p1_1, p2_1)
    col1 = jnp.array([1.0, 0.3, 0.3]) 
    
    # Green Triangle
    y_bounce = jnp.sin(time_val * 3.0) * 0.5
    t2_offset = jnp.array([-0.6, y_bounce])
    p0_2 = jnp.array([0.0, 0.2]) + t2_offset
    p1_2 = jnp.array([0.2, -0.2]) + t2_offset
    p2_2 = jnp.array([-0.2, -0.2]) + t2_offset
    d2 = sd_triangle(uv, p0_2, p1_2, p2_2)
    col2 = jnp.array([0.2, 1.0, 0.2]) 

    # Blue Triangle
    ox = 0.5 + jnp.cos(time_val * 1.2) * 0.2
    oy = jnp.sin(time_val * 1.2) * 0.2
    t3_offset = jnp.array([ox, oy])
    p0_3 = rotate_2d(jnp.array([0.0, 0.25]), -time_val * 2.0) + t3_offset
    p1_3 = rotate_2d(jnp.array([0.2, -0.2]), -time_val * 2.0) + t3_offset
    p2_3 = rotate_2d(jnp.array([-0.2, -0.2]), -time_val * 2.0) + t3_offset
    d3 = sd_triangle(uv, p0_3, p1_3, p2_3)
    col3 = jnp.array([0.2, 0.5, 1.0]) 

    # Compositing
    final_color = bg_color
    aa_blur = 0.01 # Anti-aliasing amount
    
    mask3 = 1.0 - smoothstep(0.0, aa_blur, d3)
    final_color = col3 * mask3 + final_color * (1.0 - mask3)
    
    mask2 = 1.0 - smoothstep(0.0, aa_blur, d2)
    final_color = col2 * mask2 + final_color * (1.0 - mask2)
    
    mask1 = 1.0 - smoothstep(0.0, aa_blur, d1)
    final_color = col1 * mask1 + final_color * (1.0 - mask1)
    
    return final_color

# Vectorize
render_scene_2d = jax.vmap(jax.vmap(render_pixel, in_axes=(0, None)), in_axes=(0, None))

@jax.jit
def render_frame_2d(uv_grid, time_val):
    return render_scene_2d(uv_grid, time_val)

# ==========================================
# 3. Main Application Logic
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="JAX 2D Renderer")
    parser.add_argument("--no-render", action="store_true", help="Disable window display (headless mode)")
    parser.add_argument("--save-video", type=str, default=None, help="Path to save video file (e.g. out.mp4)")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration to run in seconds (default 10s)")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS for video recording")
    parser.add_argument("--width", type=int, default=800, help="Resolution Width")
    parser.add_argument("--height", type=int, default=600, help="Resolution Height")
    
    args = parser.parse_args()

    WIDTH, HEIGHT = args.width, args.height
    
    # UV Grid Setup
    aspect = WIDTH / HEIGHT
    y = jnp.linspace(1, -1, HEIGHT)
    x = jnp.linspace(-aspect, aspect, WIDTH)
    xv, yv = jnp.meshgrid(x, y)
    uv_grid = jnp.stack([xv, yv], axis=-1)

    print(f"Initializing JAX on: {jax.devices()[0]}...")
    print("Compiling Kernel... (this may take a few seconds)")
    
    # Warmup
    start_warm = time.time()
    _ = render_frame_2d(uv_grid, 0.0).block_until_ready()
    print(f"Compilation finished in {time.time() - start_warm:.2f}s")

    # Video Writer Setup
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, args.fps, (WIDTH, HEIGHT))
        print(f"Recording to {args.save_video} at {args.fps} FPS for {args.duration}s")

    start_time = time.time()
    frame_count = 0
    
    # Simulation Loop
    try:
        while True:
            # Determine Time
            if args.save_video:
                # Fixed time step for smooth video recording
                current_time = frame_count / args.fps
                if current_time > args.duration:
                    print("\nRecording finished.")
                    break
            else:
                # Real-time clock
                current_time = time.time() - start_time
                if args.no_render and current_time > args.duration:
                    print("\nHeadless run finished.")
                    break

            # --- RENDER STEP ---
            img_jax = render_frame_2d(uv_grid, current_time)
            img_jax.block_until_ready()
            
            # Convert to CPU/Numpy
            img_np = np.array(img_jax)
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # --- OUTPUT STEPS ---

            # 1. Save to Video
            if video_writer:
                video_writer.write(img_bgr)
                # Progress indicator
                sys.stdout.write(f"\rRendering frame {frame_count}/{int(args.duration * args.fps)}")
                sys.stdout.flush()

            # 2. Display Window (if enabled)
            if not args.no_render:
                # Calculate Realtime FPS for display
                real_fps = frame_count / (time.time() - start_time + 1e-5)
                
                cv2.putText(img_bgr, f"FPS: {real_fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if args.save_video:
                    cv2.putText(img_bgr, "REC", (WIDTH - 60, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('JAX Renderer', img_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # If strictly headless and not saving video, print FPS stats periodically
            elif args.no_render and not args.save_video and frame_count % 60 == 0:
                real_fps = frame_count / (time.time() - start_time + 1e-5)
                sys.stdout.write(f"\rSimulating... FPS: {real_fps:.2f}")
                sys.stdout.flush()

            frame_count += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if video_writer:
            video_writer.release()
            print(f"\nVideo saved to {args.save_video}")
        
        if not args.no_render:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()