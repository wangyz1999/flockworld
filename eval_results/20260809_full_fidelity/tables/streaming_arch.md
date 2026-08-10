# streaming_arch

### exp01_baseline

**Quality metrics**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detection rate (up) | 0.972 | 0.286 | 0.174 |
| Position error px (down) | 1.490 | 2.583 | 3.123 |
| Reciprocity rate (up) | 0.540 | 0.225 | 0.171 |
| Displacement error px (down) | 3.062 | 62.492 | 64.899 |
| PSNR dB (up) | 27.699 | 9.420 | 8.578 |
| SSIM (up) | 0.948 | 0.386 | 0.306 |

**Volume / context (not a quality metric)**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detections / frame (volume) | 5.011 | 13.674 | 11.837 |
| GT boids visible / frame (volume) | 5.139 | 5.139 | 5.139 |
| Sightings / pair-frame (volume) | 0.074 | 0.346 | 0.279 |
| n_sightings (volume) | 5920 | 27716 | 22373 |
| n_reciprocal (volume) | 1773 | 3180 | 1930 |
| n_matched_white (volume) | 4676 | 5466 | 4584 |
| n_overlap_white (volume) | 5053 | 41682 | 30601 |
| n_pixel_events (volume) | 1773 | 3180 | 1930 |

### exp02_tiled

**Quality metrics**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detection rate (up) | 0.972 | 0.223 | 0.174 |
| Position error px (down) | 1.490 | 3.446 | 3.123 |
| Reciprocity rate (up) | 0.540 | 0.437 | 0.171 |
| Displacement error px (down) | 3.062 | 64.187 | 64.899 |
| PSNR dB (up) | 27.699 | 9.915 | 8.578 |
| SSIM (up) | 0.948 | 0.374 | 0.306 |

**Volume / context (not a quality metric)**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detections / frame (volume) | 5.011 | 22.988 | 11.837 |
| GT boids visible / frame (volume) | 5.139 | 5.139 | 5.139 |
| Sightings / pair-frame (volume) | 0.074 | 0.774 | 0.279 |
| n_sightings (volume) | 5920 | 62093 | 22373 |
| n_reciprocal (volume) | 1773 | 13596 | 1930 |
| n_matched_white (volume) | 4676 | 23098 | 4584 |
| n_overlap_white (volume) | 5053 | 154496 | 30601 |
| n_pixel_events (volume) | 1773 | 13596 | 1930 |

### exp03_diffusion_forcing

**Quality metrics**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detection rate (up) | 0.972 | 0.257 | 0.174 |
| Position error px (down) | 1.490 | 1.937 | 3.123 |
| Reciprocity rate (up) | 0.540 | 0.136 | 0.171 |
| Displacement error px (down) | 3.062 | 9.445 | 64.899 |
| PSNR dB (up) | 27.699 | 19.032 | 8.578 |
| SSIM (up) | 0.948 | 0.851 | 0.306 |

**Volume / context (not a quality metric)**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detections / frame (volume) | 5.011 | 4.357 | 11.837 |
| GT boids visible / frame (volume) | 5.139 | 5.139 | 5.139 |
| Sightings / pair-frame (volume) | 0.074 | 0.030 | 0.279 |
| n_sightings (volume) | 5920 | 2438 | 22373 |
| n_reciprocal (volume) | 1773 | 175 | 1930 |
| n_matched_white (volume) | 4676 | 328 | 4584 |
| n_overlap_white (volume) | 5053 | 690 | 30601 |
| n_pixel_events (volume) | 1773 | 175 | 1930 |

### exp04_two_stage

**Quality metrics**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detection rate (up) | 0.972 | 0.296 | 0.174 |
| Position error px (down) | 1.490 | 2.135 | 3.123 |
| Reciprocity rate (up) | 0.540 | 0.507 | 0.171 |
| Displacement error px (down) | 3.062 | 63.187 | 64.899 |
| PSNR dB (up) | 27.699 | 13.422 | 8.578 |
| SSIM (up) | 0.948 | 0.595 | 0.306 |

**Volume / context (not a quality metric)**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detections / frame (volume) | 5.011 | 9.928 | 11.837 |
| GT boids visible / frame (volume) | 5.139 | 5.139 | 5.139 |
| Sightings / pair-frame (volume) | 0.074 | 0.672 | 0.279 |
| n_sightings (volume) | 5920 | 53893 | 22373 |
| n_reciprocal (volume) | 1773 | 14014 | 1930 |
| n_matched_white (volume) | 4676 | 1471 | 4584 |
| n_overlap_white (volume) | 5053 | 41651 | 30601 |
| n_pixel_events (volume) | 1773 | 14014 | 1930 |

### exp06_tiled_df

**Quality metrics**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detection rate (up) | 0.972 | 0.157 | 0.174 |
| Position error px (down) | 1.490 | 2.509 | 3.123 |
| Reciprocity rate (up) | 0.540 | 0.135 | 0.171 |
| Displacement error px (down) | 3.062 | 37.974 | 64.899 |
| PSNR dB (up) | 27.699 | 17.496 | 8.578 |
| SSIM (up) | 0.948 | 0.838 | 0.306 |

**Volume / context (not a quality metric)**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detections / frame (volume) | 5.011 | 4.668 | 11.837 |
| GT boids visible / frame (volume) | 5.139 | 5.139 | 5.139 |
| Sightings / pair-frame (volume) | 0.074 | 0.030 | 0.279 |
| n_sightings (volume) | 5920 | 2434 | 22373 |
| n_reciprocal (volume) | 1773 | 176 | 1930 |
| n_matched_white (volume) | 4676 | 343 | 4584 |
| n_overlap_white (volume) | 5053 | 890 | 30601 |
| n_pixel_events (volume) | 1773 | 176 | 1930 |

### exp07_tiled_df_two_stage

**Quality metrics**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detection rate (up) | 0.972 | 0.256 | 0.174 |
| Position error px (down) | 1.490 | 1.855 | 3.123 |
| Reciprocity rate (up) | 0.540 | 0.119 | 0.171 |
| Displacement error px (down) | 3.062 | 51.156 | 64.899 |
| PSNR dB (up) | 27.699 | 16.058 | 8.578 |
| SSIM (up) | 0.948 | 0.805 | 0.306 |

**Volume / context (not a quality metric)**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detections / frame (volume) | 5.011 | 4.453 | 11.837 |
| GT boids visible / frame (volume) | 5.139 | 5.139 | 5.139 |
| Sightings / pair-frame (volume) | 0.074 | 0.185 | 0.279 |
| n_sightings (volume) | 5920 | 14810 | 22373 |
| n_reciprocal (volume) | 1773 | 938 | 1930 |
| n_matched_white (volume) | 4676 | 360 | 4584 |
| n_overlap_white (volume) | 5053 | 2937 | 30601 |
| n_pixel_events (volume) | 1773 | 938 | 1930 |

### exp08_tiled_two_stage

**Quality metrics**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detection rate (up) | 0.972 | 0.111 | 0.174 |
| Position error px (down) | 1.490 | 2.883 | 3.123 |
| Reciprocity rate (up) | 0.540 | 0.160 | 0.171 |
| Displacement error px (down) | 3.062 | 61.100 | 64.899 |
| PSNR dB (up) | 27.699 | 15.387 | 8.578 |
| SSIM (up) | 0.948 | 0.768 | 0.306 |

**Volume / context (not a quality metric)**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detections / frame (volume) | 5.011 | 5.530 | 11.837 |
| GT boids visible / frame (volume) | 5.139 | 5.139 | 5.139 |
| Sightings / pair-frame (volume) | 0.074 | 0.281 | 0.279 |
| n_sightings (volume) | 5920 | 22571 | 22373 |
| n_reciprocal (volume) | 1773 | 2080 | 1930 |
| n_matched_white (volume) | 4676 | 701 | 4584 |
| n_overlap_white (volume) | 5053 | 9704 | 30601 |
| n_pixel_events (volume) | 1773 | 2080 | 1930 |

### single_stream_floor_new

**Quality metrics**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detection rate (up) | 0.972 | 0.200 | 0.174 |
| Position error px (down) | 1.490 | 2.686 | 3.123 |
| Reciprocity rate (up) | 0.540 | 0.225 | 0.171 |
| Displacement error px (down) | 3.062 | 68.807 | 64.899 |
| PSNR dB (up) | 27.699 | 11.908 | 8.578 |
| SSIM (up) | 0.948 | 0.563 | 0.306 |

**Volume / context (not a quality metric)**

| metric | ceiling | model | baseline |
|---|---|---|---|
| Detections / frame (volume) | 5.011 | 9.385 | 11.837 |
| GT boids visible / frame (volume) | 5.139 | 5.139 | 5.139 |
| Sightings / pair-frame (volume) | 0.074 | 0.412 | 0.279 |
| n_sightings (volume) | 5920 | 33062 | 22373 |
| n_reciprocal (volume) | 1773 | 3771 | 1930 |
| n_matched_white (volume) | 4676 | 2987 | 4584 |
| n_overlap_white (volume) | 5053 | 37984 | 30601 |
| n_pixel_events (volume) | 1773 | 3771 | 1930 |
