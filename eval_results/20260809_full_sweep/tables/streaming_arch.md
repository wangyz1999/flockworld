# streaming_arch

### exp01_baseline

**Quality metrics**

| metric | ceiling | model |
|---|---|---|
| Detection rate (up) | 0.975 | 0.282 |
| Position error px (down) | 1.482 | 2.009 |
| Reciprocity rate (up) | 0.477 | 0.107 |
| Displacement error px (down) | 3.687 | 54.035 |
| PSNR dB (up) | 26.765 | 14.582 |
| SSIM (up) | 0.943 | 0.718 |

**Volume / context (not a quality metric)**

| metric | ceiling | model |
|---|---|---|
| Detections / frame (volume) | 5.324 | 5.276 |
| GT boids visible / frame (volume) | 5.445 | 5.445 |
| Sightings / pair-frame (volume) | 0.047 | 0.156 |
| n_sightings (volume) | 1118 | 3726 |
| n_reciprocal (volume) | 272 | 201 |
| n_matched_white (volume) | 942 | 176 |
| n_overlap_white (volume) | 1022 | 859 |
| n_pixel_events (volume) | 272 | 201 |

### exp02_tiled

**Quality metrics**

| metric | ceiling | model |
|---|---|---|
| Detection rate (up) | 0.975 | 0.199 |
| Position error px (down) | 1.482 | 2.972 |
| Reciprocity rate (up) | 0.477 | 0.355 |
| Displacement error px (down) | 3.687 | 64.704 |
| PSNR dB (up) | 26.765 | 10.614 |
| SSIM (up) | 0.943 | 0.462 |

**Volume / context (not a quality metric)**

| metric | ceiling | model |
|---|---|---|
| Detections / frame (volume) | 5.324 | 11.963 |
| GT boids visible / frame (volume) | 5.445 | 5.445 |
| Sightings / pair-frame (volume) | 0.047 | 0.496 |
| n_sightings (volume) | 1118 | 11850 |
| n_reciprocal (volume) | 272 | 2132 |
| n_matched_white (volume) | 942 | 1110 |
| n_overlap_white (volume) | 1022 | 13883 |
| n_pixel_events (volume) | 272 | 2132 |

### exp03_diffusion_forcing

**Quality metrics**

| metric | ceiling | model |
|---|---|---|
| Detection rate (up) | 0.975 | 0.278 |
| Position error px (down) | 1.482 | 1.948 |
| Reciprocity rate (up) | 0.477 | 0.118 |
| Displacement error px (down) | 3.687 | 12.910 |
| PSNR dB (up) | 26.765 | 19.514 |
| SSIM (up) | 0.943 | 0.862 |

**Volume / context (not a quality metric)**

| metric | ceiling | model |
|---|---|---|
| Detections / frame (volume) | 5.324 | 3.886 |
| GT boids visible / frame (volume) | 5.445 | 5.445 |
| Sightings / pair-frame (volume) | 0.047 | 0.040 |
| n_sightings (volume) | 1118 | 954 |
| n_reciprocal (volume) | 272 | 56 |
| n_matched_white (volume) | 942 | 130 |
| n_overlap_white (volume) | 1022 | 191 |
| n_pixel_events (volume) | 272 | 56 |

### exp04_two_stage

**Quality metrics**

| metric | ceiling | model |
|---|---|---|
| Detection rate (up) | 0.975 | 0.285 |
| Position error px (down) | 1.482 | 1.871 |
| Reciprocity rate (up) | 0.477 | 0.207 |
| Displacement error px (down) | 3.687 | 46.356 |
| PSNR dB (up) | 26.765 | 15.900 |
| SSIM (up) | 0.943 | 0.754 |

**Volume / context (not a quality metric)**

| metric | ceiling | model |
|---|---|---|
| Detections / frame (volume) | 5.324 | 4.248 |
| GT boids visible / frame (volume) | 5.445 | 5.445 |
| Sightings / pair-frame (volume) | 0.047 | 0.165 |
| n_sightings (volume) | 1118 | 3949 |
| n_reciprocal (volume) | 272 | 416 |
| n_matched_white (volume) | 942 | 212 |
| n_overlap_white (volume) | 1022 | 845 |
| n_pixel_events (volume) | 272 | 416 |

### exp06_tiled_df

**Quality metrics**

| metric | ceiling | model |
|---|---|---|
| Detection rate (up) | 0.975 | 0.166 |
| Position error px (down) | 1.482 | 2.586 |
| Reciprocity rate (up) | 0.477 | 0.170 |
| Displacement error px (down) | 3.687 | 8.994 |
| PSNR dB (up) | 26.765 | 19.778 |
| SSIM (up) | 0.943 | 0.898 |

**Volume / context (not a quality metric)**

| metric | ceiling | model |
|---|---|---|
| Detections / frame (volume) | 5.324 | 3.759 |
| GT boids visible / frame (volume) | 5.445 | 5.445 |
| Sightings / pair-frame (volume) | 0.047 | 0.029 |
| n_sightings (volume) | 1118 | 701 |
| n_reciprocal (volume) | 272 | 51 |
| n_matched_white (volume) | 942 | 142 |
| n_overlap_white (volume) | 1022 | 174 |
| n_pixel_events (volume) | 272 | 51 |

### exp07_tiled_df_two_stage

**Quality metrics**

| metric | ceiling | model |
|---|---|---|
| Detection rate (up) | 0.975 | 0.279 |
| Position error px (down) | 1.482 | 1.899 |
| Reciprocity rate (up) | 0.477 | 0.070 |
| Displacement error px (down) | 3.687 | 33.358 |
| PSNR dB (up) | 26.765 | 17.167 |
| SSIM (up) | 0.943 | 0.824 |

**Volume / context (not a quality metric)**

| metric | ceiling | model |
|---|---|---|
| Detections / frame (volume) | 5.324 | 4.204 |
| GT boids visible / frame (volume) | 5.445 | 5.445 |
| Sightings / pair-frame (volume) | 0.047 | 0.120 |
| n_sightings (volume) | 1118 | 2873 |
| n_reciprocal (volume) | 272 | 95 |
| n_matched_white (volume) | 942 | 119 |
| n_overlap_white (volume) | 1022 | 235 |
| n_pixel_events (volume) | 272 | 95 |

### exp08_tiled_two_stage

**Quality metrics**

| metric | ceiling | model |
|---|---|---|
| Detection rate (up) | 0.975 | 0.147 |
| Position error px (down) | 1.482 | 2.663 |
| Reciprocity rate (up) | 0.477 | 0.126 |
| Displacement error px (down) | 3.687 | 58.298 |
| PSNR dB (up) | 26.765 | 16.014 |
| SSIM (up) | 0.943 | 0.806 |

**Volume / context (not a quality metric)**

| metric | ceiling | model |
|---|---|---|
| Detections / frame (volume) | 5.324 | 4.850 |
| GT boids visible / frame (volume) | 5.445 | 5.445 |
| Sightings / pair-frame (volume) | 0.047 | 0.204 |
| n_sightings (volume) | 1118 | 4868 |
| n_reciprocal (volume) | 272 | 319 |
| n_matched_white (volume) | 942 | 164 |
| n_overlap_white (volume) | 1022 | 953 |
| n_pixel_events (volume) | 272 | 319 |
