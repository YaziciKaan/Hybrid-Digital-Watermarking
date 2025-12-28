# Hybrid Digital Watermarking

This repository implements a robust hybrid digital watermarking technique based on the research paper:

**"A robust hybrid digital watermarking technique against a powerful CNN-based adversarial attack"** (2020)

## Overview

This implementation combines **Discrete Wavelet Transform (DWT)** and **Singular Value Decomposition (SVD)** to embed and extract watermarks from digital images. The technique is specifically designed to be robust against both traditional image processing attacks and advanced CNN-based adversarial attacks.

## Method

The watermarking scheme operates as follows:

### Embedding Process
1. Apply 2-level DWT to the cover image (512×512)
2. Apply 1-level DWT to the watermark image (256×256)
3. Perform SVD on the HL2 band of the cover image
4. Perform SVD on the HL_w band of the watermark
5. Modify singular values: `S_new = S_cover + α × S_watermark`
6. Reconstruct the watermarked image using inverse SVD and inverse DWT

### Extraction Process
1. Apply 2-level DWT to the watermarked image
2. Perform SVD on the HL2 band
3. Extract watermark singular values: `S_watermark = (S_new - S_cover) / α`
4. Reconstruct the watermark using inverse SVD and inverse DWT

## Features

- **Hybrid DWT-SVD Approach**: Combines frequency domain and algebraic techniques
- **Robustness Testing**: Evaluates performance against multiple attacks:
  - JPEG Compression
  - Salt & Pepper Noise
  - Gaussian Blur
  - Cropout
  - Rotation
  - **AI Autoencoder Attack** (CNN-based)
- **Quality Metrics**: PSNR, SSIM, and NCC for comprehensive evaluation

## Files

- `watermark.py` - Main watermarking implementation and robustness analysis
- `autoencoder_attack.py` - CNN-based adversarial attack using autoencoder

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- OpenCV
- NumPy
- PyWavelets
- scikit-image
- PyTorch (for autoencoder attack)
- Matplotlib

## Usage

### Basic Watermarking

```bash
python3 watermark.py
```

This will:
1. Embed a watermark into the cover image
2. Test robustness against various attacks
3. Display PSNR, SSIM, and NCC metrics
4. Save visualization as `attack_analysis.png`

### AI Adversarial Attack
Trains a CNN autoencoder to remove the watermark and evaluates its effectiveness.

## Results

The implementation demonstrates:
- **High imperceptibility**: PSNR > 40 dB between cover and watermarked images
- **Strong robustness**: Watermark survives traditional attacks with NCC > 0.7
- **AI attack resilience**: Evaluation against CNN-based watermark removal

## Parameters

- `alpha` (α): Embedding strength (default: 0.2)
  - Higher values: More robust but less imperceptible
  - Lower values: More imperceptible but less robust

## Reference

This implementation is based on the methodology described in:

> **A robust hybrid digital watermarking technique against a powerful CNN-based adversarial attack** (2020)

The paper proposes a DWT-SVD hybrid approach specifically designed to withstand modern deep learning-based attacks while maintaining robustness against traditional image processing operations.