import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.metrics import structural_similarity as ssim
from autoencoder_attack import AttackAutoencoder, train_attacker

def psnr(original, watermarked):
    mse_value = np.mean((original - watermarked) ** 2)
    if mse_value == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse_value))

def calculate_metrics(original, compared):
    if original.shape != compared.shape:
        compared = cv2.resize(compared, (original.shape[1], original.shape[0]))
    
    psnr_val = psnr(original, compared)
    ssim_val = ssim(original, compared, data_range=255)
    
    return psnr_val, ssim_val

def calculate_ncc(original_wm, extracted_wm):
    original_wm = original_wm.astype(np.float32)
    extracted_wm = extracted_wm.astype(np.float32)
    
    # Ortalama değerleri çıkar
    original_wm -= np.mean(original_wm)
    extracted_wm -= np.mean(extracted_wm)
    
    numerator = np.sum(original_wm * extracted_wm)
    denominator = np.sqrt(np.sum(original_wm**2)) * np.sqrt(np.sum(extracted_wm**2))
    
    if denominator == 0:
        return 0
    return numerator / denominator

def add_watermark(cover_path, watermark_path, alpha=0.1):
    # Taşıyıcı veriye watermark ekleme
    cover = cv2.imread(cover_path, 0)
    watermark = cv2.imread(watermark_path, 0)

    cover = cv2.resize(cover, (512, 512))
    h, w = cover.shape

    watermark = cv2.resize(watermark, (h//2, w//2))

    # Taşıyıcı imgenin 2 seviye DWT'si
    LL2, (HL2, LH2, HH2), (HL1, LH1, HH1) = pywt.wavedec2(cover, 'haar', level=2)

    # Watermark verinin 1 seviye DWT'si
    LL_w, (HL_w, LH_w, HH_w) = pywt.wavedec2(watermark, 'haar', level=1)

    # Taşıyıcının HL2 bandına SVD uygulama
    Uc, Sc, Vc = np.linalg.svd(HL2, full_matrices=False)

    # Watermark'ın HL_w bandına SVD uygulama
    Uw, Sw, Vw = np.linalg.svd(HL_w, full_matrices=False)

    len_S = min(len(Sc), len(Sw))
    S_new = Sc.copy()
    S_new[:len_S] = Sc[:len_S] + alpha * Sw[:len_S]

    # Inverse SVD
    HL2_watermarked = np.dot(Uc, np.dot(np.diag(S_new), Vc))

    # Inverse DWT
    watermarked_coeffs = [LL2, (HL2_watermarked, LH2, HH2), (HL1, LH1, HH1)]
    watermarked_image = pywt.waverec2(watermarked_coeffs, 'haar')
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

    return watermarked_image, Sc, Sw, Uc, Vc, Uw, Vw, LL_w, LH_w, HH_w


def extract_watermark(watermarked_image, Sc, Uw, Vw, LL_w, LH_w, HH_w, alpha=0.1):
    LL2, (HL2_new, LH2, HH2), (HL1, LH1, HH1) = pywt.wavedec2(watermarked_image, 'haar', level=2)

    # Watermarked HL2 bandına SVD uygulama
    Uc_new, Sc_new, Vc_new = np.linalg.svd(HL2_new, full_matrices=False)
    len_S = min(len(Sc), len(Sc_new))
    Sw_extracted = (Sc_new[:len_S] - Sc[:len_S]) / alpha

    # Inverse SVD ve Inverse DWT
    HL_w_extracted = np.dot(Uw, np.dot(np.diag(Sw_extracted), Vw))
    recovered_coeffs = (LL_w, 
                        (HL_w_extracted,
                         LH_w,
                         HH_w))
    recovered_watermark = pywt.idwt2(recovered_coeffs, 'haar')
    recovered_watermark = np.clip(recovered_watermark, 0, 255).astype(np.uint8)

    return recovered_watermark

def apply_attack(image, attack_type, param):
    h, w = image.shape
    attacked_img = image.copy()

    if attack_type == "JPEG":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), param]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        attacked_img = cv2.imdecode(encimg, 0)
    
    elif attack_type == "Salt & Pepper":
        num_noise = int(param * image.size)
        coords = [np.random.randint(0, i - 1, num_noise) for i in image.shape]
        attacked_img[tuple(coords)] = 255
        coords = [np.random.randint(0, i - 1, num_noise) for i in image.shape]
        attacked_img[tuple(coords)] = 0

    elif attack_type == "Gaussian Blur":
        k = param if param % 2 == 1 else param + 1
        attacked_img = cv2.GaussianBlur(image, (k, k), 0)

    elif attack_type == "Cropout":
        border_h = int(h * (param / 100) / 2)
        border_w = int(w * (param / 100) / 2)
        attacked_img[:border_h, :] = 0
        attacked_img[-border_h:, :] = 0
        attacked_img[:, :border_w] = 0
        attacked_img[:, -border_w:] = 0

    elif attack_type == "Rotation":
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, param, 1.0)
        attacked_img = cv2.warpAffine(image, M, (w, h))
    
    elif attack_type == "AI":
        attacker = AttackAutoencoder()
        cv2.imwrite('temp_watermarked.png', image)
        attacked_img = train_attacker(attacker, 'temp_watermarked.png', 'cover.png', epochs=500)

    return attacked_img


if __name__ == "__main__":
    cover_image_path = 'cover.png'
    watermark_image_path = 'watermark.jpg'
    alpha = 0.2

    watermarked_image, Sc, Sw, Uc, Vc, Uw, Vw, LL_w, LH_w, HH_w = add_watermark(cover_image_path, watermark_image_path, alpha=alpha)
    cv2.imwrite('watermarked_image.png', watermarked_image)
    recovered_watermark = extract_watermark(watermarked_image, Sc, Uw, Vw, LL_w, LH_w, HH_w, alpha=alpha)
    cv2.imwrite('recovered_watermark.png', recovered_watermark)

    cover = cv2.imread(cover_image_path, 0)
    cover = cv2.resize(cover, (512, 512))
    
    watermark_original = cv2.imread(watermark_image_path, 0)
    watermark_original = cv2.resize(watermark_original, (256, 256))

    # Cover image ve watermarked image arasındaki metrikler
    print("Cover Image vs Watermarked Image Metrics:")
    psnr_cover, ssim_cover = calculate_metrics(cover, watermarked_image)
    print(f'PSNR: {psnr_cover:.2f} dB')
    print(f'SSIM: {ssim_cover:.4f}')
    
    # Orijinal watermark ve recovered watermark arasındaki metrikler
    print("\nOriginal Watermark vs Recovered Watermark Metrics (No Attack):")
    psnr_wm, ssim_wm = calculate_metrics(watermark_original, recovered_watermark)
    ncc_wm = calculate_ncc(watermark_original, recovered_watermark)
    print(f'PSNR: {psnr_wm:.2f} dB')
    print(f'SSIM: {ssim_wm:.4f}')
    print(f'NCC: {ncc_wm:.4f}')

    # Atakları tanımla
    attacks = [
        ("No Attack", None, None),
        ("JPEG Q=40", "JPEG", 40),
        ("Salt & Pepper 0.05", "Salt & Pepper", 0.05),
        ("Gaussian Blur 7", "Gaussian Blur", 7),
        ("Cropout 50%", "Cropout", 50),
        ("Rotation 30°", "Rotation", 30),
        ("AI Autoencoder", "AI", None)
    ]

    # Atakları uygula ve sonuçları sakla
    attacked_images = []
    recovered_watermarks = []
    
    print("\nATTACK ROBUSTNESS ANALYSIS")
    
    for attack_name, attack_type, param in attacks:
        if attack_type is None:
            # Atak yok - orijinal watermarked image
            attacked_img = watermarked_image.copy()
            recovered_wm = recovered_watermark.copy()
        else:
            # Atağı uygula
            attacked_img = apply_attack(watermarked_image, attack_type, param)
            # Watermark'ı çıkar
            recovered_wm = extract_watermark(attacked_img, Sc, Uw, Vw, LL_w, LH_w, HH_w, alpha=alpha)
        
        attacked_images.append(attacked_img)
        recovered_watermarks.append(recovered_wm)
        
        # Metrikleri hesapla
        psnr_val, ssim_val = calculate_metrics(watermark_original, recovered_wm)
        ncc_val = calculate_ncc(watermark_original, recovered_wm)
        
        print(f"\n{attack_name}:")
        print(f"PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | NCC: {ncc_val:.4f}")
    
    # Görselleştirme
    fig, axes = plt.subplots(3, len(attacks), figsize=(18, 9))
    
    for i, (attack_name, _, _) in enumerate(attacks):
        # İlk satır: Atak isimleri ve attacked images
        axes[0, i].imshow(attacked_images[i], cmap='gray')
        axes[0, i].set_title(attack_name, fontsize=10, fontweight='bold')
        axes[0, i].axis('off')
        
        # İkinci satır: Recovered watermarks
        axes[1, i].imshow(recovered_watermarks[i], cmap='gray')
        axes[1, i].set_title('Recovered WM', fontsize=9)
        axes[1, i].axis('off')
        
        # Üçüncü satır: Original watermark (referans için)
        axes[2, i].imshow(watermark_original, cmap='gray')
        axes[2, i].set_title('Original WM', fontsize=9)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('attack_analysis.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'attack_analysis.png'")
    plt.show()