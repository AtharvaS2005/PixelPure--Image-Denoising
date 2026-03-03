# --- Text-Specific Degradations ---
def add_random_cut(image, orientation='horizontal', thickness=2):
    noisy = image.copy()
    rows, cols = image.shape[:2]
    if orientation == 'horizontal':
        y = np.random.randint(rows // 6, 5 * rows // 6)
        cv2.line(noisy, (0, y), (cols, y), (255,), thickness)
    else:
        x = np.random.randint(cols // 6, 5 * cols // 6)
        cv2.line(noisy, (x, 0), (x, rows), (255,), thickness)
    return noisy

def add_erasure_line(image, thickness=3):
    noisy = image.copy()
    rows, cols = image.shape[:2]
    y = np.random.randint(rows // 8, 7 * rows // 8)
    cv2.line(noisy, (0, y), (cols, y), (255,), thickness)
    return noisy

def add_text_jitter(image, max_shift=2):
    noisy = image.copy()
    rows, cols = image.shape[:2]
    band_height = np.random.randint(10, 30)
    y = np.random.randint(0, rows - band_height)
    shift = np.random.randint(-max_shift, max_shift + 1)
    band = noisy[y:y+band_height, :]
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    band_shifted = cv2.warpAffine(band, M, (cols, band_height), borderValue=255)
    noisy[y:y+band_height, :] = band_shifted
    return noisy

def add_local_ink_bleed(image, radius=8, intensity=0.5):
    noisy = image.copy()
    rows, cols = image.shape[:2]
    x = np.random.randint(radius, cols - radius)
    y = np.random.randint(radius, rows - radius)
    mask = np.zeros_like(noisy, dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, (255,), -1)
    blurred = cv2.GaussianBlur(noisy, (radius*2+1, radius*2+1), 0)
    noisy[mask == 255] = (noisy[mask == 255] * (1 - intensity) + blurred[mask == 255] * intensity).astype(np.uint8)
    return noisy
# --- Additional Realistic Noises ---
def add_curved_shadow(image, strength=0.5):
    rows, cols = image.shape[:2]
    mask = np.ones((rows, cols), dtype=np.float32)
    curve = np.sin(np.linspace(0, np.pi, cols))
    for i in range(rows):
        mask[i, :] *= 1 - strength * (curve * (np.random.uniform(0.7, 1.3)))
    shadowed = image.astype('float32') * mask
    return np.clip(shadowed, 0, 255).astype('uint8')

def add_partial_occlusion(image, box_size=40):
    noisy = image.copy()
    rows, cols = image.shape[:2]
    x = np.random.randint(0, cols - box_size)
    y = np.random.randint(0, rows - box_size)
    color = np.random.randint(180, 255)
    noisy[y:y+box_size, x:x+box_size] = color
    return noisy

def add_watermark(image, text='CONFIDENTIAL', opacity=0.08):
    noisy = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(image.shape[1], image.shape[0]) / 400
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = (image.shape[1] - text_size[0]) // 2
    y = (image.shape[0] + text_size[1]) // 2
    overlay = noisy.copy()
    cv2.putText(overlay, text, (x, y), font, font_scale, (200,), thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, opacity, noisy, 1 - opacity, 0, noisy)
    return noisy

def add_fold_mark(image, thickness=2):
    noisy = image.copy()
    rows, cols = image.shape[:2]
    x = np.random.randint(cols // 4, 3 * cols // 4)
    cv2.line(noisy, (x, 0), (x, rows), (220,), thickness)
    return noisy

def add_edge_fade(image, fade_width=40):
    rows, cols = image.shape[:2]
    mask = np.ones((rows, cols), dtype=np.float32)
    for i in range(fade_width):
        alpha = 1 - (i / fade_width)
        mask[:, i] *= alpha
        mask[:, cols - 1 - i] *= alpha
        mask[i, :] *= alpha
        mask[rows - 1 - i, :] *= alpha
    faded = image.astype('float32') * mask
    return np.clip(faded, 0, 255).astype('uint8')
import os
import cv2
import numpy as np
from glob import glob

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
CLEAN_DIR = os.path.join(ROOT_DIR, 'Data', 'Clean')
NOISY_DIR = os.path.join(ROOT_DIR, 'Data', 'Noisy')
os.makedirs(NOISY_DIR, exist_ok=True)

# Noise functions
def add_salt_pepper_noise(image, amount=0.02, salt_vs_pepper=0.5):
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))
    # Salt
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255
    # Pepper
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    return noisy

def add_gaussian_noise(image, mean=0, var=10):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype('float32')
    noisy = image.astype('float32') + gauss
    noisy = np.clip(noisy, 0, 255).astype('uint8')
    return noisy

def add_shading(image, intensity=0.5):
    rows, cols = image.shape[:2]
    gradient = np.tile(np.linspace(1-intensity, 1, cols), (rows, 1))
    shaded = image.astype('float32') * gradient
    shaded = np.clip(shaded, 0, 255).astype('uint8')
    return shaded

def add_blur(image, ksize=3):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def add_streaks(image, num_streaks=2, thickness=2):
    noisy = image.copy()
    rows, cols = image.shape[:2]
    for _ in range(num_streaks):
        x = np.random.randint(0, cols)
        cv2.line(noisy, (x, 0), (x, rows), (np.random.randint(180, 255)), thickness)
    return noisy

def add_speckles(image, amount=0.012):
    noisy = image.copy()
    num_speckles = int(amount * image.size)
    coords = [np.random.randint(0, i - 1, num_speckles) for i in image.shape]
    noisy[tuple(coords)] = np.random.randint(0, 256, num_speckles)
    return noisy

def add_shadow(image, intensity=0.7):
    rows, cols = image.shape[:2]
    mask = np.ones((rows, cols), dtype='float32')
    x1, x2 = np.random.randint(0, cols//2), np.random.randint(cols//2, cols)
    mask[:, x1:x2] *= intensity
    shadowed = image.astype('float32') * mask
    shadowed = np.clip(shadowed, 0, 255).astype('uint8')
    return shadowed

def add_jpeg_artifacts(image, quality=30):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return cv2.cvtColor(decimg, cv2.COLOR_BGR2GRAY) if len(image.shape) == 2 else decimg

def add_smudge(image, ksize=15):
    noisy = image.copy()
    rows, cols = image.shape[:2]
    x, y = np.random.randint(0, cols-ksize), np.random.randint(0, rows-ksize)
    roi = cv2.GaussianBlur(noisy[y:y+ksize, x:x+ksize], (ksize, ksize), 0)
    noisy[y:y+ksize, x:x+ksize] = roi
    return noisy

def add_edge_artifacts(image, thickness=10):
    noisy = image.copy()
    cv2.rectangle(noisy, (0, 0), (noisy.shape[1]-1, noisy.shape[0]-1), (np.random.randint(0, 50)), thickness)
    return noisy

def add_ghosting(image, shift=5, alpha=0.3):
    M = np.float32([[1, 0, shift], [0, 1, shift]])
    ghost = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    combined = cv2.addWeighted(image, 1, ghost, alpha, 0)
    return combined

def add_low_contrast(image, factor=0.5):
    mean = np.mean(image)
    return np.clip((image - mean) * factor + mean, 0, 255).astype('uint8')

def add_skew(image, angle=2):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), np.random.uniform(-angle, angle), 1)
    skewed = cv2.warpAffine(image, M, (cols, rows), borderValue=255)
    return skewed

def add_bleed_through(image, alpha=0.2):
    flipped = cv2.flip(image, -1)
    return cv2.addWeighted(image, 1, flipped, alpha, 0)

def add_random_marks(image, num_marks=3):
    noisy = image.copy()
    rows, cols = image.shape[:2]
    for _ in range(num_marks):
        x1, y1 = np.random.randint(0, cols), np.random.randint(0, rows)
        x2, y2 = np.random.randint(0, cols), np.random.randint(0, rows)
        cv2.line(noisy, (x1, y1), (x2, y2), (np.random.randint(0, 255)), np.random.randint(1, 3))
    return noisy

# Compose all noises
def apply_all_noises(image):
    # List of legitimate, moderate noises (general + text-specific)
    noise_fns = [
        lambda img: add_salt_pepper_noise(img, amount=0.01),
        lambda img: add_gaussian_noise(img, var=5),
        lambda img: add_shading(img, intensity=0.15),
        lambda img: add_blur(img, ksize=3),
        lambda img: add_streaks(img, num_streaks=1),
        lambda img: add_shadow(img, intensity=0.5),
        lambda img: add_jpeg_artifacts(img, quality=60),
        lambda img: add_smudge(img, ksize=7),
        lambda img: add_edge_artifacts(img, thickness=4),
        lambda img: add_low_contrast(img, factor=0.85),
        lambda img: add_skew(img, angle=0.7),
        lambda img: add_curved_shadow(img, strength=0.15),
        lambda img: add_fold_mark(img, thickness=1),
        lambda img: add_edge_fade(img, fade_width=15),
        lambda img: add_speckles(img, amount=0.005),
        # Text-specific
        lambda img: add_random_cut(img, orientation=np.random.choice(['horizontal', 'vertical']), thickness=1),
        lambda img: add_erasure_line(img, thickness=1),
        lambda img: add_text_jitter(img, max_shift=1),
        lambda img: add_local_ink_bleed(img, radius=4, intensity=0.25),
    ]
    # Randomly select 3–5 noises to apply
    num_noises = np.random.randint(3, 6)
    selected_fns = np.random.choice(noise_fns, size=num_noises, replace=False)
    for fn in selected_fns:
        image = fn(image)
    return image

# Process all images
image_paths = glob(os.path.join(CLEAN_DIR, '*'))
for img_path in image_paths:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    noisy_img = apply_all_noises(img)
    base = os.path.basename(img_path)
    out_path = os.path.join(NOISY_DIR, base)
    cv2.imwrite(out_path, noisy_img)

print(f"Processed {len(image_paths)} images. Noisy images saved to {NOISY_DIR}.")
