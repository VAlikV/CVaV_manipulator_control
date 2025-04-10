import torch
from ggcnn.utils.dataset_processing import image
from ggcnn.models.ggcnn import GGCNN

# === 1. Загрузка depth изображения ===
depth_img = image.DepthImage.from_file('path/to/depth.png')

# === 2. Препроцессинг ===
depth_img_resized = depth_img.resize((300, 300))
depth_tensor = torch.from_numpy(depth_img_resized.img).unsqueeze(0).unsqueeze(0).float()

# === 3. Загрузка модели ===
model = GGCNN()
model.load_state_dict(torch.load('trained_models/ggcnn.pth'))
model.eval()

# === 4. Предсказание ===
with torch.no_grad():
    pred = model(depth_tensor)

# === 5. Выходы ===
quality_map = pred[0][0].numpy()
angle_map = pred[1][0].numpy()
width_map = pred[2][0].numpy()

# Теперь можно найти максимум в карте quality и построить точку захвата