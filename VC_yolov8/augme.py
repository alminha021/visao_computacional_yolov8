import os
import cv2
import albumentations as A
from shutil import copyfile

# === CONFIGURAÇÕES ===
classes_minoritarias_ids = [2, 5, 6, 9, 10]  # IDs das classes que você quer aumentar
dataset_dir = "train/images"          # Pasta de imagens originais
labels_dir = "train/labels"           # Pasta de labels .txt
output_images_dir = "dataset_augmented/train/images"
output_labels_dir = "dataset_augmented/train/labels"
num_augments_per_image = 5            # Número de imagens aumentadas por imagem

# Cria as pastas de saída
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Transformações de aumento
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
])

# Função para checar se a imagem possui alguma das classes minoritárias
def tem_uma_das_classes(label_path, class_ids):
    with open(label_path, 'r') as f:
        for line in f:
            cls = int(line.split()[0])
            if cls in class_ids:
                return True
    return False

# Loop pelas imagens
for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(labels_dir, label_file)
    image_name = label_file.replace(".txt", ".jpg")  # ajuste se não forem .jpg
    image_path = os.path.join(dataset_dir, image_name)

    # Copia imagem e label originais
    copyfile(image_path, os.path.join(output_images_dir, image_name))
    copyfile(label_path, os.path.join(output_labels_dir, label_file))

    # Se tiver uma das classes minoritárias, gera aumentações
    if tem_uma_das_classes(label_path, classes_minoritarias_ids):
        img = cv2.imread(image_path)

        for i in range(num_augments_per_image):
            augmented = transform(image=img)
            aug_img = augmented['image']

            new_image_name = image_name.replace(".jpg", f"_aug{i}.jpg")
            new_label_name = label_file.replace(".txt", f"_aug{i}.txt")

            cv2.imwrite(os.path.join(output_images_dir, new_image_name), aug_img)
            copyfile(label_path, os.path.join(output_labels_dir, new_label_name))
