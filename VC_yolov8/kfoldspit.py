import os
import random
import shutil
import yaml

base_dir = "dataset_augmented/train"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

output_dir = "final_cross"
os.makedirs(output_dir, exist_ok=True)

# Lista de todos os arquivos de imagem .jpg
all_images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
all_images.sort()  # Para garantir ordem fixa

random.seed(42)  # fixar seed para reprodutibilidade
random.shuffle(all_images)

num_folds = 3
fold_size = len(all_images) // num_folds

for fold in range(num_folds):
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold != num_folds - 1 else len(all_images)

    val_images = all_images[val_start:val_end]
    train_images = all_images[:val_start] + all_images[val_end:]

    fold_dir = os.path.join(output_dir, f"fold_{fold+1}")
    train_img_dir = os.path.join(fold_dir, "train", "images")
    train_lbl_dir = os.path.join(fold_dir, "train", "labels")
    val_img_dir = os.path.join(fold_dir, "val", "images")
    val_lbl_dir = os.path.join(fold_dir, "val", "labels")

    # Criar pastas
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Copiar imagens e labels treino
    for img_name in train_images:
        shutil.copy(os.path.join(images_dir, img_name), os.path.join(train_img_dir, img_name))
        label_name = img_name.replace(".jpg", ".txt")
        shutil.copy(os.path.join(labels_dir, label_name), os.path.join(train_lbl_dir, label_name))

    # Copiar imagens e labels validação
    for img_name in val_images:
        shutil.copy(os.path.join(images_dir, img_name), os.path.join(val_img_dir, img_name))
        label_name = img_name.replace(".jpg", ".txt")
        shutil.copy(os.path.join(labels_dir, label_name), os.path.join(val_lbl_dir, label_name))

    # Criar arquivo YAML do fold
    yaml_content = {
        'train': os.path.abspath(os.path.join(fold_dir, "train", "images")),
        'val': os.path.abspath(os.path.join(fold_dir, "val", "images")),
        'nc': 12,  # Número de classes, ajuste se precisar
        'names': ['big bus', 'big truck', 'bus-l-', 'bus-s-', 'car', 'mid truck',
                  'small bus', 'small truck', 'truck-l-', 'truck-m-', 'truck-s-', 'truck-xl-']
    }
    with open(os.path.join(fold_dir, "data.yaml"), 'w') as f:
        yaml.dump(yaml_content, f)

print("Divisão em 3 folds concluída! Pasta 'crossval' criada com os dados e YAMLs.")
