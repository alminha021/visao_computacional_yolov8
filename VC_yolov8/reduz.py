import os
import shutil
import random
from collections import defaultdict

base_dir = "dataset_augmented/train"
label_dir = os.path.join(base_dir, "labels")
image_dir = os.path.join(base_dir, "images")

balanced_dir = "train_balanced_reduz"
os.makedirs(os.path.join(balanced_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(balanced_dir, "labels"), exist_ok=True)

# Passo 1: Mapear classes para arquivos
class_files = defaultdict(set)
file_classes = {}

for fname in os.listdir(label_dir):
    if not fname.endswith(".txt"):
        continue

    path = os.path.join(label_dir, fname)
    with open(path, "r") as f:
        lines = f.readlines()

    classes_in_file = set(line.strip().split()[0] for line in lines)
    file_classes[fname] = classes_in_file

    for cls in classes_in_file:
        class_files[cls].add(fname)

# Passo 2: Ordenar classes por frequência ascendente (minoritárias primeiro)
sorted_classes = sorted(class_files.items(), key=lambda x: len(x[1]))

print("Classes ordenadas por frequência (menor para maior):")
for cls, files in sorted_classes:
    print(f"Classe {cls}: {len(files)} arquivos")

# Percentuais decrescentes para filtrar
percentages = [1.0, 0.8, 0.6, 0.4, 0.2]  # Pode ajustar ou parametrizar

# Passo 3: Começar com o conjunto completo de arquivos
selected_files = set(file_classes.keys())

for i, (cls, files) in enumerate(sorted_classes):
    if i >= len(percentages):
        pct = percentages[-1]  # Usa último percentual se tiver mais classes
    else:
        pct = percentages[i]

    # Arquivos do conjunto atual que contém essa classe
    files_with_class = selected_files.intersection(files)
    n_keep = int(len(files_with_class) * pct)

    print(f"\nClasse {cls} - mantendo {pct*100:.0f}% dos {len(files_with_class)} arquivos = {n_keep}")

    # Randomicamente seleciona arquivos para manter
    keep_files = set(random.sample(files_with_class, n_keep))

    # Remove os arquivos que NÃO serão mantidos (dentro do conjunto selecionado)
    remove_files = files_with_class - keep_files
    selected_files -= remove_files

# Passo 4: Copiar os arquivos filtrados para o diretório balanceado
print(f"\nTotal de arquivos selecionados após filtro: {len(selected_files)}")

for fname in selected_files:
    label_path = os.path.join(label_dir, fname)
    image_name = fname.replace(".txt", ".jpg")
    image_path = os.path.join(image_dir, image_name)

    if not os.path.exists(image_path):
        continue

    shutil.copyfile(label_path, os.path.join(balanced_dir, "labels", fname))
    shutil.copyfile(image_path, os.path.join(balanced_dir, "images", image_name))

print("Cópia dos arquivos filtrados concluída.")
