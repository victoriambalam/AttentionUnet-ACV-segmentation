import os
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt

# Funciones para leer archivos del Dataset
def guardar_slices(volumen, carpeta_salida, id_paciente):
    os.makedirs(carpeta_salida, exist_ok=True)  # Crea la carpeta si no existe

    for i in range(volumen.shape[2]):  # Iterar sobre slices
        nombre_salida = os.path.join(carpeta_salida, f"{id_paciente}_{i:03d}.png")
        slice_img = volumen[:, :, i]
        slice_img = np.flipud(np.rot90(slice_img, k=1))        
        plt.imsave(nombre_salida, slice_img, cmap="gray")

def procesar_paciente(ruta_paciente):
    id_paciente = os.path.basename(ruta_paciente)
    # Definir rutas de los archivos MRI y máscara
    if not (os.path.exists(archivo_dim_mri) and os.path.exists(archivo_ima_mri) and 
            os.path.exists(archivo_dim_mask) and os.path.exists(archivo_ima_mask)):
        print(f" Archivos faltantes en {ruta_paciente}, omitiendo...")
        return
    volumen_mask = (volumen_mask > 0).astype(np.uint8)

    # Guardar slices
    guardar_slices(volumen_mri, carpeta_input, id_paciente)
    guardar_slices(volumen_mask, carpeta_mask, id_paciente)

ruta_base = 'dataset_path'

# Iterar sobre los pacientes en la base de datos
for paciente in os.listdir(ruta_base):
    ruta_paciente = os.path.join(ruta_base, paciente)
    if os.path.isdir(ruta_paciente):
        procesar_paciente(ruta_paciente)

# AUMENTO DE DATOS
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Transformaciones a aplicar
def augment_and_save(img, mask, base_name, aug_id):
    if aug_id == 1:  # mirror horizontal
        img = TF.hflip(img)
        mask = TF.hflip(mask)
    elif aug_id == 2:  # Rotación +15°
        img = TF.rotate(img, 15, fill=0)
        mask = TF.rotate(mask, 15, fill=0)
    elif aug_id == 3:  # Rotación -15°
        img = TF.rotate(img, -15, fill=0)
        mask = TF.rotate(mask, -15, fill=0)
    elif aug_id == 4:  # Translación (-10, -10)
        img = TF.affine(img, angle=0, translate=(-10, -10), scale=1.0, shear=0, fill=0)
        mask = TF.affine(mask, angle=0, translate=(-10, -10), scale=1.0, shear=0, fill=0)
    elif aug_id == 5:  # Translación (+10, +10)
        img = TF.affine(img, angle=0, translate=(10, 10), scale=1.0, shear=0, fill=0)
        mask = TF.affine(mask, angle=0, translate=(10, 10), scale=1.0, shear=0, fill=0)
    elif aug_id == 6:  # Escalado 0.9 (centrado)
        img = TF.affine(img, angle=0, translate=(0, 0), scale=0.9, shear=0, fill=0)
        mask = TF.affine(mask, angle=0, translate=(0, 0), scale=0.9, shear=0, fill=0)
    elif aug_id == 7:  # Ruido gaussiano
        tensor_img = TF.to_tensor(img)  # [0,1]
        noise = torch.randn_like(tensor_img) * 0.03  # ruido leve
        noisy = torch.clamp(tensor_img + noise, 0., 1.)
        img = TF.to_pil_image(noisy)  # sin escalar a 255
        # máscara no cambia

    img.save(os.path.join(output_img_dir, f"{base_name}_aug{aug_id}.png"))
    mask.save(os.path.join(output_mask_dir, f"{base_name}_aug{aug_id}.png"))

# Proceso de aumento
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(".png"):
        base_name = os.path.splitext(filename)[0]
        img_path = os.path.join(input_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # Guardar imagen original
        img.save(os.path.join(output_img_dir, f"{base_name}.png"))
        mask.save(os.path.join(output_mask_dir, f"{base_name}.png"))

        # Guardar versiones aumentadas
        for aug_id in range(1, 8):
            augment_and_save(img, mask, base_name, aug_id)
