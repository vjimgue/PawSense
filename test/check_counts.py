
import os

def count_items(directory):
    enc_folders = 0
    enc_files = 0
    
    # Resolve absolute path to avoid confusion
    abs_dir = os.path.abspath(directory)
    
    if not os.path.exists(abs_dir):
        return 0, 0
        
    for root, dirs, files in os.walk(abs_dir):
        # Filter out empty dirs strings if any (though os.walk usually handles this)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        enc_folders += len(dirs)
        
        # Count images
        enc_files += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
    return enc_folders, enc_files

# Paths are relative to the script location (in 'test/' folder), so we go up one level
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
source_dir = os.path.join(base_path, "dataset_perros", "images", "Images")
dest_dir = os.path.join(base_path, "dataset_recortado")

print(f"Base path: {base_path}")
print(f"Checking Source: {source_dir}")
src_folders, src_files = count_items(source_dir)

print(f"Checking Dest:   {dest_dir}")
dst_folders, dst_files = count_items(dest_dir)

print("\n--- Resultados ---")
print(f"Original (dataset_perros) - Carpetas: {src_folders}, Imagenes: {src_files}")
print(f"Recortado (dataset_recortado) - Carpetas: {dst_folders}, Imagenes: {dst_files}")

if src_folders == dst_folders and src_files == dst_files:
    print("\n✅ Los conteos coinciden exactamente.")
else:
    print("\n❌ Los conteos NO coinciden.")
    folder_diff = src_folders - dst_folders
    file_diff = src_files - dst_files
    print(f"Diferencia de carpetas: {folder_diff}")
    print(f"Diferencia de imagenes: {file_diff}")
