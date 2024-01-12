import os

def synchronize_folders(folder1, folder2, extension1, extension2):
    files1 = [file.split('.')[0] for file in os.listdir(folder1) if file.endswith(extension1)]
    files2 = [file.split('.')[0] for file in os.listdir(folder2) if file.endswith(extension2)]

    common_files = set(files1)

    for file in files2:
        if file not in common_files:
            file_path = os.path.join(folder2, file)
            os.remove(file_path + ".wav")

# Rutas de las carpetas y extensiones
folder_path1 = "C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/database_v3/espectrogramas/male_male"
folder_path2 = "C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/database_v3/audios/male_male"
extension1 = ".png"
extension2 = ".wav"

# Sincronizar las carpetas eliminando archivos que no tienen correspondencia
synchronize_folders(folder_path1, folder_path2, extension1, extension2)
