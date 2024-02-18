import os

def synchronize_folders(folder1, folder2, extension1, extension2):
    # Get a list of files in folder1 with the specified extension1, removing the extension
    files1 = [file.split('.')[0] for file in os.listdir(folder1) if file.endswith(extension1)]
    
    # Get a list of files in folder2 with the specified extension2, removing the extension
    files2 = [file.split('.')[0] for file in os.listdir(folder2) if file.endswith(extension2)]

    # Find the common files by creating a set of unique file names from folder1
    common_files = set(files1)

    # Iterate through files in folder2
    for file in files2:
        # Check if the file is not in the common_files set
        if file not in common_files:
            # Construct the full path of the file in folder2 with its extension and remove it
            file_path = os.path.join(folder2, file)
            os.remove(file_path + ".wav") 

# Paths of the folders and file extensions
folder_path1 = "C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/database_v3/espectrogramas/female_female"
folder_path2 = "C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/database_v3/audios/female_female"
extension1 = ".png"
extension2 = ".wav"

# Synchronize the folders by removing files in folder2 that do not have a corresponding file in folder1
synchronize_folders(folder_path1, folder_path2, extension1, extension2)
