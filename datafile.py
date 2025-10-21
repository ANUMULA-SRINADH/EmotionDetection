import kagglehub

# Download latest version
path = kagglehub.dataset_download("astraszab/facial-expression-dataset-image-folders-fer2013")

print("Path to dataset files:", path)