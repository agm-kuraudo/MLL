# Define the image name and container name
$imageName = "agm-karaudo/ml_trader_image_10"

# Build the Docker image
docker build -t $imageName .
