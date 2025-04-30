# CSC 4311: Face Matcher
This project is a facial recognition matcher. It takes an input image, compares it against known identities, and either matches or stores a new identity based on cosine similarity of facial embeddings.

The application is fully **Dockerized** and published on Docker Hub for easy use.

The application can be demoed using the images in `demo_faces`. It is recommended to add multiple images to help the `FaceMatcher` build a diverse portfolio.

## 🚀 Quick Start

### 1. Pull the Docker image

```bash
    docker pull jboss17/face-detector-pipeline:latest
```
### 2. Run the Container

```bash
    docker run -it jboss17/face-detector-pipeline:latest
```
## Running the FaceMatcher

Inside the container, run: 
```bash
    python scripts/app.py --image /path/to/image.jpg
```
Required argument:

`--image`: Path to image you want to process.

Optional arguments:

`--add_name`: If provided, adds a new identity to the known faces database.

`--threshold`: Cosine similarity threshold for matching (default=0.6)

## Example Usage

Match identity: 

```bash
    python scripts/app.py --image demo_faces/match.jpg
```

Add new identity: 

```bash
    python scripts/app.py --image demo_faces/addID_001.jpg --add_name "Dwight Schrute"
```

Adjust similarity threshold: 
```bash
    python scripts/app.py --image /path/to/image/unknown.jpg --threshold 0.5
```