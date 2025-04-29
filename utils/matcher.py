import torch
import torch.nn.functional as F
import pickle
import os

class FaceMatcher:
    """
    A lightweight face recognition engine based on embedding comparison.

    Stores embeddings for known identities and uses cosine similarity
    to match new faces. Automatically handles unknown faces by assigning
    new unique IDs.

    Attributes:
        known_faces (dict): Mapping of names to list of embeddings.
        threshold (float): Cosine similarity threshold for a match.
        _unknown_counter (int): Counter for labeling unknown faces.
    """
    def __init__(self, known_faces=None, threshold=.6):
        self.known_faces = known_faces if known_faces else {}
        self.threshold = threshold
        self._unknown_counter = 1

    def add(self, name, embedding):
        """
        Add a new embedding under a given identity name.

        Args:
            name (str): Person's name or ID.
            embedding (Tensor): Face embedding tensor.
        """
        if name not in self.known_faces:
            self.known_faces[name] = []
        self.known_faces[name].append(embedding)

    def match(self, embedding):
        """
        Compare a new embedding to all known embeddings and find the best match.
        If no match passes threshold, assign unique identity.

        Args:
            embedding (Tensor): Embedding of a detected face.

        Returns:
            str: Name of the matched identity or newly assigned unknown ID.
        """

        # If no known faces exist yet, create first unknown entry --> "person__001"
        if not self.known_faces:
            new_id = f"person_{self._unknown_counter:03d}"
            self._unknown_counter += 1
            self.known_faces[new_id] = [embedding]
            self.save()
            return new_id

        best_match = None
        best_score = -1

        # Compare against all known faces
        for name, embeddings in self.known_faces.items():
            all_embeddings = torch.stack(embeddings)
            similarities = F.cosine_similarity(embedding.unsqueeze(0), all_embeddings)
            # print(f"{name}: {similarities.tolist()}")
            max_sim = torch.max(similarities).item()
            # print(f"Max Sim: {max_sim}")

            # Updates best match if it beats previous best and passes threshold
            if max_sim > best_score and max_sim > self.threshold:
                best_score = max_sim
                best_match = name

        # No match passes threshold --> assign new ID
        if best_match is None:
            new_id = f"person_{self._unknown_counter:03d}"
            self._unknown_counter += 1
            self.known_faces[new_id] = [embedding]
            self.save()
            return new_id

        return best_match

    def save(self, path="data/known_faces.pkl"):
        """
       Save the current known faces and unknown counter to a file.

       Args:
           path (str): Path to save the serialized matcher state.
       """
        # Move embeddings to CPU for portability
        to_save = {key: [e.cpu() for e in val] for key, val in self.known_faces.items()}
        with open(path, 'wb') as f:
            pickle.dump({
                "known_faces": to_save,
                "unknown_counter": self._unknown_counter
            }, f)

    def load(self, path="data/known_faces.pkl"):
        """
       Load known faces and counter from a saved file.

       Args:
           path (str): Path to load the matcher state from.
       """
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.known_faces = data.get("known_faces", {})
                self._unknown_counter = data.get("unknown_counter", 1)
        else:
            print(f"No known faces found at {path}. Starting fresh.")