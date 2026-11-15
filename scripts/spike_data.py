"""Spike passages and a no-key stand-in for embeddings.

The passages are short summaries written for this test, NOT Plutarch's text.
They are chosen for near-miss names: two Gracchi, two Catos.
"""

import hashlib
import math
import re

DIMS = 1024  # matches chunks.embedding vector(1024)

PASSAGES: list[tuple[str, str]] = [
    (
        "Tiberius Gracchus",
        (
            "As tribune, Tiberius Gracchus proposed a land law to give public land to "
            "poor citizens. He had his fellow tribune Octavius removed from office, and "
            "was killed on the Capitol by senators led by Scipio Nasica."
        ),
    ),
    (
        "Caius Gracchus",
        (
            "Caius Gracchus, the younger brother, became tribune ten years later. He "
            "passed a grain law and founded colonies, and died on the Aventine after "
            "the consul Opimius moved against his followers."
        ),
    ),
    (
        "Cato the Elder",
        (
            "Marcus Cato, a farmer's son from Tusculum, rose to be censor and was famous "
            "for strict morals. He ended his speeches by saying Carthage must be destroyed."
        ),
    ),
    (
        "Cato the Younger",
        (
            "Cato the Younger was a Stoic who opposed Caesar. After Caesar's victory in "
            "Africa, he took his own life at Utica rather than accept Caesar's pardon."
        ),
    ),
    (
        "Crassus",
        (
            "Crassus was the richest man in Rome. He defeated the slave revolt of "
            "Spartacus and later died fighting the Parthians at Carrhae."
        ),
    ),
    (
        "Pompey",
        (
            "Pompey, called Magnus, cleared the sea of pirates and won wars in the east. "
            "Caesar defeated him at Pharsalus, and he was killed when he landed in Egypt."
        ),
    ),
]


def hashed_vector(text: str, dims: int = DIMS) -> list[float]:
    """Word counts hashed into a fixed-size unit vector.

    Deterministic and free. Texts that share words come out close, so vector
    search gives sensible results without an embedding API.
    """
    vec = [0.0] * dims
    for word in re.findall(r"[a-z]+", text.lower()):
        bucket = int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "big")
        vec[bucket % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
