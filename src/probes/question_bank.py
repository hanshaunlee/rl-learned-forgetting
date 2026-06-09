"""Question bank: derive (question, answer, caption_relevant) tuples from COCO annotations.

For any image_id, exposes:
  - all_questions(image_id): list of every answerable question
  - sample_question(image_id, p_caption_relevant): one sampled tuple

Questions are binary yes/no. Caption relevance is computed by checking whether
the entity asked about appears in any of the image's 5 COCO captions.
"""
from pathlib import Path
from typing import Optional

import numpy as np
from pycocotools.coco import COCO

from src.paths import COCO_ANNOTATIONS

# Question types — fixed IDs let downstream models embed question type
QTYPE_OBJECT_PRESENCE = 0
QTYPE_SPATIAL = 1
QTYPE_COUNT = 2

# Spatial regions — half-image partitions
SPATIAL_REGIONS = ["left", "right", "top", "bottom"]

# Count thresholds
COUNT_THRESHOLDS = [1, 3, 5]  # "more than N objects?"

# Map from COCO category name -> set of caption synonyms (lowercase)
# We'll use this for caption-relevance matching.
CATEGORY_SYNONYMS = {
    "person": {"person", "man", "woman", "boy", "girl", "child", "kid", "people", "guy", "lady"},
    "bicycle": {"bicycle", "bike"},
    "car": {"car", "vehicle", "automobile"},
    "motorcycle": {"motorcycle", "motorbike", "bike"},
    "airplane": {"airplane", "plane", "aircraft", "jet"},
    "bus": {"bus"},
    "train": {"train"},
    "truck": {"truck"},
    "boat": {"boat", "ship", "vessel"},
    "traffic light": {"traffic light", "stoplight", "signal"},
    "dog": {"dog", "puppy"},
    "cat": {"cat", "kitten"},
    "tv": {"tv", "television", "monitor"},
    "couch": {"couch", "sofa"},
    "potted plant": {"plant", "houseplant"},
    # Categories not listed default to the lowercased category name itself
}

class QuestionBank:
    """Loads COCO annotations once, then derives questions per image_id."""

    def __init__(self):
        self.coco = COCO(str(COCO_ANNOTATIONS / "instances_val2017.json"))
        self.captions = COCO(str(COCO_ANNOTATIONS / "captions_val2017.json"))

        # Cache all COCO category names and IDs
        self.cat_ids = self.coco.getCatIds()
        self.cats = self.coco.loadCats(self.cat_ids)
        self.cat_id_to_name = {c["id"]: c["name"] for c in self.cats}

    def _object_presence_questions(self, image_id: int) -> list[dict]:
        """For each category, is it present in the image? Returns one question per category."""
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        present_cat_ids = {a["category_id"] for a in anns if not a.get("iscrowd", 0)}

        questions = []
        for cat in self.cats:
            answer = cat["id"] in present_cat_ids
            questions.append({
                "qtype": QTYPE_OBJECT_PRESENCE,
                "qid": f"obj_{cat['id']}",
                "text": f"Was there a {cat['name']} in the image?",
                "category_name": cat["name"],
                "category_id": cat["id"],
                "answer": int(answer),
            })
        return questions

    def _spatial_questions(self, image_id: int) -> list[dict]:
        """Was there any object centered in each half-image region?"""
        img_info = self.coco.loadImgs(image_id)[0]
        W, H = img_info["width"], img_info["height"]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = [a for a in self.coco.loadAnns(ann_ids) if not a.get("iscrowd", 0)]

        # Compute object centers
        centers = []
        for a in anns:
            x, y, w, h = a["bbox"]
            centers.append((x + w/2, y + h/2))

        def any_in(region):
            for cx, cy in centers:
                if region == "left"   and cx < W/2: return True
                if region == "right"  and cx >= W/2: return True
                if region == "top"    and cy < H/2: return True
                if region == "bottom" and cy >= H/2: return True
            return False

        questions = []
        for region in SPATIAL_REGIONS:
            questions.append({
                "qtype": QTYPE_SPATIAL,
                "qid": f"spatial_{region}",
                "text": f"Was there an object in the {region} half of the image?",
                "region": region,
                "answer": int(any_in(region)),
            })
        return questions

    def _count_questions(self, image_id: int) -> list[dict]:
        """Were there more than N objects?"""
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = [a for a in self.coco.loadAnns(ann_ids) if not a.get("iscrowd", 0)]
        n_objects = len(anns)

        questions = []
        for threshold in COUNT_THRESHOLDS:
            questions.append({
                "qtype": QTYPE_COUNT,
                "qid": f"count_gt_{threshold}",
                "text": f"Were there more than {threshold} objects in the image?",
                "threshold": threshold,
                "answer": int(n_objects > threshold),
            })
        return questions

    def all_questions(self, image_id: int) -> list[dict]:
        """Return every derivable question for this image (no caption relevance yet)."""
        return (
            self._object_presence_questions(image_id)
            + self._spatial_questions(image_id)
            + self._count_questions(image_id)
        )

    def _caption_words(self, image_id: int) -> set[str]:
        """Lowercased set of words across all 5 captions for this image."""
        cap_ann_ids = self.captions.getAnnIds(imgIds=image_id)
        captions = self.captions.loadAnns(cap_ann_ids)
        words = set()
        for c in captions:
            # Simple lowercase tokenization on whitespace, strip basic punctuation
            tokens = c["caption"].lower().replace(",", " ").replace(".", " ").split()
            words.update(tokens)
        return words

    def _is_caption_relevant(self, question: dict, caption_words: set[str]) -> bool:
        """Is the entity being asked about mentioned in any caption?"""
        if question["qtype"] == QTYPE_OBJECT_PRESENCE:
            cat_name = question["category_name"]
            # Check synonyms if we have them, otherwise the lowercase category name
            synonyms = CATEGORY_SYNONYMS.get(cat_name, {cat_name.lower()})
            # Multi-word categories: check if all words appear (rough)
            for syn in synonyms:
                if " " in syn:
                    if all(w in caption_words for w in syn.split()):
                        return True
                else:
                    if syn in caption_words:
                        return True
            return False

        # Spatial and count questions aren't tied to specific entities — treat as always
        # "non-caption-specific" so caption-relevance weighting affects only object-presence.
        return False

    def annotated_questions(self, image_id: int) -> list[dict]:
        """All questions with caption_relevant flag added."""
        questions = self.all_questions(image_id)
        caption_words = self._caption_words(image_id)
        for q in questions:
            q["caption_relevant"] = self._is_caption_relevant(q, caption_words)
        return questions
    
    def sample_question(
        self,
        image_id: int,
        p_caption_relevant: float = 0.7,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Sample one question for this image, biased toward caption-relevant ones.

        With probability p_caption_relevant, sample from the subset of questions whose
        asked-about entity is mentioned in a caption. With probability 1 - p_caption_relevant
        (or always, if no caption-relevant questions exist), sample uniformly from all
        questions.
        """
        if rng is None:
            rng = np.random.default_rng()

        questions = self.annotated_questions(image_id)
        relevant = [q for q in questions if q["caption_relevant"]]

        if relevant and rng.random() < p_caption_relevant:
            return dict(relevant[rng.integers(len(relevant))])
        return dict(questions[rng.integers(len(questions))])

# Template index layout (must match attribute_probe.py):
#   0..79   object presence (by COCO category index, sorted by category id)
#   80..83  spatial (left, right, top, bottom)
#   84..86  count (>1, >3, >5)

def _build_category_index(coco):
    """Map COCO category_id -> contiguous index 0..79 (sorted by category id)."""
    cat_ids = sorted(coco.getCatIds())
    return {cid: i for i, cid in enumerate(cat_ids)}


def question_to_template_idx(question: dict, cat_id_to_idx: dict) -> int:
    """Map a question dict to its template index 0..86."""
    if question["qtype"] == QTYPE_OBJECT_PRESENCE:
        return cat_id_to_idx[question["category_id"]]            # 0..79
    if question["qtype"] == QTYPE_SPATIAL:
        return 80 + SPATIAL_REGIONS.index(question["region"])    # 80..83
    if question["qtype"] == QTYPE_COUNT:
        return 80 + 4 + COUNT_THRESHOLDS.index(question["threshold"])  # 84..86
    raise ValueError(f"Unknown qtype {question['qtype']}")

if __name__ == "__main__":
    qb = QuestionBank()
    image_id = qb.coco.getImgIds()[0]

    # Sample 1000 questions and see how often we land on caption-relevant ones
    rng = np.random.default_rng(0)
    samples = [qb.sample_question(image_id, p_caption_relevant=0.7, rng=rng) for _ in range(1000)]
    n_relevant = sum(1 for q in samples if q["caption_relevant"])
    print(f"Out of 1000 samples with p=0.7: {n_relevant} caption-relevant ({n_relevant/10:.1f}%)")

    # Same with p=0
    samples = [qb.sample_question(image_id, p_caption_relevant=0.0, rng=rng) for _ in range(1000)]
    n_relevant = sum(1 for q in samples if q["caption_relevant"])
    print(f"Out of 1000 samples with p=0.0: {n_relevant} caption-relevant ({n_relevant/10:.1f}%)")