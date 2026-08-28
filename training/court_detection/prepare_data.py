"""
Prepares the Court_Detector Roboflow export for Ultralytics YOLOv8-seg training.

Why this exists: inspection of the raw export showed it does NOT match Ultralytics'
expected `<split>/images/*.jpg` + `<split>/labels/*.txt` layout -- images sit directly
in `train/`, `valid/`, `test/` with a `labels/` subfolder alongside them. Ultralytics'
own label-path resolution replaces the LAST "/images/" segment of an image path with
"/labels/"; since our image paths have no "/images/" segment at all, that resolution
silently fails to find any labels. Rather than mutate the original Roboflow export in
place, this script builds a parallel `data/` folder here using hardlinks (same volume,
zero extra disk usage, confirmed working) into the conventional layout.

Also confirmed by inspection (see chat) -- NOT bbox format despite the original repo
README describing this as such: this is YOLO-SEGMENTATION format (variable-length
polygons: `class_id x1 y1 x2 y2 ...`), single class id 0, no class-names file shipped
anywhere in the export. Class name "court" below is OUR label, not authoritative --
flagged back to the user; confirm/rename if the FYP writeup needs a specific name.

SPLIT REBUILT AT THE SOURCE-VIDEO LEVEL (see chat + training/court_detection/README.md
for the full investigation). The raw Roboflow train/valid/test split is FRAME-LEVEL
RANDOM: filenames reveal 4 distinct source-video batches (by naming pattern -- bare
`{n}_jpg...`, `frame_{n}_jpg...`, `vid1_{n}_jpg...`, `vid2_{n}_jpg...`), and the
original split scattered frames from the same video across train/valid/test with gaps
as small as 1 frame -- confirmed via perceptual hashing that ~58% of the original
validation set was near-duplicate to a training frame (hamming distance <=10/256).
This script now pools ALL images from the raw train+valid+test folders, groups them by
source-video tag, and assigns each group WHOLLY to one split (GROUP_SPLIT below) so no
two frames of the same source video ever land in different splits. Group sizes don't
divide evenly into a clean 70/20/10 with only 4 groups; the assignment below was chosen
to land close to that ratio (train 67.0%, valid 19.6%, test 13.4% of 545 total images)
while keeping train diverse (2 of the 4 source videos) and giving valid/test each a
single held-out video.
"""
import os
import glob
import re

SRC_ROOT = os.path.join("data", "VisioTennis", "Dataset", "Court_Detector")
DST_ROOT = os.path.join("training", "court_detection", "data")
CLASS_NAMES = {0: "court"}  # ASSUMED -- no class-names file shipped in the export

# source-video tag -> destination split. Each tag's images stay entirely within
# one split (no frame-level mixing). Tag = filename prefix before the numeric frame id.
GROUP_SPLIT = {
    "": "train",        # bare `{n}_jpg...`          (232 images, 210 labeled)
    "vid1_": "train",   # `vid1_{n}_jpg...`           (133 images, 132 labeled)
    "frame_": "valid",  # `frame_{n}_jpg...`          (107 images, 107 labeled)
    "vid2_": "test",    # `vid2_{n}_jpg...`           ( 73 images,  73 labeled)
}


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def tag_of(filename):
    base = re.sub(r"_jpg\.rf\.[0-9a-f]+\.jpg$", "", filename)
    return re.sub(r"\d+$", "", base)


def clear_dst():
    import shutil
    for split in ("train", "valid", "test"):
        split_dir = os.path.join(DST_ROOT, split)
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)


def prepare():
    clear_dst()
    for split in ("train", "valid", "test"):
        os.makedirs(os.path.join(DST_ROOT, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DST_ROOT, split, "labels"), exist_ok=True)

    counts = {s: [0, 0] for s in ("train", "valid", "test")}  # [images, labels]
    unmapped = []
    for orig_split in ("train", "valid", "test"):
        src_dir = os.path.join(SRC_ROOT, orig_split)
        for img_path in sorted(glob.glob(os.path.join(src_dir, "*.jpg"))):
            fname = os.path.basename(img_path)
            tag = tag_of(fname)
            if tag not in GROUP_SPLIT:
                unmapped.append(fname)
                continue
            dst_split = GROUP_SPLIT[tag]
            base = os.path.splitext(fname)[0]
            link_or_copy(img_path, os.path.join(DST_ROOT, dst_split, "images", base + ".jpg"))
            counts[dst_split][0] += 1

            lbl_src = os.path.join(src_dir, "labels", base + ".txt")
            if os.path.isfile(lbl_src):
                link_or_copy(lbl_src, os.path.join(DST_ROOT, dst_split, "labels", base + ".txt"))
                counts[dst_split][1] += 1

    if unmapped:
        raise RuntimeError(f"{len(unmapped)} images had a tag not in GROUP_SPLIT: {unmapped[:10]}")

    for split, (n_img, n_lbl) in counts.items():
        print(f"[{split}] linked {n_img} images, {n_lbl} labels ({n_img - n_lbl} background-only)")


def write_data_yaml():
    yaml_path = os.path.join(DST_ROOT, "data.yaml")
    names_str = "\n".join(f"  {k}: {v}" for k, v in CLASS_NAMES.items())
    content = (
        f"path: {os.path.abspath(DST_ROOT)}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"names:\n{names_str}\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"wrote {yaml_path}")
    return yaml_path


def main():
    prepare()
    write_data_yaml()


if __name__ == "__main__":
    main()
