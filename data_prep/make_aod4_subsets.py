"""Create the AOD4 bird-drone subsets used in the paper from the released file lists.

Input : AOD4 bird/drone-only copy (images/{train,valid,test}, labels/{...}), classes 0=bird 1=drone
        (airplane and helicopter removed, see README).
Output: AOD4_bd_<p>/ for p in 1, 1_s1, 1_s2, 1_s3, 5, 10, 25, 50 (train = listed images;
        valid and test = the full AOD4 valid/test sets), plus a YAML per subset and for the full set.
        Also AOD4_bd_clean_test/ = test set minus the 199 images listed in splits/leaked_test_images.txt.
Files are symlinked, not copied.
"""
import os

AOD4 = os.path.abspath(os.environ.get("AOD4_ROOT", "data/AOD4_bird_drone_only"))
OUT = os.path.abspath(os.environ.get("DATA_ROOT", "data"))
SPLITS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splits")
YAML = "path: {p}\ntrain: images/train\nval: images/valid\ntest: images/test\n\nnc: 2\nnames:\n  0: bird\n  1: drone\n"

def label_of(img):
    return os.path.splitext(img)[0] + ".txt"

def link(src_split, names, dst_root, dst_split):
    for kind in ("images", "labels"):
        os.makedirs(os.path.join(dst_root, kind, dst_split), exist_ok=True)
    for n in names:
        for kind, fn in (("images", n), ("labels", label_of(n))):
            s = os.path.join(AOD4, kind, src_split, fn)
            d = os.path.join(dst_root, kind, dst_split, fn)
            if os.path.exists(s) and not os.path.lexists(d):
                os.symlink(s, d)

def listdir(split):
    return sorted(os.listdir(os.path.join(AOD4, "images", split)))

with open(os.path.join(AOD4, "aod4_bd.yaml"), "w") as fh:
    fh.write(YAML.format(p=AOD4))

for tag in ["1", "1_s1", "1_s2", "1_s3", "5", "10", "25", "50"]:
    names = [l.strip() for l in open(os.path.join(SPLITS, f"AOD4_bd_{tag}_train.txt")) if l.strip()]
    root = os.path.join(OUT, f"AOD4_bd_{tag}")
    link("train", names, root, "train")
    link("valid", listdir("valid"), root, "valid")
    link("test", listdir("test"), root, "test")
    with open(os.path.join(root, f"aod4_bd_{tag}.yaml"), "w") as fh:
        fh.write(YAML.format(p=root))
    print(f"AOD4_bd_{tag}: {len(names)} train images")

leaked = {l.strip() for l in open(os.path.join(SPLITS, "leaked_test_images.txt")) if l.strip()}
clean = [n for n in listdir("test") if n not in leaked]
root = os.path.join(OUT, "AOD4_bd_clean_test")
link("test", clean, root, "test")
link("valid", listdir("valid"), root, "valid")
os.makedirs(os.path.join(root, "images", "train"), exist_ok=True)
with open(os.path.join(root, "aod4_bd_clean.yaml"), "w") as fh:
    fh.write(YAML.format(p=root).replace("train: images/train", "train: images/valid"))
print(f"AOD4_bd_clean_test: {len(clean)} test images (leakage-free)")
