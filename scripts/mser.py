"""
mser.py
Simple MSER implementation (from-scratch) + OpenCV wrapper.

Usage:
    - from mser import mser_from_scratch, mser_opencv
    - masks, contours, boxes = mser_from_scratch(gray_image, delta=5, min_area=60, max_area=14400, max_variation=0.25)
    - boxes2 = mser_opencv(gray_image)  # requires OpenCV
"""

import numpy as np
from collections import defaultdict, deque
from scipy import ndimage as ndi
import cv2  # optional; only used by mser_opencv wrapper


# -------------------------
# OpenCV MSER wrapper
# -------------------------
def mser_opencv(gray, delta=5, min_area=60, max_area=14400, max_variation=0.25):
    """
    Use OpenCV's MSER detector. Returns bounding boxes.
    Input gray: uint8 grayscale image (0..255).
    Returns: list of bounding boxes (x,y,w,h) and contours.
    """
    if not hasattr(cv2, "MSER_create"):
        raise RuntimeError("OpenCV MSER not available in cv2 build.")
    mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area, _max_variation=max_variation)
    regions, bboxes = mser.detectRegions(gray)
    # convert regions to contours format
    contours = [np.array(r).reshape(-1, 2) for r in regions]
    return bboxes, contours, regions


# -------------------------
# From-scratch MSER
# -------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.zeros(n, dtype=np.int32)
        # seed pixel index for component (first pixel added)
        self.seed = np.full(n, -1, dtype=np.int32)

    def find(self, a):
        # path compression
        p = a
        while self.parent[p] != p:
            p = self.parent[p]
        # compress path
        while self.parent[a] != p:
            nxt = self.parent[a]
            self.parent[a] = p
            a = nxt
        return p

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        # union by size (attach smaller to larger)
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        # attach rb to ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        # if seed not set on ra, keep its seed; else keep existing
        if self.seed[ra] == -1:
            self.seed[ra] = self.seed[rb]
        return ra


def mser_from_scratch(gray, delta=5, min_area=60, max_area=14400, max_variation=0.25):
    """
    A simplified MSER detector implemented from scratch.
    - gray: 2D uint8 image
    - delta: window for variation calculation (typically 1-10)
    - min_area / max_area: region size constraints (in pixels)
    - max_variation: threshold for region stability

    Returns:
      masks: list of boolean masks (same shape as gray) for each MSER region
      contours: list of Nx2 arrays of (x, y) contour coords
      boxes: list of (x,y,w,h) bounding boxes
    Notes:
      This implementation is not as optimized as OpenCV's C++ MSER; it's educational and works well for moderate-image sizes.
    """
    assert gray.ndim == 2
    h, w = gray.shape
    npx = h * w

    # Flatten index helper
    def idx(r, c):
        return r * w + c

    # Pixel coordinates sorted by intensity (ascending)
    flat = gray.ravel()
    order = np.argsort(flat, kind='stable')  # indices into flattened array
    # We'll process pixels from darkest -> brightest (extremal region = darker-than-surroundings).
    # For bright-on-dark extremal regions, invert image before calling.

    uf = UnionFind(npx)
    active = np.zeros(npx, dtype=bool)  # whether pixel has been added to structure

    # history: for each root id store dict level -> size
    # We'll store sizes after finishing each intensity level.
    size_history = defaultdict(lambda: dict())
    # root_by_seed to map root to seed pixel (first pixel added to comp)
    # union-find object holds seed array, we'll set seed when create pixel

    # Precompute pixel row/col from flattened index for neighbors
    rows = np.arange(npx) // w
    cols = np.arange(npx) % w

    # Process intensity levels grouped
    unique_levels, inverse = np.unique(flat[order], return_inverse=True)
    # Build buckets: for each unique intensity value, list of flat indices that have that intensity (in processed order)
    buckets = []
    base = 0
    for lvl in unique_levels:
        # find range in order where intensity == lvl
        # inverse gives positions per ordered element; simpler: find mask over order
        # but better to iterate and pick consecutive runs
        # We'll find positions where flat[order] == lvl
        # But for large images, do faster:
        pass

    # Simpler approach: iterate levels 0..255 and process pixels with that intensity in the order list
    # Build map from intensity -> list of flat positions (in order they appear in 'order')
    intensity_buckets = [[] for _ in range(256)]
    for pos in order:
        intensity_buckets[flat[pos]].append(pos)

    # neighbor offsets (8-connected)
    neigh_offsets = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]

    # Map from root -> last recorded level (so we don't duplicate)
    last_recorded_level = dict()

    # Process levels
    for level in range(256):
        bucket = intensity_buckets[level]
        if not bucket:
            continue
        # For each pixel in this intensity bucket, activate and union with active neighbors
        for p in bucket:
            active[p] = True
            uf.parent[p] = p
            uf.size[p] = 1
            uf.seed[p] = p  # first pixel of component is its seed
            r = rows[p]; c = cols[p]
            # union to any already active neighbors
            for dr, dc in neigh_offsets:
                rr = r + dr; cc = c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    continue
                q = idx(rr, cc)
                if active[q]:
                    newroot = uf.union(p, q)
        # after finishing this level, record size for each current root
        # we iterate through bucket and record roots (roots may repeat; dedupe)
        roots = set()
        for p in bucket:
            roots.add(uf.find(p))
        for rt in roots:
            size_history[rt][level] = uf.size[rt]
            last_recorded_level[rt] = level

    # Now compute variation for each component across levels
    # For each root (component) we have a dict level->size.
    # For a given level l, variation v(l) = (size(l+delta) - size(l-delta)) / size(l)
    # We need to check local minima of v and pick those < max_variation and within area bounds.
    # We'll pick for each component the level(s) where it's stable; then get final mask by thresholding image <= level and extracting connected component containing the seed pixel.

    masks = []
    contours = []
    boxes = []

    # convenience: convert size_history keys to list
    for root, hist in size_history.items():
        # hist: mapping level -> size (only for levels where that root existed)
        levels = np.array(sorted(hist.keys()))
        sizes = np.array([hist[l] for l in levels], dtype=float)

        if len(levels) == 0:
            continue

        # compute variation for levels where we can evaluate (i.e. have levels +/- delta)
        # For each index i where levels[i] has levels[i]-delta and levels[i]+delta present
        level_to_index = {l: i for i, l in enumerate(levels)}
        variations = {}
        for i, l in enumerate(levels):
            l_minus = l - delta
            l_plus = l + delta
            if l_minus in hist and l_plus in hist and hist[l] > 0:
                v = (hist[l_plus] - hist[l_minus]) / float(hist[l])
                variations[l] = v
        if not variations:
            continue

        # find local minima in variations (simple approach: compare to neighbors in level sequence)
        var_items = sorted(variations.items())  # list of (level, variation)
        # pick global minima below threshold
        min_level, min_var = min(var_items, key=lambda x: x[1])
        if min_var <= max_variation:
            # area constraint (use area at min_level)
            area_at_min = hist[min_level]
            if area_at_min >= min_area and area_at_min <= max_area:
                # reconstruct region mask by thresholding at intensity <= min_level and then selecting connected component containing the seed pixel
                seed_idx = uf.seed[root]
                if seed_idx < 0:
                    continue
                thresh_mask = (gray <= min_level)
                # label connected components in thresh_mask
                labeled, num = ndi.label(thresh_mask)
                seed_r = rows[seed_idx]; seed_c = cols[seed_idx]
                lab = labeled[seed_r, seed_c]
                if lab == 0:
                    continue
                comp_mask = (labeled == lab)
                # optional: filter by area again
                comp_area = int(comp_mask.sum())
                if comp_area < min_area or comp_area > max_area:
                    continue
                # compute contour using OpenCV if available, else simple border extraction
                try:
                    # convert boolean mask to uint8 image for findContours
                    im8 = (comp_mask.astype('uint8') * 255)
                    contours_found, _ = cv2.findContours(im8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours_found:
                        cnt = sorted(contours_found, key=lambda x: cv2.contourArea(x))[-1]
                        cnt = cnt.reshape(-1, 2)
                    else:
                        cnt = np.column_stack(np.where(comp_mask))[:, ::-1]
                except Exception:
                    cnt = np.column_stack(np.where(comp_mask))[:, ::-1]
                ys, xs = np.where(comp_mask)
                x0 = int(xs.min()); y0 = int(ys.min()); x1 = int(xs.max()); y1 = int(ys.max())
                masks.append(comp_mask)
                contours.append(cnt)
                boxes.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))

    return masks, contours, boxes


# -------------------------
# Example script usage
# -------------------------
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print("Usage: python mser.py <imagefile>")
        sys.exit(1)
    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Running from-scratch MSER (this might be slow on large images)...")
    masks, contours, boxes = mser_from_scratch(gray, delta=5, min_area=30, max_area=20000, max_variation=0.25)
    print(f"Found {len(boxes)} regions")

    # draw boxes
    disp = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(disp, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    plt.title(f"MSER from-scratch: {len(boxes)} regions")
    plt.axis('off')
    plt.show()

    # also show OpenCV's MSER for comparison if available
    try:
        bboxes2, contours2, regions2 = mser_opencv(gray)
        disp2 = img.copy()
        for (x, y, w, h) in bboxes2:
            cv2.rectangle(disp2, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 1)
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(disp2, cv2.COLOR_BGR2RGB))
        plt.title(f"OpenCV MSER: {len(bboxes2)} boxes")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print("OpenCV MSER not available or failed:", e)
