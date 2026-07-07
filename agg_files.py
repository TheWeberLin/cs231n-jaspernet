import ast
import pandas as pd


def collapse_to_image_rows(
    df,
    group_keys=("CameraPath", "Name"),
    bbox_cols=("bbox_x", "bbox_y", "bbox_w", "bbox_h"),
    species_col="pred_species",
):
    """
    Collapse one-row-per-detection into one-row-per-image.

    - group_keys: kept as-is (one value per image)
    - bbox_cols:  merged into a single 'bbox' column -> list of [x, y, w, h]
    - species_col and <species_col>_name: turned into {value: count} dicts
      (species/names that don't occur are simply absent — no count-0 keys)
    - every other column: if its value is constant across the image's
      detections, kept as a single scalar; if it varies, turned into a list
      (in detection order).
    """
   
    group_keys = list(group_keys)
    bbox_cols = list(bbox_cols)
    name_col = f"{species_col}_name"
    topk_col = f"{species_col}_topk"
    rough_cat = "rough_category"

    # Columns that become {value: count} dicts instead of lists.
    count_cols = {species_col, name_col, rough_cat}

    # If loaded from CSV, topk comes back as a string — parse back to objects.
    if topk_col in df.columns and df[topk_col].dtype == object:
        def _parse(v):
            if isinstance(v, str):
                try:
                    return ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    return v
            return v
        df[topk_col] = df[topk_col].map(_parse)

    # Every column that isn't a group key or a bbox component.
    other_cols = [c for c in df.columns if c not in group_keys and c not in bbox_cols]

    def _collapse_series(s):
        vals = s.tolist()
        # Treat as constant only if every value is equal (guard against
        # unhashable entries like dicts/lists by comparing directly).
        first = vals[0]
        try:
            if all(v == first for v in vals):
                return first
        except Exception:
            pass
        return vals

    def _count_dict(s):
        counts = {}
        for v in s.tolist():
            counts[v] = counts.get(v, 0) + 1   # only seen values become keys
        return counts

    records = []
    for keys, g in df.groupby(group_keys, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group_keys, keys))

        # bbox: always a list of [x, y, w, h], one per detection
        if all(c in g.columns for c in bbox_cols):
            rec["bbox"] = [[r[c] for c in bbox_cols] for _, r in g.iterrows()]

        for col in other_cols:
            if col in count_cols:
                rec[col] = _count_dict(g[col])
            else:
                rec[col] = _collapse_series(g[col])

        records.append(rec)

    return pd.DataFrame(records)