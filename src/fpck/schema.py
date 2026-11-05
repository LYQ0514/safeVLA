"""Feature/evidence contracts shared by collection, fitting, and evaluation."""
REQUIRED_SAMPLE_KEYS={"sample_id","split","features","target","task","episode","chunk_hash","buffer_index","label_source"}
def validate_sample(row):
    missing=REQUIRED_SAMPLE_KEYS-set(row)
    if missing: raise ValueError("missing keys: "+",".join(sorted(missing)))
    if row["split"] not in {"train","calibration","test"}: raise ValueError("invalid split")
    if not isinstance(row["features"],list) or not row["features"]: raise ValueError("features must be nonempty list")
    if not all(isinstance(x,(int,float)) for x in row["features"]): raise ValueError("features must be numeric")
    if not isinstance(row["target"],(int,float)): raise ValueError("target must be numeric")
    return True
