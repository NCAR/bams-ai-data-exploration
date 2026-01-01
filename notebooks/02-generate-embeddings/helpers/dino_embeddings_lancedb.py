import torch
import lancedb
import timm
import io
import argparse
import numpy as np
import pyarrow as pa
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class LanceBlobDataset(Dataset):
    def __init__(self, db_uri: str, table_name: str, transforms=None, id_col="id", blob_col="image", existing_ids=None):
        self.db_uri = db_uri
        self.table_name = table_name
        self.transforms = transforms
        self.id_col = id_col
        self.blob_col = blob_col
        self.existing_ids = existing_ids or set()
        
        # Connect briefly to map indices to row IDs (if filtering is needed)

        db = lancedb.connect(db_uri)
        tbl = db.open_table(table_name)
        self.length = tbl.count_rows()
        self.table = None

    def _init_worker_conn(self):
        if self.table is None:
            db = lancedb.connect(self.db_uri)
            self.table = db.open_table(self.table_name)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._init_worker_conn()
        
        try:
            # We fetch the ID first to see if we should skip it
            
            row = self.table.to_lance().take([idx], columns=[self.id_col, self.blob_col])
            img_id = str(row[self.id_col][0].as_py())
            
            # RESUME LOGIC: Return None immediately if already processed
            if img_id in self.existing_ids:
                return None

            blob = row[self.blob_col][0].as_py()
            if blob is None: return None
            
            # Decode
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            if self.transforms:
                tensor = self.transforms(img)
            
            return tensor, img_id
            
        except Exception:
            return None

def collate_skip_empty(batch):
    """Filters out skipped (None) items."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, []
    tensors, ids = zip(*batch)
    return torch.stack(tensors), ids


def main():
    parser = argparse.ArgumentParser()
    # Basic Config
    parser.add_argument("--db", type=str, default="./my_lancedb")
    parser.add_argument("--src", type=str, default="raw_images")
    parser.add_argument("--model", type=str, default="vit_large_patch14_dinov2.lvd142m")
    
    # Tuning
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    
    # Output Optimizations
    parser.add_argument("--resume", action="store_true", help="Skip images already in DB")
    parser.add_argument("--fp16_out", action="store_true", help="Store embeddings as float16 (50% smaller)")
    parser.add_argument("--limit", type=int, default=None, help="Stop after processing N images")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Summary ---")
    print(f"Model: {args.model} | Device: {device}")
    print(f"Output: {'Float16' if args.fp16_out else 'Float32'} | Resume: {args.resume}")

    # A. Setup DB & Resume Logic
    db = lancedb.connect(args.db)
    
    # Create tables if they don't exist
    img_tbl = db.create_table("image_embeddings", schema=None, exist_ok=True)
    patch_tbl = db.create_table("patch_embeddings", schema=None, exist_ok=True)
    
    existing_ids = set()
    if args.resume:
        print("Scanning existing embeddings...")
        # scan only the ID column
        try:
            # Check if table has data
            if img_tbl.count_rows() > 0:
                scanner = img_tbl.to_lance().scanner(columns=["id"])
                for batch in scanner.to_batches():
                    existing_ids.update(batch["id"].to_pylist())
            print(f"Found {len(existing_ids)} existing images. Resuming...")
        except Exception as e:
            print(f"Resume scan failed (table might be empty): {e}")

    # B. Load Model
    print("Loading Model...")
    model = timm.create_model(args.model, pretrained=True, num_classes=0, global_pool='')
    model.to(device).eval()
    if device == "cuda": model.half() # AMP Inference

    # C. Data Pipeline
    data_cfg = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_cfg, is_training=False)
    
    dataset = LanceBlobDataset(
        args.db, 
        args.src, 
        transforms=transforms, 
        existing_ids=existing_ids
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_skip_empty,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    # D. Processing Loop
    total_processed = 0
    storage_dtype = np.float16 if args.fp16_out else np.float32

    # Tqdm total is approximate because we might skip many rows
    pbar = tqdm(loader, unit="batch", desc="Inference")
    
    for batch_tensors, batch_ids in pbar:
        if batch_tensors is None: continue

        # 1. Inference
        batch_tensors = batch_tensors.to(device, non_blocking=True)
        if device == "cuda": batch_tensors = batch_tensors.half()

        with torch.inference_mode():
            features = model(batch_tensors)

        # 2. Extract Embeddings
        # (Assuming CLS pooling for simplicity)
        global_embs = features[:, 0, :]    # [CLS]
        patch_embs = features[:, 1:, :]    # Patches

        # Move to CPU & Cast to storage dtype
        global_np = global_embs.float().cpu().numpy().astype(storage_dtype)
        patch_np = patch_embs.float().cpu().numpy().astype(storage_dtype)

        # 3. Format Data
        img_rows = []
        patch_rows = []

        for i, img_id in enumerate(batch_ids):
            img_rows.append({
                "id": img_id, 
                "vector": global_np[i], 
                "pooling": "cls"
            })
            
            # Flatten patches
            curr_patches = patch_np[i]
            for p_idx in range(curr_patches.shape[0]):
                patch_rows.append({
                    "id": img_id,
                    "patch_index": p_idx,
                    "vector": curr_patches[p_idx]
                })

        # 4. Write
        if img_rows: img_tbl.add(img_rows)
        if patch_rows: patch_tbl.add(patch_rows)
        
        # 5. Check Limits
        total_processed += len(img_rows)
        pbar.set_postfix({"Count": total_processed})
        
        if args.limit and total_processed >= args.limit:
            print(f"\nLimit of {args.limit} reached. Stopping.")
            break

    print(f"Done. Processed {total_processed} new images.")

if __name__ == "__main__":
    main()