# pipeline/sam2_utils.py
import numpy as np

class AnySamPredictor:
    def __init__(self, model_type: str, checkpoint: str, device: str = "cuda"):
        self.device = device
        self.is_sam2 = True
        try:
            # SAM 2 first
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # model_type should be a config path string like:
            # "configs/sam2.1/sam2.1_hiera_l.yaml"
            sam2 = build_sam2(model_type, checkpoint, device=device)
            self.pred = SAM2ImagePredictor(sam2)
            self.kind = "sam2"
        except Exception as e_sam2:
            try:
                # Fallback: original SAM (only if installed)
                from segment_anything import sam_model_registry, SamPredictor
                sam = sam_model_registry[model_type](checkpoint=checkpoint)
                sam.to(device)
                self.pred = SamPredictor(sam)
                self.is_sam2 = False
                self.kind = "sam"
            except Exception as e_sam:
                raise RuntimeError(
                    "Init SAM-2 和老 SAM 都失败了。\n"
                    f"SAM-2 error: {e_sam2}\nSAM error: {e_sam}\n"
                    "请确认：\n"
                    "  • 已 pip install -e deps/sam2_official，并可见 sam2/configs；\n"
                    "  • 传入的 model_type 为 YAML 配置路径（或老 SAM 的型号名）。"
                )

    def set_image(self, image_rgb: np.ndarray):
        self.pred.set_image(image_rgb)  # HWC, uint8, RGB

    def predict_with_points(self, pts_xy: np.ndarray, labels: np.ndarray, multimask=True):
        masks, scores, _ = self.pred.predict(
            point_coords=pts_xy, point_labels=labels,
            multimask_output=multimask
        )
        # [N,H,W] -> [H,W,N]
        if masks.ndim == 3 and masks.shape[0] < masks.shape[-1]:
            import numpy as _np
            masks = _np.transpose(masks, (1, 2, 0))
        return masks, scores
