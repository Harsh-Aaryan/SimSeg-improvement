import cv2
import argparse
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import pydensecrf.densecrf as dcrf
from transformers import AutoTokenizer
from torchvision import transforms as T

# Apex / Torch DDP imports
try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    pass
from torch.nn.parallel import DistributedDataParallel as torch_DDP

# SimSeg imports
from simseg.core import init_device, cfg, update_cfg
from simseg.models import PIPELINE
from simseg.utils import build_from_cfg, ENV, logger
from simseg.utils.prompt import openai_imagenet_template
from simseg.utils.interpolate_pe import interpolate_pos_embed
from simseg.core.hooks.checkpoint import get_dist_state_dict
from simseg.tasks.clip.config import task_cfg_init_fn, update_clip_config

def create_color_mask(mask, num_classes):
    # Deterministic palette: reproducible random colors per class
    np.random.seed(0)
    palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(num_classes):
        color_mask[mask == cls_id] = palette[cls_id]
    return color_mask

def dense_crf(img, probs, n_labels=2):
    h, w = probs.shape
    probs = np.expand_dims(probs, 0)
    probs = np.append(1 - probs, probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, n_labels)
    U = -np.log(probs + 1e-8)
    U = U.reshape((n_labels, -1))
    U = np.ascontiguousarray(U).astype(np.float32)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=40, srgb=13, rgbim=img, compat=10)

    Q = d.inference(3)
    return np.argmax(np.array(Q), axis=0).reshape((h, w))


def zero_shot_classifier(model, classnames, make_template, tokenizer, ENV):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = make_template(classname)
            texts = tokenizer(texts, padding='max_length', truncation=True, max_length=25)
            input_ids = torch.tensor(texts["input_ids"]).to(ENV.rank)
            attention_mask = torch.tensor(texts["attention_mask"]).to(ENV.rank)

            class_embeddings = model.module.forward_text_feature(input_ids, attention_mask)
            class_embeddings = model.module.forward_text_project(class_embeddings, attention_mask)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.transpose(0, 1)


def run_single_image(image_path, model, cfg, tokenizer, seg_categories, top_cls_num, output_path):
    raw_img = Image.open(image_path).convert("RGB")

    # Build preprocessing transform (since cfg.transforms.val may not exist)
    transform = T.Compose([
        T.Resize(cfg.transforms.input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(cfg.transforms.input_size),
        T.ToTensor(),
        T.Normalize(mean=cfg.transforms.normalize.mean,
                    std=cfg.transforms.normalize.std),
    ])

    img_tensor = transform(raw_img).unsqueeze(0).to(ENV.device)

    image_mean = torch.tensor(cfg.transforms.normalize.mean, device=ENV.device).view(1, 3, 1, 1)
    image_norm = torch.tensor(cfg.transforms.normalize.std, device=ENV.device).view(1, 3, 1, 1)

    patch_size = 16
    num_patch = cfg.transforms.input_size // patch_size

    label_text_feature = zero_shot_classifier(model, seg_categories, openai_imagenet_template, tokenizer, ENV)

    with torch.no_grad():
        image_feature = model.module.forward_image_feature(img_tensor)
        image_feature_pooled = model.module.forward_image_project(image_feature)
        image_feature = model.module.image_projection(image_feature)

    image_raw = (((img_tensor * image_norm) + image_mean) * 255).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    image_a = Image.fromarray(image_raw[0])

    im_f_a = F.normalize(image_feature[0], dim=-1, p=2)
    im_avg_a = image_feature_pooled[0]

    i2t_scores = torch.sum(im_avg_a.unsqueeze(0) * label_text_feature, dim=1)
    topk_scores, topk_index = i2t_scores.topk(top_cls_num)
    threshold = topk_scores.mean() + 1.0 * topk_scores.std()

    raw_H, raw_W = raw_img.size[1], raw_img.size[0]
    temp_pred = np.zeros((len(seg_categories), raw_H, raw_W))

    candidate_class_num = 5
    for i, index in enumerate(topk_index[:candidate_class_num]):
        if index in [0, 255]:
            continue

        attn_ai2at = im_f_a @ label_text_feature[index].unsqueeze(-1)
        attn_ai2at = attn_ai2at.reshape(num_patch, num_patch)
        attn_ai2at = F.interpolate(attn_ai2at.unsqueeze(0).unsqueeze(0),
                                   scale_factor=patch_size, mode="nearest")[0][0]
        attn_ai2at = attn_ai2at.cpu().numpy()

        score = float(i2t_scores[index])
        if score < threshold:
            break

        norm_attn = (attn_ai2at - attn_ai2at.min()) / (attn_ai2at.max() - attn_ai2at.min() + 1e-6)
        binary_mask = dense_crf(np.array(image_a).astype(np.uint8), norm_attn) * 255
        binary_mask = binary_mask.astype('uint8')

        kernel = np.ones((7, 7), dtype=np.uint8)
        final_mask = cv2.dilate(binary_mask, kernel, 5)
        final_mask = cv2.erode(final_mask, kernel, 3)
        final_mask = cv2.resize(final_mask.astype(np.uint8), dsize=(raw_W, raw_H), interpolation=cv2.INTER_NEAREST)

        temp_pred[index] = final_mask * score

    mask_out = temp_pred.argmax(0).astype(np.uint8)
    cv2.imwrite(output_path, mask_out)

    color_mask = create_color_mask(mask_out, len(seg_categories))
    cv2.imwrite(output_path.replace(".png", "_color.png"), color_mask)

    overlay = cv2.addWeighted(np.array(raw_img), 0.6, color_mask, 0.4, 0)
    cv2.imwrite(output_path.replace(".png", "_overlay.png"), overlay)


def parse_args():
    parser = argparse.ArgumentParser(description='SimSeg Single Image Inference')
    parser.add_argument('--cfg', type=str, required=True, help='experiment config file')
    parser.add_argument('--ckpt_path', type=str, required=True, help='checkpoint path')
    parser.add_argument('--image_path', type=str, required=True, help='input image path')
    parser.add_argument('--output_path', type=str, default='mask.png', help='output mask file')
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    update_cfg(task_cfg_init_fn, args.cfg, [], preprocess_fn=update_clip_config)
    ENV.cfg = cfg
    ENV.cfg_snapshot = deepcopy(cfg)
    ENV.local_rank = args.local_rank

    init_device(cfg)

    model = build_from_cfg(cfg.model.name, cfg, PIPELINE).to(ENV.device)
    model = torch_DDP(model, device_ids=[ENV.local_rank], output_device=ENV.local_rank, find_unused_parameters=False)

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model_checkpoint = checkpoint['state_dict']

    if 'image_encoder.model.model.pos_embed' in model_checkpoint:
        pos_embed_reshaped = interpolate_pos_embed(model_checkpoint['image_encoder.model.model.pos_embed'],
                                                   model.module.image_encoder.model.model)
        model_checkpoint['image_encoder.model.model.pos_embed'] = pos_embed_reshaped
        logger.info('Interpolate PE succeeded.')

    model.load_state_dict(get_dist_state_dict(model_checkpoint), strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder.tag)

    with open(f'data/label_category/{cfg.data.valid_name[0]}.txt', 'r') as f:
        categories = f.readlines()
    seg_categories = [label.strip() for label in categories]

    top_cls_num = 30 if cfg.data.valid_name[0] == 'pascal_context' else 10

    run_single_image(args.image_path, model, cfg, tokenizer, seg_categories, top_cls_num, args.output_path)


if __name__ == "__main__":
    main()
