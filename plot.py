import numpy as np
import glob
import os
import cv2
from PIL import Image
from rich.progress import track
import argparse

import matplotlib
matplotlib.use('Agg')
from matplotlib import cm

from pixood import PixOOD

def decode_segmap(label_mask, normalize=False):
    n_classes = 19
    label_colours = np.array([ [128, 64, 128], [244, 35, 232], [70, 70, 70], 
        [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], 
        [220, 220, 0], [107, 142, 35], [152, 251, 152], [0, 130, 180], 
        [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], 
        [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    if normalize:
        rgb = rgb / 255.0
    return rgb.astype(np.uint8)

def plot(params, evaluator):
    out_dir = os.path.join(params["out_dir"], params["dname"])
    out_dname = params["dname"].replace(" ", "_")
    alpha = params["blend_alpha"]
    os.makedirs(out_dir, exist_ok=True)

    img_list = sorted(glob.glob(os.path.join(params["img_dir"], "*.png")))
    if len(img_list) == 0:
        img_list = sorted(glob.glob(os.path.join(params["img_dir"], "*.jpg")))

    cmap = cm.get_cmap("turbo")
    
    length = len(img_list)
    for i in track(range(0, length), description="Processing images..."):
        filename = os.path.basename(img_list[i])[:-4] + ".png"
        
        if os.path.isfile(out_dir + "/" + filename[:-4] + ".jpg"):
            continue

        pil_img = Image.open(img_list[i])
        out = evaluator.evaluate(pil_img, return_anomaly_score=False)

        anomaly_score = (1.0-out.pred_score_all[:, :, evaluator.eval_labels].max(dim=-1)[0].cpu().numpy())
        anomaly_thr = (anomaly_score >= params["thr"])[:, :, None]
        anomaly_color = 255*cmap(anomaly_score)[:,:,:3]

        img = np.array(pil_img).astype(float)
        road_color = np.ones_like(img, dtype=float)*np.array([[[128, 64, 128]]])

        valid_road_mask = np.zeros_like(anomaly_thr).astype(float)

        blend_img = (img*(1-valid_road_mask)*(~anomaly_thr).astype(float) +
                    alpha*img*valid_road_mask + (1-alpha)*road_color*valid_road_mask + 
                    alpha*img*anomaly_thr.astype(float) + (1-alpha)*anomaly_color*anomaly_thr.astype(float))

        size_4x = (int(img.shape[1]/5.0), int(img.shape[0]/5.0)) 
        img_res = cv2.resize(img, size_4x, interpolation=cv2.INTER_LINEAR)
        segm_res = cv2.resize(decode_segmap(out.pred_y.cpu().numpy()), size_4x, interpolation=cv2.INTER_NEAREST)
        blend_img[0:size_4x[1], 0:size_4x[0], :] = img_res 
        blend_img[0:size_4x[1], size_4x[0]:2*size_4x[0], :] = segm_res 

        if blend_img.shape[0] != params["output_image_sz"][0] or blend_img.shape[1] != params["output_image_sz"][1]: 
            blend_img = cv2.resize(blend_img, (params["output_image_sz"][1], params["output_image_sz"][0]), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(out_dir + "/" + filename[:-4] + ".jpg", blend_img.astype(np.uint8)[:,:,::-1])

    os.system("ffmpeg -f image2 -r {} -pattern_type glob -i '{}/*.jpg' -vcodec libx264 -profile:v high444 -refs 16 -crf 23 -preset ultrafast {}/{}.mp4".format(
        params["fps"], out_dir, params["out_dir"], out_dname))


if __name__ == "__main__":
    params = {
        "blend_alpha": 0.3,
        "output_image_sz": [1080, 1920],
        "evaluator_dir": "./",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--out_dir', type=str, default="./_out/vis/")
    parser.add_argument('--dname', type=str, default="test")
    parser.add_argument('--fps', type=int, default=10,  help="fps of output video")
    parser.add_argument('--thr', type=float, default=0.995)
    args = parser.parse_args()

    params["img_dir"] = args.img_dir
    params["out_dir"] = args.out_dir 
    params["dname"] = args.dname
    params["fps"] = args.fps
    params["thr"] = args.thr

    model_eval_kwargs = { 
              "exp_dir": params["evaluator_dir"], 
              "eval_labels": np.arange(19).tolist(), 
              "eval_scale_factor": 5, 
    }
    evaluator = PixOOD(**model_eval_kwargs)

    print("NAME:    ", params["dname"])
    print("IN DIR:  ", params["img_dir"])
    print("OUT DIR: ", params["out_dir"])
    print(params)
    plot(params, evaluator)
