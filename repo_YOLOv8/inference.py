import cv2
from utils import util
from utils.dataset import resize
import numpy as np
import torch
import os

def detect(image_path, input_size=640, augment=False):
    image, shape = load_image(image_path)
    h, w = image.shape[: 2]
    image, ratio, pad = resize(image, input_size, augment)
    shapes = shape, ((h / shape[1], w / shape[1]), pad)

    samples = image.transpose((2, 0, 1))[::-1]
    samples = np.ascontiguousarray(samples)
    samples = torch.from_numpy(samples)

    model = torch.load('./weights/best.pt', map_location='cuda')['model'].float()
    model.half()
    model.eval()

    samples = samples.cuda()
    samples = samples.half()
    samples = samples / 255
    samples = samples.unsqueeze(0)
    _, _, height, width = samples.shape

    outputs = model(samples)
    outputs = util.non_max_suppression(outputs, 0.5, 0.5)

    outputs = outputs[0]
    detections = outputs.clone()
    util.scale(detections[:, :4], samples.shape[2:], shapes[0], None)
    detections = detections.cpu().numpy()
    return detections


def draw_box(image_path, detections):
    img = cv2.imread(image_path)
    reference = {0: 'pedestrian',
                  1: 'rider',
                  2: 'car',
                  3: 'truck',
                  4: 'bus',
                  5: 'train',
                  6: 'motorcycle',
                  7: 'bicycle',
                  8: 'traffic light',
                  9: 'traffic sign'}
    for top_left_x, top_left_y, bottom_right_x, bottom_right_y, conf, category in detections:
        img = cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), color=(255, 0, 0), thickness=1)
        cv2.putText(img, reference[int(category)], (int(top_left_x), int(top_left_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0 , 0), 2)

    cv2.imwrite("./outputs/{}_nano.jpg".format(image_path.split("/")[-1].split('.')[0]), img)
    # cv2.imshow("result", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def load_image(image_path, input_size=640, augment=False):
    image = cv2.imread(image_path)
    h, w = image.shape[: 2]
    r = input_size / max(h, w)
    if r != 1:
        image = cv2.resize(image,dsize=(int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)

    return image, (h, w)

if __name__ == "__main__":
    dest = "./test_image/"
    for i in os.listdir(dest):
        detections = detect(dest + i)
        draw_box(dest + i, detections)


