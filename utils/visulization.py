import colorsys
import numpy as np
from PIL import ImageFont, ImageDraw
from config import FONT_DIR,CATEGORY_DICTIONARY, CATEGORY_NUM


def drawBox(image, classes, scores, boxes, need_score=False):
    hsv_tuples = [(x / CATEGORY_NUM, 1., 1.)
                  for x in range(CATEGORY_NUM)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    font = ImageFont.truetype(font=FONT_DIR,
                              size=int(np.floor(3e-2 * image.size[1] + 0.5)))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(classes))):
        predicted_class = CATEGORY_DICTIONARY[c]
        box = boxes[i]
        if need_score:
            score = scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
        else:
            label = '{}'.format(predicted_class)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for j in range(thickness):
            draw.rectangle(
                [left + j, top + j, right - j, bottom - j],  # reversed: box[0],box[1],box[2],box[3] -> box[1],box[0],box[3],box[2]
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    return image
