import numpy as np
import cv2


_colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255)
}


def draw_bbox(img, bbox, color='red'):
    if isinstance(color, str):
        color = _colors[color]

    x1, y1, x2, y2 = bbox
    src = (int(x1), int(y1))
    des = (int(x2), int(y2))
    img = cv2.rectangle(img.copy(), src, des, color, 2)
    return img


class ImageDisplay(object):
    def __init__(self, row, col):
        self.num_i = 0
        self.img_size = 250
        self.img_show = np.zeros((self.img_size * row, self.img_size * col, 3), dtype=np.uint8)  # initial a empty image
        self.col = col
        self.row = row
        self.is_show = True

    def show_image(self, input_image):
        if self.is_show:
            cv2.namedWindow("image")

            num_r = self.num_i // self.col
            num_c = self.num_i - num_r * self.col

            input_image = cv2.resize(input_image, (self.img_size, self.img_size))

            if self.num_i < self.row * self.col:
                self.img_show[self.img_size * num_r:self.img_size * (num_r + 1),
                              self.img_size * num_c:self.img_size * (num_c + 1)] = input_image

            self.num_i = self.num_i + 1
            if self.num_i >= self.row * self.col:
                while True:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(self.img_show, 'Please press L to the next sample, and ESC to exit', (10, 30),
                                font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.imshow('image', self.img_show)
                    input_key = cv2.waitKey(0)
                    if input_key == 27:  # ESC key to exit
                        cv2.destroyAllWindows()
                        self.is_show = False
                        break
                    elif input_key == 108:  # l key to the next
                        self.num_i = 0
                        break
                    else:
                        continue