from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import sys
def main(img_file, model_loc):
    model = load_model(model_loc)
    test_X = ((np.array(Image.open(img_file))-127.5)/127.5).reshape(1,50,700,1)
    pred_y = float(model.predict(test_X)[0])
    print('Alert!!!!') if pred_y < 0.5 else print('No Problem')
if __name__=='__main__':
    img_file = sys.argv[1]
    model_loc = sys.argv[2]
    main(img_file, model_loc)
