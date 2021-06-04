from random import randint,seed
from PIL import Image,ImageDraw,ImageFont
from os import makedirs
from glob import glob
import numpy as np
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
seed(1234) # 結果を再現できるようにするために乱数のシードを固定
def main():
    system_font_list = fm.findSystemFonts()
    system_font_list = [font.replace('\\','/') for font in system_font_list]
    # 文字化けするfontを除外
    NG_FONT_LIST = ['BSSYM7.TTF','holomdl2.ttf','marlett.ttf','MTEXTRA.TTF','OUTLOOK.TTF','REFSPCL.TTF','segmdl2.ttf','symbol.ttf','TSPECIAL1.TTF','webdings.ttf','wingding.ttf','WINGDNG2.TTF','WINGDNG3.TTF',]
    font_list = []
    for font in system_font_list:
        if font.split('/')[-1] not in NG_FONT_LIST:
            font_list.append(font)
    output_dir = './img/'
    makedirs(output_dir, exist_ok=True)
    seed(1234) # 結果を再現できるようにするために乱数のシードを固定
    pixel_size = (700,50)

    for text in ['Amazon ElastiCache','Amazon ElasticCache']:
        for i in range(5):
            for font_path in font_list:
                font = ImageFont.truetype(font_path, randint(20,30))
                font = ImageFont.truetype(font_path, 30)
                img = Image.new('L', (pixel_size[0],pixel_size[1]),(255))
                d = ImageDraw.Draw(img)
                d.text((0, 0), text, font=font, fill=(0))

                # 描画部分だけ切り出し
                img_array = np.array(img)
                w,h=0,0
                for j in range(pixel_size[0]-1,-1,-1):
                    if not (np.all(img_array[:,j]==255)):
                        w = j+1
                        break
                for j in range(pixel_size[1]-1,-1,-1):
                    if not (np.all(img_array[j,:]==255)):
                        h = j+1
                        break
                img_array = img_array[0:h,0:w]
                max_pad_h = pixel_size[1] - img_array.shape[0]
                max_pad_w = pixel_size[0] - img_array.shape[1]
                pad_h = randint(0,max_pad_h)
                pad_w = randint(0,max_pad_w)
                # 真っ白なキャンバスを作成
                canvas_array = np.ones((pixel_size[1],pixel_size[0]),dtype=np.uint8)*255
                # 文字をランダムに移植
                canvas_array[pad_h:pad_h+img_array.shape[0],pad_w:pad_w+img_array.shape[1]] = img_array
                img = Image.fromarray(canvas_array)

                file_path = output_dir + text.replace(' ','') + '_' +font_path.split('/')[-1] + str(i) +  '.png'
                img.save(file_path)
    img_file_list = sorted(glob('./img/*.png'))
    train_X = np.zeros((len(img_file_list),pixel_size[1],pixel_size[0]),dtype=np.uint8)
    train_y = np.zeros((len(img_file_list)),dtype=np.float32)

    for i,img_file in enumerate(img_file_list):
        img = Image.open(img_file)
        img_array = np.array(img)
        train_X[i,:,:] = img_array
        if 'ElasticCache' in img_file:
            train_y[i] = 1
    train_X = ((train_X-127.5)/127.5).astype(np.float32)
    np.save('./train_X.npy',train_X)
    np.save('./train_y.npy',train_y)

if __name__=='__main__':
    main()