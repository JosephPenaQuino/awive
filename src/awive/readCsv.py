"""Read csv."""
import pandas as pd
import matplotlib.cm
import cv2 as cv
from awive.utils import ImageLoader

path = ('/home/joseph/Documents/Thesis/Dataset/TowardHarmonization/dataset/'
        'CastorRiver/20190410/Reference/ref5.csv')

image_dataset = ('/home/joseph/Documents/Thesis/Dataset/TowardHarmonization/'
                 'dataset/CastorRiver/20190410/Video3/Frames')
image_len =  565
image_offset = 51
resize_ratio = 5

def get_df():
    df = pd.read_csv(path)
    df = df.drop(['d'], axis=1)
    df = df[df['v'].notna()]
    df['v'] = df['v'].str.replace(',', '.').astype(float)
    df['v'] = pd.to_numeric(df['v'])
    return df


def main():
    df = get_df()
    image_loader = ImageLoader(image_dataset, '', 5, image_offset)
    im = image_loader.load(0)
    im = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    cmap = matplotlib.cm.get_cmap('Spectral')
    minVal = df['v'].min()
    maxVal = df['v'].max()
    rangeVal = maxVal - minVal

    for i, val in df.iterrows():
        v =(val['v']-minVal)/rangeVal
        start_point = (int(val['x']), int(val['y']))
        maxI = 400
        end_point = (int(val['x'] - maxI * v), int(val['y'] - maxI * v))
        rgb = cmap(v)
        color = (255*rgb[0], 255*rgb[1], 255*rgb[2])
        thickness = 20
        im = cv.arrowedLine(im, start_point, end_point, color, thickness)


    width = im.shape[0]
    height = im.shape[1]
    # M = cv.getRotationMatrix2D((width//2, height//2), 44, 1.0)
    M = cv.getRotationMatrix2D((width//2, height//2), 44+180, 1.0)
    im = cv.warpAffine(im, M, (width, height))
    im = im[:2880, 2268:4447]

    new_shape = (width // resize_ratio, height // resize_ratio)
    im = cv.resize(im, new_shape)
    cv.imshow('main', im)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
