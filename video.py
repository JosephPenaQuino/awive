from utils import ImageLoader
import cv2 as cv

# image_dataset = ('/home/joseph/Documents/Thesis/Dataset/TowardHarmonization/'
#                  'dataset/CastorRiver/20190410/Video2/Frames')
image_dataset = ('/home/joseph/Documents/Thesis/Dataset/TowardHarmonization/'
                 'dataset/CastorRiver/20190709/Video4/Frames')


image_len =  811
image_offset = 1
resize_ratio = 5

def play():
    image_loader = ImageLoader(image_dataset, '', 5, image_offset)
    for i in range (image_offset, image_len):
        im = image_loader.load(i)
        new_shape = (im.shape[0] // resize_ratio, im.shape[1] // resize_ratio)
        im = cv.resize(im, new_shape)
        cv.imshow('Video', im)
        if cv.waitKey(1) & 0xFF == ord('q'):
            print ('Finished by key \'q\'')
            break
    cv.destroyAllWindows()


def main():
    play()


if __name__ == '__main__':
    main()
