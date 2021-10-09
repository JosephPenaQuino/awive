from utils import ImageLoader
import cv2 as cv

# image_dataset = ('/home/joseph/Documents/Thesis/Dataset/TowardHarmonization/'
#                  'dataset/CastorRiver/20190410/Video2/Frames')
# image_dataset = ('/home/joseph/Documents/Thesis/Dataset/TowardHarmonization/'
                 # 'dataset/CastorRiver/20190709/Video4/Frames')

image_dataset = ('/home/joseph/Documents/Thesis/Dataset/TowardHarmonization/'
                 'dataset/CastorRiver/20190410/Video3/Frames')

# image_len =  811
# image_offset = 1
# resize_ratio = 5

image_len =  565
image_offset = 51
resize_ratio = 5
# images =  []


def play():
    image_loader = ImageLoader(image_dataset, '', 5, image_offset)
    im = image_loader.load(0)
    width = im.shape[0]
    height = im.shape[1]
    M = cv.getRotationMatrix2D((width//2, height//2), 44+180, 1.0)

    for i in range (image_offset, image_len):
        im = image_loader.load(i)
        if im is None:
            print(f'frame {i:04} is none')
            break
        im = cv.warpAffine(im, M, (width, height))
        im = im[:2880, 2268:4448]
        new_shape = (im.shape[0] // resize_ratio, im.shape[1] // resize_ratio)
        im = cv.resize(im, new_shape)
        # print(im.shape)
        # images.append(im)
        cv.imwrite(f'images/im_{i-image_offset:04}.bmp', im)
        cv.imshow('Video', im)
        if cv.waitKey(1) & 0xFF == ord('q'):
            print ('Finished by key \'q\'')
            break
    cv.destroyAllWindows()

    # fourcc = cv.VideoWriter_fourcc(*'h263')
    # fourcc = cv.VideoWriter_fourcc(*'DIVX')
    # fourcc = cv.VideoWriter_fourcc('M','J','P','G')
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # out = cv.VideoWriter('project.avi',
    #                      fourcc,
    #                      20,
    #                      im.shape)
    # for i in range(len(images)):
    #     print(f'writing frame {i:04}')
        # images[i] = cv.cvtColor(images[i], cv.COLOR_GRAY2BGR)
        # out.write(images[i])
        # cv.imwrite(f'images/im_{i:04}.bmp', images[i])
    # out.release()


def main():
    play()


if __name__ == '__main__':
    main()
