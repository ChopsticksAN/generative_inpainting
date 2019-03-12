import argparse
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')
    
drawing = False # true if mouse is pressed
ix,iy = -1,-1
color = (255,255,255)
size = 6

def erase_img(img) :
    def erase_rect(event,x,y,flags,param):
        global ix,iy,drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            if drawing == True:
                # cv2.circle(img,(x,y),10,(255,255,255),-1)
                cv2.rectangle(img,(x-size,y-size),(x+size,y+size),color,-1)
                cv2.rectangle(mask,(x-size,y-size),(x+size,y+size),color,-1)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                # cv2.circle(img,(x,y),10,(255,255,255),-1)
                cv2.rectangle(img,(x-size,y-size),(x+size,y+size),color,-1)
                cv2.rectangle(mask,(x-size,y-size),(x+size,y+size),color,-1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # cv2.circle(img,(x,y),10,(255,255,255),-1)
            cv2.rectangle(img,(x-size,y-size),(x+size,y+size),color,-1)
            cv2.rectangle(mask,(x-size,y-size),(x+size,y+size),color,-1)


    mask = np.zeros(img.shape)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',erase_rect)
    #cv2.namedWindow('mask')
    #cv2.setMouseCallback('mask',erase_rect)
    

    while(1):
        #img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        #print(k)
        if k == 13:
            break

    #test_img = cv2.resize(img, (args.input_height, args.input_width))/127.5 - 1
    #test_mask = cv2.resize(mask, (args.input_height, args.input_width))/255.0
    #fill mask region to 1
    #test_img = (test_img * (1-test_mask)) + test_mask
    cv2.destroyAllWindows()
    return img, mask

def showAndWrite(result) :
    cv2.imwrite(args.output, result[0][:, :, ::-1])
    origin = cv2.imread(args.image)
    origin = cv2.resize(origin, (image.shape[2], image.shape[1]))
    hitch = np.hstack((origin, image[0], result[0][:,:,::-1]))
    cv2.imshow("results", hitch)
    cv2.waitKey(0)

if __name__ == "__main__":
    ng.get_gpus(1)
    args = parser.parse_args()

    model = InpaintCAModel()
    ori_image = cv2.imread(args.image)
    ori_image = cv2.resize(ori_image, (256, ori_image.shape[0] * 256 // ori_image.shape[1]))
    image, mask = erase_img(ori_image)
    assert image.shape == mask.shape


    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        showAndWrite(result)

