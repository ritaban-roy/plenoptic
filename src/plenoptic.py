from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.signal import correlate2d
import gc
from tqdm import tqdm

def gamma_correction(x):
    x = np.where(x <= 0.0404482, x/12.92, ((x+0.055)/1.055)**2.4)
    return x

def load_lightfield(img_path):
    lightfield_img = io.imread(img_path)/255.0
    height, width, channels = lightfield_img.shape

    patch_size = 16
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size

    lightfield = np.empty((num_patches_height, num_patches_width, patch_size, patch_size, channels))

    for i in range(num_patches_height):
        for j in range(num_patches_width):
            patch = lightfield_img[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :]
            lightfield[i, j] = patch
    lightfield = np.transpose(lightfield, (2,3,0,1,4))
    return lightfield

def create_mosaic():
    lightfield = load_lightfield('../data/chessboard_lightfield.png')
    img_h = lightfield.shape[0]*lightfield.shape[2]
    img_w = lightfield.shape[1]*lightfield.shape[3]
    mosaic = np.zeros((img_h, img_w, 3))
    for i in range(16):
        for j in range(16):
            mosaic[i*lightfield.shape[2] : (i+1)*lightfield.shape[2], j*lightfield.shape[3]:(j+1)*lightfield.shape[3], :] = lightfield[i, j, :, :, :].squeeze()
    mosaic = np.clip(mosaic, a_min=0, a_max=1)
    io.imsave(f'../data/mosaic.png', (mosaic*255).astype(np.uint8))


def refocus(lightfield, depth):
    lensletSize = 16
    maxUV = (lensletSize - 1) / 2
    us = np.arange(lensletSize) - maxUV
    vs = np.arange(lensletSize) - maxUV
    
    center = np.array([maxUV, maxUV])
    
    width = lightfield.shape[3]
    height = lightfield.shape[2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    stack_image = np.zeros((lightfield.shape[2], lightfield.shape[3], 3))
    for u in us:
        for v in vs:
            u_int = int(center[0] + u)
            v_int = int(center[1] - v)
            z = lightfield[u_int, v_int, :, :, :].squeeze()
            points = (np.arange(z.shape[0]), np.arange(z.shape[1]))
            new_x = x + depth*v
            new_y = y + depth*u
            new_x = np.clip(new_x, 0, width - 1)
            new_y = np.clip(new_y, 0, height - 1)
            new_points = (new_y, new_x)
            interpolated_image = interpn(points, z, new_points, bounds_error=False)
            stack_image += interpolated_image
    stack_image = stack_image/(us.shape[0]**2)
    stack_image = np.clip(stack_image, a_min=0, a_max=1)
    io.imsave(f'../data/stack_image_d{depth}.png', (stack_image*255).astype(np.uint8))
    return stack_image

def depth_from_focus():
    s1 = 1
    s2 = 3
    focal_stack = np.load('focal_stack_2.npz')['arr_0']
    I_lum = gamma_correction(focal_stack.copy())
    I_lum = (I_lum[:, :, 0, :]*0.2126 + I_lum[:, :, 1, :]*0.7152 + I_lum[:, :, 2, :]*0.0722).squeeze()
    I_lf = gaussian_filter(I_lum, sigma=s1)
    I_hf = I_lum - I_lf
    w_sharp = gaussian_filter(np.square(I_hf), sigma=s2)
    
    w_sharp_exp = np.expand_dims(w_sharp, axis=2)
    I_allfocus = np.sum(w_sharp_exp * focal_stack, axis=-1)/np.sum(w_sharp, axis=-1, keepdims=True)
    I_allfocus = np.clip(I_allfocus, a_min=0, a_max=1)
    io.imsave(f'../data/allfocus_image_s1_{s1}_s2_{s2}.png', (I_allfocus*255).astype(np.uint8))
    depths = np.expand_dims(np.arange(-0.4, 1.8, 0.2), axis=(0,1))
    #print(w_sharp.shape, depths.shape)
    
    I_depth = np.sum(w_sharp*depths.squeeze(), axis=-1)/np.sum(w_sharp, axis=-1)
    #print(np.min(I_depth), np.max(I_depth))
    I_depth = np.clip((I_depth-np.min(I_depth))/np.max(I_depth), a_min=0, a_max=1)
    io.imsave(f'../data/allfocus_depth_s1_{s1}_s2_{s2}.png', (I_depth*255).astype(np.uint8))
    #print(I_allfocus.shape, I_depth.shape)
    

def create_focal_stack():
    lightfield = load_lightfield('../data/chessboard_lightfield.png')
    depths = np.arange(-0.4, 1.8, 0.2)
    stack_images = []
    for depth in depths:
        stack_images.append(refocus(lightfield, depth))
    focal_stack = np.stack(stack_images, axis=-1)
    np.savez_compressed("focal_stack_2.npz", focal_stack)
    
def refocus_aperture(lightfield, aperture, depths):
    lensletSize = 16
    maxUV = (lensletSize - 1) / 2
    i = lensletSize - aperture*2
    us = np.arange(aperture*2, i) - maxUV
    vs = np.arange(aperture*2, i) - maxUV
    
    center = np.array([maxUV, maxUV])
    
    width = lightfield.shape[3]
    height = lightfield.shape[2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    stack_images = []
    for depth in depths:
        stack_image = np.zeros((lightfield.shape[2], lightfield.shape[3], 3))
        for u in us:
            for v in vs:
                u_int = int(center[0] + u)
                v_int = int(center[1] - v)
                z = lightfield[u_int, v_int, :, :, :].squeeze()
                points = (np.arange(z.shape[0]), np.arange(z.shape[1]))
                #new_points = (points[0]+depth*u, points[1]+depth*v, points[2])
                new_x = x + depth*v
                new_y = y + depth*u

                # Ensure the new coordinates stay within the image boundaries
                new_x = np.clip(new_x, 0, width - 1)
                new_y = np.clip(new_y, 0, height - 1)
                # Define the coordinates for interpolation as a tuple (new_x, new_y)
                new_points = (new_y, new_x)
                #print(points[0].shape, points[1].shape, z.shape, new_points[0].shape)
                interpolated_image = interpn(points, z, new_points, bounds_error=False)
                #print(interpolated_image.shape, z.shape)
                stack_image += interpolated_image
        stack_image = stack_image/(us.shape[0]**2)
        stack_images.append(np.clip(stack_image, a_min=0, a_max=1))
    #io.imsave(f'../data/stack_image_d{depth}.png', (stack_image*255).astype(np.uint8))
    focal_stack = np.stack(stack_images, axis=-1)
    return focal_stack

def create_confocal_stack():
    lightfield = load_lightfield('../data/chessboard_lightfield.png')
    apertures = np.arange(0, 3.5, 0.5)
    depths = np.arange(-0.4, 1.8, 0.2)
    confocal_stack_images = []
    for aperture in apertures:
        confocal_stack_images.append(refocus_aperture(lightfield, aperture, depths))
    confocal_stack = np.stack(confocal_stack_images, axis=-1)
    #print(confocal_stack.shape)
    np.savez_compressed("confocal_stack_2.npz", confocal_stack)

def confocal_stereo():
    confocal_stack = np.load('confocal_stack_2.npz')['arr_0']
    I_lum = gamma_correction(confocal_stack)
    I_lum = (I_lum[:, :, 0, :, :]*0.2126 + I_lum[:, :, 1, :, :]*0.7152 + I_lum[:, :, 2, :, :]*0.0722).squeeze()
    #16, 14, 12, 10, 8, 6, 4
    #apertures = np.square(16 - 4*np.arange(0, 3.5, 0.5))
    #print(apertures)
    #AFI = np.swapaxes(AFI, -1, -2)
    f_star = np.argmin(np.var(I_lum, axis=-1, keepdims=False), axis=-1, keepdims=False)
    f_star = np.clip((f_star-np.min(f_star))/np.max(f_star), a_min=0, a_max=1)
    #f_star = closing(f_star)
    io.imsave(f'../data/confocal_depth.png', (f_star*255).astype(np.uint8))

def select_patch():
    short_vid = np.load('short_video_2_gray.npz')['arr_0']
    mid_frame = short_vid[short_vid.shape[0]//2]
    plt.imshow(mid_frame, cmap='gray')
    points = plt.ginput(n=1, timeout=0, show_clicks=True)
    plt.close()
    np.savez('selected_point_2.npz', points)

def template_matching():
    short_vid = np.load('short_video_2_gray.npz')['arr_0']
    
    patch_size = 80
    mid_frame = short_vid[short_vid.shape[0]//2]
    selected_point = np.load('selected_point_2.npz')['arr_0'][0].astype(np.int32)
    template = mid_frame[selected_point[1]-patch_size//2:selected_point[1]+patch_size//2, selected_point[0]-patch_size//2:selected_point[0]+patch_size//2]
    #template = gamma_correction(template)
    #template = (template[:, :, 0]*0.2126 + template[:, :, 1]*0.7152 + template[:, :, 2]*0.0722).squeeze()
    template_box = uniform_filter(template, 2)
    
    window = 100#2*patch_size
    rows = np.arange(selected_point[1]- window, selected_point[1] + window)
    cols = np.arange(selected_point[0]- window, selected_point[0] + window)
    matched_point = None
    matched_points = []
    plt.imshow(template, cmap='gray')
    plt.show()
    #exit()
    for frame in short_vid:
        max_ncc = -1
        #plt.imshow(frame[selected_point[1]- window : selected_point[1] + window, selected_point[0]- window : selected_point[0] + window], cmap='gray')
        #plt.show()
        for i in rows:
            for j in cols:
                I_t = frame[i-patch_size//2:i+patch_size//2, j-patch_size//2:j+patch_size//2]
                #print(I_t.shape, i, j)
                #I_t = gamma_correction(template)
                #I_t = (I_t[:, :, 0]*0.2126 + I_t[:, :, 1]*0.7152 + I_t[:, :, 2]*0.0722).squeeze()
                I_t_box = uniform_filter(I_t)
                cc = np.sum((I_t - I_t_box) * (template-template_box))
                ncc = cc / np.sqrt((np.sum(np.square(I_t - I_t_box)) * np.sum(np.square(template-template_box))))
                if ncc > max_ncc:
                    matched_point = [i, j]
                    max_ncc = ncc
        matched_points.append(matched_point)
        #plt.imshow(frame[matched_point[0]-patch_size//2:matched_point[0]+patch_size//2, matched_point[1]-patch_size//2:matched_point[1]+patch_size//2], cmap='gray')
        #plt.show()
    matched_points = np.array(matched_points)
    np.savez('matched_points_2.npz', matched_points)

def unstructured_focus():
    short_vid = np.load('video_2.npz')['arr_0']
    selected_point = np.load('selected_point_2.npz')['arr_0'][0].astype(np.int32)
    matched_points = np.load('matched_points_2.npz')['arr_0']
    width = short_vid[0].shape[1]
    height = short_vid[0].shape[0]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    stack_image = np.zeros((height, width, 3))
    for idx, match in tqdm(enumerate(matched_points[:-15])):
        z = short_vid[idx]
        points = (np.arange(z.shape[0]), np.arange(z.shape[1]))
        u = match[0]
        v = match[1]
        #new_points = (points[0]+depth*u, points[1]+depth*v, points[2])
        new_x = x + (v - selected_point[0])
        new_y = y + (u - selected_point[1])

        # Ensure the new coordinates stay within the image boundaries
        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)
        new_points = (new_y, new_x)
        interpolated_image = interpn(points, z, new_points, bounds_error=False)
        stack_image += interpolated_image
    stack_image = stack_image/np.max(stack_image)
    stack_image = np.clip(stack_image, a_min=0, a_max=1)
    io.imsave(f'../data/my_confocal_image_d{selected_point[0]}.png', (stack_image*255).astype(np.uint8))

def confocal_mosaic():
    lightfield = np.load('confocal_stack_2.npz')['arr_0']
    print(lightfield.shape)
    #exit()
    img_h = lightfield.shape[0]*lightfield.shape[-1]
    img_w = lightfield.shape[1]*lightfield.shape[-2]
    mosaic = np.zeros((img_h, img_w, 3))
    for i in range(lightfield.shape[0]):
        for j in range(lightfield.shape[1]):
            AFI = np.transpose(lightfield[i, j, :, :, :].squeeze(), (2,1,0))
            if(i%100 == 0 and j%100 == 0):
                AFI_i_j = np.clip(AFI, a_min=0, a_max=1)
                io.imsave(f'../data/AFI_{i}_{j}.png', (AFI_i_j*255).astype(np.uint8)) 
            mosaic[i*lightfield.shape[-1] : (i+1)*lightfield.shape[-1], j*lightfield.shape[-2]:(j+1)*lightfield.shape[-2], :] = AFI
    mosaic = np.clip(mosaic, a_min=0, a_max=1)
    io.imsave(f'../data/AFI_mosaic.png', (mosaic*255).astype(np.uint8)) 

def run_q3():
   select_patch()
   template_matching()
   unstructured_focus()
    
def run_q1_confocal():
    create_confocal_stack()
    confocal_stereo()
    confocal_mosaic()

def run_q1_focal():
    create_mosaic()
    create_focal_stack()
    depth_from_focus()
    
    


if __name__ == '__main__':
    #run_q1_focal()
    #run_q1_confocal()
    run_q3()
    