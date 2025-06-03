import odak
import torch
import pycvvdp
import pyfvvdp
            
from torchvision.io import read_image
from odak.learn.perception.learned_perceptual_losses import CVVDP, FVVDP, LPIPS
from odak.learn.perception.image_quality_losses import PSNR, SSIM, MSSSIM

# I have images in /home/vgl/output/multi_color/holoeye_perceptual_normalized/conventional
# the output images are named reconstruction_00.png reconstruction_01.png reconstruction_02.png
# and the ground truth images are named target_00.png target_01.png target_02.png
# I want to evaluate the PSNR, SSIM, MSSSIM, LPIPS, CVVDP, FVVDP between the output and the ground truth images.


# Define the paths to the images
output_image_paths = ["/home/vgl/output/multi_color/holoeye/conventional/reconstruction_00.png",
                 "/home/vgl/output/multi_color/holoeye/conventional/reconstruction_01.png",
                 "/home/vgl/output/multi_color/holoeye/conventional/reconstruction_02.png"]

target_image_paths = ["/home/vgl/output/multi_color/holoeye/conventional/target_00.png",
                    "/home/vgl/output/multi_color/holoeye/conventional/target_01.png",
                    "/home/vgl/output/multi_color/holoeye/conventional/target_02.png"]

# Define the perceptual losses
psnr = PSNR()
ssim = SSIM()
msssim = MSSSIM()
lpips = LPIPS()
# cvvdp = pycvvdp.cvvdp(display_name = 'standard_4k', device = torch.device('cpu'))
# fvvdp = pyfvvdp.fvvdp(display_name = 'standard_4k', heatmap = 'none', device = torch.device('cpu'))
cvvdp = CVVDP()
fvvdp = FVVDP()

output_images = []
target_images = []
# read the images with pytorch
for output_image_path, target_image_path in zip(output_image_paths, target_image_paths):
    img_out = read_image(output_image_path).float()
    img_out = img_out / 255.0
    print(f"{type(img_out) = }, {img_out.dtype = }, {img_out.shape = }")
    output_images.append(img_out)
    img_target = read_image(target_image_path).float()
    img_target = img_target / 255.0
    target_images.append(img_target)
    print(f"{type(img_target) = }, {img_target.dtype = }, {img_target.shape = }")
    print("min: ", img_out.min(), img_target.min())
    print("max: ", img_out.max(), img_target.max())

print("Output images: ", output_images[0].shape)
print("Target images: ", target_images[0].shape)


# Evaluate the perceptual losses
for output_image, target_image in zip(output_images, target_images):

    psnr_l = psnr(output_image, target_image)
    ssim_l = ssim(output_image, target_image)
    msssim_l = msssim(output_image, target_image)

    # convert the images to the range [-1, 1] for lpips
    output_image_lpips = 2 * output_image - 1
    target_image_lpips = 2 * target_image - 1
    lpips_l = lpips(output_image_lpips, target_image_lpips)

    # do these for cvvdp and fvvdp
    # prediction = output_image.unsqueeze(0)
    # target = target_image.unsqueeze(0)
    # cvvdp_l = cvvdp.predict(prediction, target, dim_order = 'BCHW')[0]

    # prediction = output_image.unsqueeze(0)
    # target = target_image.unsqueeze(0)
    # fvvdp_l = fvvdp.predict(prediction, target, dim_order = 'BCHW')[0]

    cvvdp_l = cvvdp(output_image, target_image)
    fvvdp_l = fvvdp(output_image, target_image)


    print("PSNR: ", psnr_l)
    print("SSIM: ", ssim_l)
    print("MSSSIM: ", msssim_l)
    print("LPIPS: ", lpips_l)
    print("CVVDP: ", cvvdp_l)
    print("FVVDP: ", fvvdp_l)
    print("-------------------") 

