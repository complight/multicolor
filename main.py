import sys
import argparse
import torch
import odak
from odak.learn.wave import multi_color_hologram_optimizer, multiplane_loss, propagator


__title__ = 'Multi-color Holograms'


def main():
    settings_filename = './settings/jasper.txt'
    parser = argparse.ArgumentParser(description = __title__)
    parser.add_argument(
                        '--settings',
                        type = argparse.FileType('r'),
                        help = 'Filename for the settings file. Default is {}'.format(settings_filename)
                       )
    args = parser.parse_args()
    if type(args.settings) != type(None):
        settings_filename = str(args.settings.name)
    process(settings_fn = settings_filename)


def compansate_illumination(settings, target_image, device):
    illimuniation_form = odak.learn.tools.load_image(
                                                     settings["beam"]["beam profile"],
                                                     normalizeby = 2 ** settings["target"]["color depth"],
                                                     torch_style = True
                                                    ).to(device)
    illimuniation_form_max = torch.amax(illimuniation_form, dim = (1, 2)).view(3, 1, 1)
    compensation = illimuniation_form / illimuniation_form_max
    target_image_com = target_image / compensation
    target_image_com = target_image_com / torch.amax(target_image_com, dim = (1, 2)).view(3, 1, 1)
    target_image = target_image_com * torch.amax(target_image, dim = (1, 2)).view(3, 1, 1)
    return target_image


def process(settings_fn):
    settings = odak.tools.load_dictionary(settings_fn)
    device = torch.device(settings['general']['device'])
    resolution = settings['spatial light modulator']['resolution']
    target_image = odak.learn.tools.load_image(
                                               settings["target"]["image filename"],
                                               normalizeby = 2 ** settings["target"]["color depth"],
                                               torch_style = True
                                              ).to(device)[0:3, 0:resolution[0], 0:resolution[1]]
    target_depth = odak.learn.tools.load_image(
                                               settings["target"]["depth filename"],
                                               normalizeby = 2 ** settings["target"]["color depth"],
                                               torch_style = True
                                              ).to(device)
    if len(target_depth.shape) > 2:
        target_depth = torch.mean(target_depth, dim = 0)
    target_depth = target_depth[0:resolution[0], 0:resolution[1]]
    if settings["beam"]["beam profile"] != '':
        target_image = compansate_illumination(settings, target_image, device)
    loss_function = multiplane_loss(
                                    target_image = target_image,
                                    target_depth = target_depth,
                                    target_blur_size = settings["target"]["defocus blur size"],
                                    number_of_planes = settings["target"]["number of depth layers"],
                                    blur_ratio = settings["target"]["blur ratio"],
                                    weights = settings["target"]["weights"],
                                    scheme = settings["target"]["scheme"],
                                    reduction = settings['general']['reduction'],
                                    device = device
                                   )
    targets, focus_target, depth = loss_function.get_targets()
    propagator_mc = propagator(
                               wavelengths = settings['beam']['wavelengths'],
                               pixel_pitch = settings['spatial light modulator']['pixel pitch'],
                               resolution = settings['spatial light modulator']['resolution'],
                               aperture_size = settings['beam']['pinhole size'],
                               number_of_frames = settings['target']['number of frames'],
                               number_of_depth_layers = settings['target']['number of depth layers'],
                               volume_depth = settings['target']['volume depth'],
                               image_location_offset = settings['target']['location offset'],
                               propagation_type = settings['beam']['propagation type'],
                               propagator_type = settings['beam']['propagator type'],
                               method = settings['general']['method'],
                               device = device
                              )
    mcho = multi_color_hologram_optimizer(
                                          wavelengths = settings["beam"]["wavelengths"], 
                                          resolution = settings["spatial light modulator"]["resolution"], 
                                          targets = targets,
                                          propagator = propagator_mc,
                                          number_of_frames = settings["target"]["number of frames"],
                                          number_of_depth_layers = settings['target']['number of depth layers'],
                                          learning_rate = settings["general"]["learning rate"], 
                                          learning_rate_floor = settings["general"]["learning rate floor"],
                                          double_phase = settings["general"]["double phase constrain"],
                                          method = settings["general"]["method"],
                                          channel_power_filename = settings["target"]["channel power filename"],
                                          device = device,
                                          loss_function = loss_function,
                                          peak_amplitude = settings["target"]["peak amplitude"],
                                          optimize_peak_amplitude = settings["target"]["optimize peak amplitude"],
                                          img_loss_thres = settings["target"]["img loss threshold"],
                                          reduction = settings['general']['reduction'],
                                        )
    hologram_phases, frame_reconstructions, laser_powers, channel_powers, peak_amplitude = mcho.optimize(
                                                                                                         number_of_iterations = settings["general"]["iterations"],
                                                                                                         weights = settings["general"]["loss weights"]
                                                                                                        )
    settings['target']['peak amplitude'] = peak_amplitude
    save(
         settings, 
         device,
         hologram_phases,
         laser_powers,
         channel_powers,
         frame_reconstructions,
         targets,
         target_image,
         target_depth,
         depth,
         settings['target']['peak amplitude'],
         settings['target']['color depth']
        )


def save(settings, device, hologram_phases, laser_powers, channel_powers, frame_reconstructions, targets, target_image, target_depth, depth, intensity_scale, color_depth):
    output_folder = settings["general"]["output directory"]
    directory = output_folder + settings["general"]["method"]
    odak.tools.check_directory(directory)
    odak.tools.save_dictionary(settings, '{}/settings.txt'.format(directory))
    checker_complex = odak.learn.wave.linear_grating(
                                                     settings["spatial light modulator"]["resolution"][0],
                                                     settings["spatial light modulator"]["resolution"][1],
                                                     add = odak.pi,
                                                     axis = 'y'
                                                    ).to(device)
    checker = odak.learn.wave.calculate_phase(checker_complex)
    for depth_id in range(targets.shape[0]):
        odak.learn.tools.save_image(
                                    "{}/target_{:02d}.png".format(directory, depth_id), targets[depth_id] * intensity_scale, 
                                    cmin = 0., 
                                    cmax = intensity_scale,
                                    color_depth = color_depth
                                   )
        odak.learn.tools.save_image(
                                    "{}/reconstruction_{:02d}.png".format(directory, depth_id), 
                                    torch.sum(frame_reconstructions[:, depth_id], dim = 0), 
                                    cmin = 0., 
                                    cmax = intensity_scale, 
                                    color_depth = color_depth
                                   ) 
    hologram_phases_w_grating = torch.zeros_like(hologram_phases)
    for frame_id in range(hologram_phases.shape[0]):
        phase = hologram_phases[frame_id]
        phase_normalized = phase % (2 * odak.pi)
        for depth_id in range(targets.shape[0]):
            odak.learn.tools.save_image(
                                        "{}/reconstruction_frame_{:02d}_depth_{:03d}.png".format(directory, frame_id, depth_id), 
                                        frame_reconstructions[frame_id, depth_id], 
                                        cmin = 0., 
                                        cmax = intensity_scale, 
                                        color_depth = color_depth
                                       )
        odak.learn.tools.save_image(
                                    "{}/phase_{:02d}.png".format(directory, frame_id), 
                                    phase_normalized, 
                                    cmin = 0., 
                                    cmax = odak.pi * 2
                                   )
        phase_grating = phase + checker
        phase_grating_normalized = phase_grating % (2 * odak.pi)
        hologram_phases_w_grating[frame_id] = phase_grating_normalized
        odak.learn.tools.save_image(
                                    "{}/phase_grated_{:02d}.png".format(directory, frame_id), 
                                    phase_grating_normalized, 
                                    cmin = 0., 
                                    cmax = odak.pi * 2
                                   )
    if hologram_phases.shape[0] == 3:
        odak.learn.tools.save_image(
                                    '{}/phase_combined.png'.format(directory), 
                                    hologram_phases % (2 * odak.pi), 
                                    cmin = 0., 
                                    cmax = odak.pi * 2.
                                   )
        odak.learn.tools.save_image(
                                    '{}/phase_combined_w_grating.png'.format(directory), 
                                    hologram_phases_w_grating, 
                                    cmin = 0., 
                                    cmax = odak.pi * 2.
                                   )
    odak.learn.tools.save_torch_tensor('{}/laser_powers.pt'.format(directory), laser_powers)
    odak.learn.tools.save_torch_tensor('{}/channel_powers.pt'.format(directory), channel_powers)
    data = {
            "targets" : targets,
            "target" : target_image,
            "target depth" : target_depth,
            "depth" : depth,
            "intensity scale" : intensity_scale,
            "laser powers" : laser_powers,
            "channel powers" : channel_powers,
            "hologram phases" : hologram_phases,
            "settings" : settings
           }
    odak.learn.tools.save_torch_tensor('{}/data.pt'.format(directory), data)
    print('Output stored at {}. Check `odak.log` for more information.'.format(directory))


if __name__ == "__main__":
    sys.exit(main())
