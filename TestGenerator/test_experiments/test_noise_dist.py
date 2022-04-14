from tqdm import tqdm
from TestGenerator.test_forward_pass import test_binary_generator, show_random_slices

# Project directory for Generator weights
proj_dir = 'Trained_Generators/Noise_models'

# Noise types to choose from
noises = ["normal", "laplace", "uniform", "cauchy", "exponential"]

# Lower bound of dimension of generated volume
min_xyz = (60, 60, 60)

# If True, run all possible noise combinations
# If False, run one particular combination
run_all_combos = False

# This loop is customized for trying different
if run_all_combos:
    for n1 in tqdm(noises):
        for n2 in noises:
            proj_name = f'{n1}_noise'
            bin_volume = test_binary_generator(proj_dir, proj_name, min_xyz, noise_type="cauchy")
            plot_file_name = f"{n1.upper()}-{n2.upper()}.jpg"
            title = f"{n1.upper()} GAN - {n2.upper()} NOISE"
            show_random_slices(bin_volume, plot_file_name, title)
else:
    proj_name = 'cauchy_noise'
    bin_volume = test_binary_generator(proj_dir, proj_name, min_xyz, noise_type="cauchy")
    print(bin_volume.shape)
    plot_file_name = "test_plot.jpg"
    title = "test_title"
    show_random_slices(bin_volume, plot_file_name, title)