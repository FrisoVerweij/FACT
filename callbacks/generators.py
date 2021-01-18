import torch
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl


class GenerateCallbackDigit(pl.Callback):
    '''
    Creates a plot based around a digit
    '''

    def __init__(self, to_sample_from, dataset, n_samples=5, every_n_epochs=5, save_to_disk=False, border_size=5):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.to_sample_from = to_sample_from
        self.n_samples = n_samples
        self.save_to_disk = save_to_disk
        self.border_size = border_size
        self.to_rgb = False if dataset == 'cifar10' else True

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.sample_and_save(trainer, pl_module, trainer.current_epoch + 1)

    def sample_and_save(self, trainer, pl_module, epoch):
        """
        Function that generates and save samples from the VAE.
        The generated samples and mean images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
            epoch - The epoch number to use for TensorBoard logging and saving of the files.
        """
        for i in range(self.n_samples):
            samples, y = pl_module.sample(self.to_sample_from[i].unsqueeze(0))

            samples = add_border_to_samples(samples, y, border_size=self.border_size, to_rgb=self.to_rgb)

            grid = make_grid(samples, nrow=7)
            name = 'samples_{}_{}'.format(i, epoch)
            logger = trainer.logger.experiment
            logger.add_image('sample_{}'.format(i), grid, epoch)

            if self.save_to_disk:
                save_image(grid, trainer.logger.log_dir + "/" + name + "_sample.png")


class GenerateCallbackLatent(pl.Callback):
    '''
    Creates a plot based around the latent space.
    '''

    def __init__(self, to_sample_from, dataset, n_samples=8, latent_dimensions=8, every_n_epochs=5, save_to_disk=False,
                 border_size=5):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.to_sample_from = to_sample_from
        self.n_samples = n_samples
        self.latent_dimensions = latent_dimensions
        self.save_to_disk = save_to_disk
        self.border_size = border_size
        self.to_rgb = False if dataset == 'cifar10' else True

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.sample_and_save(trainer, pl_module, trainer.current_epoch + 1)

    def sample_and_save(self, trainer, pl_module, epoch):
        """
        Function that generates and save samples from the VAE.
        The generated samples and mean images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
            epoch - The epoch number to use for TensorBoard logging and saving of the files.
        """
        results = []
        for i in range(self.n_samples):
            samples, y = pl_module.sample(self.to_sample_from[i].unsqueeze(0))
            samples = add_border_to_samples(samples, y, border_size=self.border_size, to_rgb=self.to_rgb)
            results.append(samples)

        ### Loop over the latent dimensions
        for i in range(self.latent_dimensions):

            latent_dim_samples = []
            start_index = 7 * i
            end_index = start_index + 7

            for samples in results:
                latent_dim_samples += samples[start_index: end_index]

            grid = make_grid(latent_dim_samples, nrow=7)
            name = 'latent_samples_{}_{}'.format(i, epoch)
            logger = trainer.logger.experiment
            logger.add_image('latent_sample_{}'.format(i), grid, epoch)

        if self.save_to_disk:
            save_image(grid, trainer.logger.log_dir + "/" + name + "_sample.png")


def grey_to_rgb(sample):
    return sample.repeat(3, 1, 1)


def add_border_to_samples(samples, labels, to_rgb=True, border_size=5):
    if to_rgb:
        samples = [grey_to_rgb(sample) for sample in samples]

    result = []
    for sample, label in zip(samples, labels):
        ### Create the border tensor
        border_tensor = create_border(sample, label, )

        ### Add border tensor to
        r = combine_border_and_sample(sample, border_tensor, border_size=border_size)

        result.append(r)

    return result


### To add colors, add a tensor with the right rgb colors.
COLORLIST = [
    torch.tensor([0.298, 0.447, 0.690]),
    torch.tensor([0.866, 0.517, 0.321]),
    torch.tensor([0.333, 0.658, 0.407]),
    torch.tensor([0.768, 0.305, 0.321]),
    torch.tensor([0.505, 0.447, 0.701]),
    torch.tensor([0.576, 0.470, 0.376]),
    torch.tensor([0.854, 0.545, 0.764]),
    torch.tensor([0.549, 0.549, 0.549]),
    torch.tensor([0.800, 0.725, 0.454]),
    torch.tensor([0.392, 0.709, 0.803]),
]


def create_border(sample, label, border_size=5, colors=COLORLIST):
    border_tensor = torch.zeros(sample.shape[0], sample.shape[1] + border_size * 2,
                                sample.shape[2] + border_size * 2).to(sample.device)
    index = label
    color = colors[index]

    ### Make sure the color is of the right shape to copy it.
    color = color.reshape(3, -1).repeat(1, border_tensor.shape[2])

    for i in range(border_size):
        border_tensor[:, i, :] = color
        border_tensor[:, -i, :] = color
        border_tensor[:, :, i] = color
        border_tensor[:, :, -i] = color
    return border_tensor


def combine_border_and_sample(sample, border, border_size=5):
    result = torch.zeros(border.shape).to(border.device)
    result += border
    result[:, border_size:-border_size, border_size: -border_size] = sample
    return result
