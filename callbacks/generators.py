import torch
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from sys import platform

class GenerateCallbackDigit(pl.Callback):
    '''
    Creates a plot based around a digit
    '''

    def __init__(self, to_sample_from, dataset, n_samples=5, every_n_epochs=5, save_to_disk=False, border_size=5, show_prob=True):
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
        self.border_size = 10 if show_prob else border_size
        self.show_prob = show_prob
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

        # Now we actually loop over our latents
        for i in range(self.n_samples):
            samples, y, y_prob = pl_module.sample(self.to_sample_from[i].unsqueeze(0))
            samples = add_border_to_samples(samples, y, border_size=self.border_size, to_rgb=self.to_rgb)

            if self.show_prob:
                for j in range(len(samples)):
                    to_pil = transforms.Compose([transforms.ToPILImage()])
                    to_tens = transforms.Compose([transforms.ToTensor()])

                    sample = to_pil(samples[j])

                    d = ImageDraw.Draw(sample)


                    if platform == "linux" or platform == "linux2":
                        font = ImageFont.load_default()
                    elif platform == "win32":
                        font = ImageFont.truetype("arial.ttf", size=9)
                    else:
                        font = ImageFont.truetype("arial.ttf", size=9)

                    d.text((0, 0), str(round(float(y_prob[j][0]), 3)), fill=(255, 255, 255), font=font)

                    samples[j] = to_tens(sample)

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
                 border_size=5, show_prob=True):
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
        self.border_size = 10 if show_prob else border_size
        self.show_prob = show_prob
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

        #  This is to simply show the images as they are
        latent, mu, logvar = pl_module.encoder(self.to_sample_from.to('cuda' if torch.cuda.is_available() else 'cpu'))
        imgs = pl_module.decoder(latent)
        imgs = make_grid(imgs)
        save_image(imgs, trainer.logger.log_dir + "/" + "example_images.png")

        orgiginal_imgs = make_grid(self.to_sample_from)
        save_image(orgiginal_imgs, trainer.logger.log_dir + "/" + "original_images.png")
      
        sweep_length = pl_module.sweep_length
        results = create_samples(self.to_sample_from, pl_module, border_size=self.border_size, to_rgb=self.to_rgb,
                                 show_probs=self.show_prob)
        grids = create_latent_grids(results, self.latent_dimensions, nrow=sweep_length)

        ### Loop over the latent dimensions
        for i, grid in enumerate(grids):

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
        border_tensor = create_border(sample, label, border_size=border_size)

        ### Add border tensor to
        r = combine_border_and_sample(sample, border_tensor, border_size=border_size)

        result.append(r)

    return result


### To add colors, add a tensor with the right rgb colors.
COLORLIST = [
    torch.tensor([0.121, 0.466, 0.705]),
    torch.tensor([1.000, 0.498, 0.054]),
    torch.tensor([0.172, 0.627, 0.172]),
    torch.tensor([0.839, 0.152, 0.156]),
    torch.tensor([0.580, 0.403, 0.741]),
    torch.tensor([0.549, 0.337, 0.294]),
    torch.tensor([0.890, 0.466, 0.760]),
    torch.tensor([0.498, 0.498, 0.498]),
    torch.tensor([0.737, 0.741, 0.133]),
    torch.tensor([0.090, 0.745, 0.811]),
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


def create_samples(to_sample_from, model, to_rgb=True, border_size=5, show_probs=True):
    results = []
    for i in range(len(to_sample_from)):
        samples, y, y_prob = model.sample(to_sample_from[i].unsqueeze(0))
        samples = add_border_to_samples(samples, y, to_rgb=to_rgb, border_size=border_size, )

        if show_probs:
            for j in range(len(samples)):
                to_pil = transforms.Compose([transforms.ToPILImage()])
                to_tens = transforms.Compose([transforms.ToTensor()])

                sample = to_pil(samples[j].cpu())

                d = ImageDraw.Draw(sample)

                if platform == "linux" or platform == "linux2":
                    font = ImageFont.load_default()
                elif platform == "win32":
                    font = ImageFont.truetype("arial.ttf", size=9)
                else:
                    font = ImageFont.truetype("arial.ttf", size=9)

                d.text((0, 0), str(round(float(y_prob[j][0]), 3)), fill=(255, 255, 255), font=font)

                samples[j] = to_tens(sample)

        results.append(samples)
    return results


def create_latent_grids(results, latent_dimensions, nrow=7):
    grids = []
    ### Loop over the latent dimensions
    for i in range(latent_dimensions):

        latent_dim_samples = []
        start_index = 7 * i
        end_index = start_index + 7

        for samples in results:
            latent_dim_samples += samples[start_index: end_index]

        grid = make_grid(latent_dim_samples, nrow=nrow)
        grids.append(grid)
    return grids