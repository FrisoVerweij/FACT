from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl


class GenerateCallbackDigit(pl.Callback):
    '''
    Creates a plot based around a digit
    '''

    def __init__(self, to_sample_from, n_samples=5, every_n_epochs=5, save_to_disk=False, ):
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
            samples = pl_module.sample(self.to_sample_from[i].unsqueeze(0))
            grid = make_grid(samples, nrow=7)
            name = 'samples_{}_{}'.format(i, epoch)
            logger = trainer.logger.experiment
            logger.add_image('sample_{}'.format(i), grid, epoch)

            if self.save_to_disk:
                save_image(grid, trainer.logger.log_dir + '\\' + name + "_sample.png")


class GenerateCallbackLatent(pl.Callback):
    '''
    Creates a plot based around the latent space.
    '''

    def __init__(self, to_sample_from, n_samples=8, latent_dimensions=8, every_n_epochs=5, save_to_disk=False, ):
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
            samples = pl_module.sample(self.to_sample_from[i].unsqueeze(0))
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
            save_image(grid, trainer.logger.log_dir + '\\' + name + "_sample.png")
