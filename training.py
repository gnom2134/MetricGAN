import librosa
from scipy.io.wavfile import write
from torch.utils.data import SubsetRandomSampler, DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch

from models import Generator, Discriminator, DiscriminatorLoss, GeneratorLoss
from dataset import TIMITDataset, MixedDataset, TRAIN_DATA_FOLDER, TEST_DATA_FOLDER, NOISE_DATA_FOLDER
from preprocessing import prepare_signal, back_to_wav
from metrics import calculate_pesq, calculate_stoi


DEVICE = torch.device("cuda:0")

# TODO: fix connection between generator and discriminator


def create_loaders(dataset, batch_size=1, create_val=True):
    if create_val:
        val_split = int(np.floor(0.2 * len(dataset)))
        indices = list(range(len(dataset)))
        np.random.seed(42)
        np.random.shuffle(indices)
        val_indices, train_indices = indices[:val_split], indices[val_split:]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        return train_loader, val_loader
    else:
        return DataLoader(dataset, batch_size=batch_size)


def calculate_metric(model, data_loader, metric):
    pass


def train_metric_gan(learning_iterations=200, batch_size=1, lr=1e-3, d_epochs=10, g_epochs=10):
    metric_history = []
    generator = Generator(DEVICE)
    discriminator = Discriminator(DEVICE)

    generator_dataset = TIMITDataset(TRAIN_DATA_FOLDER, NOISE_DATA_FOLDER)

    train_loader, val_loader = create_loaders(generator_dataset, batch_size)

    discriminator_criterion = DiscriminatorLoss(DEVICE)
    discriminator_optimizer = optim.Adam(discriminator.network.parameters(), lr=lr, betas=(0.9, 0.999))
    discriminator_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(discriminator_optimizer, 10, 2)

    generator_criterion = GeneratorLoss(DEVICE, s=1.)
    generator_optimizer = optim.Adam(
        [{"params": generator.lstm.parameters()}, {"params": generator.fc.parameters()}],
        lr=lr,
        betas=(.9, .999)
    )
    generator_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(generator_optimizer, 10, 2)

    for _ in tqdm(range(learning_iterations), desc='GAN learning iteration'):
        generated_data = []

        with torch.no_grad():
            generator.eval()
            for x, y in tqdm(train_loader, desc='Generating data...', leave=False):
                x, phase, sl = prepare_signal(x)
                x = x.to(DEVICE)
                x = x.reshape((1, -1, 257))
                generated_data.append((back_to_wav(x * np.maximum(0.05, generator(x).cpu().numpy()), phase, sl), y))

        d_dataset = MixedDataset(generated_data, TRAIN_DATA_FOLDER, NOISE_DATA_FOLDER)
        d_train_loader, d_val_loader = create_loaders(d_dataset, batch_size=batch_size)

        generator.train(False)
        discriminator.train(True)
        p_bar = tqdm(range(d_epochs), desc="Training discriminator...", leave=False)
        for epoch in p_bar:
            for x, y in tqdm(d_train_loader, desc="I'm just doing my thing...", leave=False):
                x, phase, sl = prepare_signal(x)
                y, y_phase, y_sl = prepare_signal(y)
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                x = x.reshape((1, -1, 257))
                y = y.reshape((1, -1, 257))

                try:
                    metric = calculate_stoi(back_to_wav(x, phase, sl), back_to_wav(y, y_phase, y_sl))
                except Exception as err:
                    print(err)
                    metric = 0.

                discriminator_optimizer.zero_grad()
                res = discriminator(x, y)
                loss = discriminator_criterion(res, metric)
                loss.backward()
                discriminator_optimizer.step()

                print(f"Metric: {metric}, discriminator error: {np.abs(res.detach().cpu().numpy() - metric)}")

            discriminator_scheduler.step()
            p_bar.set_description(desc=f"Training discriminator... Epoch: {epoch}")

        avg_metric = 0
        generator.train(True)
        discriminator.train(False)
        p_bar = tqdm(range(g_epochs), desc="Training generator...", leave=False)
        for epoch in p_bar:
            for x, y in tqdm(train_loader, desc="I'm just doing my thing...", leave=False):
                x, phase, sl = prepare_signal(x)
                y, y_phase, y_sl = prepare_signal(y)
                x = x.reshape((1, -1, 257))
                y = y.reshape((1, -1, 257))

                try:
                    metric = calculate_stoi(back_to_wav(x, phase, sl), back_to_wav(y, y_phase, y_sl))
                except Exception as err:
                    print(err)
                    metric = 0.

                generator_optimizer.zero_grad()
                res = discriminator(x, y)
                loss = generator_criterion(res, metric)
                loss.backward()
                generator_optimizer.step()
                avg_metric = res.detach().cpu().numpy()

                print(f"Metric: {metric}")

            generator_scheduler.step()
            p_bar.set_description(desc=f"Training generator... Epoch: {epoch}, avg_metric: {avg_metric}")

        metric_history.append(avg_metric)

    return generator


if __name__ == "__main__":
    batch_size = 1
    trained_generator = train_metric_gan(batch_size=batch_size)

    test_dataset = TIMITDataset(TEST_DATA_FOLDER, NOISE_DATA_FOLDER)
    test_loader = create_loaders(test_dataset, batch_size=batch_size, create_val=False)
