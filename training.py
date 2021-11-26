from torch.utils.data import SubsetRandomSampler, DataLoader
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import pickle

from models import Generator, Discriminator, DiscriminatorLoss, GeneratorLoss
from dataset import TIMITDataset, MixedDataset, TRAIN_DATA_FOLDER, TEST_DATA_FOLDER, NOISE_DATA_FOLDER
from preprocessing import prepare_signal, back_to_wav
from metrics import calculate_stoi


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def calculate_metric(model, data_loader, metric, data_loader_name='train'):
    with torch.no_grad():
        model.eval()
        damaged_history = []
        gen_history = []
        for x, y in tqdm(data_loader, desc='Calculating results on ' + data_loader_name, leave=False):
            x_norm, x_phase, x_sl = prepare_signal(x.squeeze(), normalize=True)
            x_non_norm, x_phase_, x_sl_ = prepare_signal(x.squeeze())
            y, y_phase, y_sl = prepare_signal(y)

            x_norm = x_norm.to(DEVICE)
            x_non_norm = x_non_norm.to(DEVICE)
            y = y.to(DEVICE)

            x_norm = x_norm.squeeze().T.unsqueeze(0)
            x_non_norm = x_non_norm.squeeze().T.unsqueeze(0)
            y = y.squeeze().T.unsqueeze(0)

            try:
                metric_value = metric(
                    back_to_wav(x_non_norm.detach().squeeze().T, x_phase_, x_sl_),
                    back_to_wav(y.detach().squeeze().T, y_phase, y_sl)
                )
            except Exception as err:
                print(err)
                metric_value = 0.

            damaged_history.append(metric_value)

            x_gen = x_non_norm * torch.clamp_min(model(x_norm), 0.05)

            try:
                metric_value = metric(
                    back_to_wav(x_gen.detach().squeeze().T, x_phase_, x_sl_),
                    back_to_wav(y.detach().squeeze().T, y_phase, y_sl)
                )
            except Exception as err:
                print(err)
                metric_value = 0.

            gen_history.append(metric_value)

        return np.mean(damaged_history), np.mean(gen_history)


def train_metric_gan(learning_iterations=50, d_lr=1e-3, g_lr=1e-2, d_epochs=1, g_epochs=1, save_generator='generator.pickle'):
    batch_size = 1

    generator = Generator(DEVICE)
    discriminator = Discriminator(DEVICE)

    generator_dataset = TIMITDataset(TRAIN_DATA_FOLDER, NOISE_DATA_FOLDER)

    train_loader, val_loader = create_loaders(generator_dataset, batch_size)

    discriminator_criterion = DiscriminatorLoss(DEVICE)
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=d_lr,
        betas=(0.9, 0.999)
    )

    generator_criterion = GeneratorLoss(DEVICE, s=1.)
    generator_optimizer = optim.Adam(
        generator.parameters(),
        lr=g_lr,
        betas=(.9, .999)
    )

    val_history = []
    train_history = []

    for iteration in tqdm(range(learning_iterations), desc='GAN learning iteration'):
        generated_data = []

        if iteration % 10 == 0:
            val_history.append(calculate_metric(generator, val_loader, calculate_stoi, data_loader_name='val')[1])
            train_history.append(calculate_metric(generator, train_loader, calculate_stoi)[1])

        with torch.no_grad():
            generator.eval()
            for x, y in tqdm(train_loader, desc='Generating data...', leave=False):
                x_norm, x_phase, x_sl = prepare_signal(x.squeeze(), normalize=True)
                x_non_norm, x_phase_, x_sl_ = prepare_signal(x.squeeze())
                x_norm = x_norm.to(DEVICE)
                x_norm = x_norm.squeeze().T.unsqueeze(0)
                x_non_norm = x_non_norm.to(DEVICE)
                x_non_norm = x_non_norm.squeeze().T.unsqueeze(0)

                generated_data.append((back_to_wav((x_non_norm * torch.clamp_min(generator(x_norm), 0.05)).squeeze().T, x_phase_, x_sl_), y))

        d_dataset = MixedDataset(generated_data, TRAIN_DATA_FOLDER, NOISE_DATA_FOLDER)
        d_train_loader = create_loaders(d_dataset, batch_size=batch_size, create_val=False)

        generator.train(True)
        discriminator.train(True)
        p_bar = tqdm(range(d_epochs), desc="Training discriminator...", leave=False)
        for epoch in p_bar:
            p_bar_2 = tqdm(d_train_loader, desc="I'm just doing my thing...", leave=False)
            for (x, y), clear_s in p_bar_2:
                x_non_norm, x_phase_, x_sl_ = prepare_signal(x)
                y, y_phase, y_sl = prepare_signal(y)
                clear_s, s_phase, s_sl = prepare_signal(clear_s)

                x_non_norm = x_non_norm.to(DEVICE)
                y = y.to(DEVICE)
                clear_s = clear_s.to(DEVICE)

                x_non_norm = x_non_norm.squeeze().T.unsqueeze(0)
                y = y.squeeze().T.unsqueeze(0)
                clear_s = clear_s.squeeze().T.unsqueeze(0)

                try:
                    metric = calculate_stoi(
                        back_to_wav(x_non_norm.detach().squeeze().T, x_phase_, x_sl_),
                        back_to_wav(y.detach().squeeze().T, y_phase, y_sl)
                    )
                except Exception as err:
                    print(err)
                    metric = 0.

                discriminator.zero_grad()

                res = discriminator(x_non_norm, y)
                loss = discriminator_criterion(res, metric)
                loss.backward()

                clear_res = discriminator(clear_s, clear_s)
                loss = discriminator_criterion(clear_res, 1.)
                loss.backward()

                discriminator_optimizer.step()

                p_bar_2.set_description(desc=f"Metric: {metric}, discriminator error: {np.abs(res.detach().cpu().numpy() - metric)}, discriminator value: {res.detach().cpu().numpy()}")

            p_bar.set_description(desc=f"Training discriminator... Epoch: {epoch}")

        p_bar = tqdm(range(g_epochs), desc="Training generator...", leave=False)
        for epoch in p_bar:
            metric_history = []
            p_bar_2 = tqdm(train_loader, desc="I'm just doing my thing...", leave=False)
            for x, y in p_bar_2:
                x_norm, x_phase, x_sl = prepare_signal(x, normalize=True)
                x_non_norm, x_phase_, x_sl_ = prepare_signal(x)
                y, y_phase, y_sl = prepare_signal(y)

                x_non_norm = x_non_norm.to(DEVICE)
                x_norm = x_norm.to(DEVICE)
                y = y.to(DEVICE)

                x_norm = x_norm.squeeze().T.unsqueeze(0)
                x_non_norm = x_non_norm.squeeze().T.unsqueeze(0)
                y = y.squeeze().T.unsqueeze(0)

                try:
                    metric_start = calculate_stoi(
                        back_to_wav(x_non_norm.detach().squeeze().T, x_phase_, x_sl_),
                        back_to_wav(y.detach().squeeze().T, y_phase, y_sl)
                    )
                except Exception as err:
                    print(err)
                    metric_start = 0.

                generator.zero_grad()
                discriminator.zero_grad()
                x_gen = x_non_norm * torch.clamp_min(generator(x_norm), 0.05)

                try:
                    metric_gen = calculate_stoi(
                        back_to_wav(x_gen.detach().squeeze().T, x_phase_, x_sl_),
                        back_to_wav(y.detach().squeeze().T, y_phase, y_sl)
                    )
                except Exception as err:
                    print(err)
                    metric_gen = 0.

                res = discriminator(x_gen, y)
                loss = generator_criterion(res)
                loss.backward()
                generator_optimizer.step()

                metric_history.append(metric_gen)
                p_bar_2.set_description(desc=f'Start metric: {metric_start}, Generated metric: {metric_gen}, Metric improvement: {metric_gen - metric_start}')

            p_bar.set_description(desc=f"Training generator... Epoch: {epoch}, avg_metric: {np.mean(metric_history)}")

        with open(save_generator, 'wb') as file:
            pickle.dump(generator, file)

    return generator, train_history, val_history


if __name__ == "__main__":
    trained_generator, train_history, val_history = train_metric_gan()

    plt.plot(val_history, label='val')
    plt.plot(train_history, label='train')
    plt.legend()
    plt.show()

    test_dataset = TIMITDataset(TEST_DATA_FOLDER, NOISE_DATA_FOLDER)
    test_loader = create_loaders(test_dataset, batch_size=1, create_val=False)

    print(
        'Trained generator on test with pesq metric(before, after):',
        calculate_metric(trained_generator, test_loader, calculate_stoi, data_loader_name='test')
    )
