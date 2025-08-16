import numpy as np
import torch
import h5py
import os
import time
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter_zi, lfilter
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from net.DL import TriComponentProcessor
import random

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Configure Chinese font support
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # Fix negative sign display


# --------------------------
# Signal Processing and Denoising Functions
# --------------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter_zi(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y, _ = lfilter(b, a, data, zi=zi * data[0])
    return y


def process_3componentN(data):
    processed = []
    for sample in data:
        component_features = []
        for ch in range(3):
            dn = butter_bandpass_filter_zi(sample[ch], 1, 45, 100)
            _, _, Zxx = signal.stft(dn, fs=100,
                                    nperseg=20,
                                    noverlap=10,
                                    nfft=20)
            if np.max(np.abs(Zxx)) > 1:
                real = np.real(Zxx) / np.max(np.abs(Zxx))
                imag = np.imag(Zxx) / np.max(np.abs(Zxx))
            component_features.extend([real, imag])

        stft_cube = np.stack(component_features, axis=0)
        processed.append(stft_cube)

    return np.array(processed)


def process_3componentC(data):
    processed = []
    for sample in data:
        component_features = []
        for ch in range(3):
            dn = butter_bandpass_filter_zi(sample[ch], 1, 45, 100)
            _, _, Zxx = signal.stft(dn, fs=100,
                                    nperseg=20,
                                    noverlap=10,
                                    nfft=20)

            if np.max(np.abs(Zxx)) > 1:
                amp = np.abs(Zxx) / np.max(np.abs(Zxx))
            component_features.extend([amp])

        stft_cube = np.stack(component_features, axis=0)
        processed.append(stft_cube)

    return np.array(processed)


def TriComponent_MaskLoss(y_true, y_pred):
    total_loss = 0.0
    for c in range(3):
        true_c = y_true[:, c, :, :]
        pred_c = y_pred[:, c, :, :]
        noise = true_c - pred_c
        ratio = torch.abs(noise) / (torch.abs(pred_c) + 1e-8)
        mask = 1 / (1 + ratio)
        component_loss = torch.mean(torch.abs(1 - mask))
        total_loss += component_loss
    return total_loss / 3


class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = np.Inf if mode == 'min' else -np.Inf

    def __call__(self, epoch, model, current):
        if self.monitor == 'val_loss':
            if self.mode == 'min' and current < self.best:
                self.best = current
                if self.save_best_only:
                    print(f'Saving model to {self.filepath}')
                    torch.save(model.state_dict(), self.filepath)
            elif self.mode == 'max' and current > self.best:
                self.best = current
                if self.save_best_only:
                    print(f'Saving model to {self.filepath}')
                    torch.save(model.state_dict(), self.filepath)


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy_batch, clean_batch in val_loader:
            noisy_batch, clean_batch = noisy_batch.to(device), clean_batch.to(device)
            outputs, _ = model(noisy_batch)
            loss = criterion(clean_batch, outputs)
            val_loss += loss.item()
    return val_loss / len(val_loader)


# --------------------------
# Noise Injection Functions
# --------------------------
def add_noise_with_scaling(seismic_data, noise_data, scaling_factor_range=(0.1, 1)):
    """Add noise scaled by random proportion of seismic data amplitude"""
    noisy_data = np.zeros_like(seismic_data)

    for i in range(seismic_data.shape[0]):
        for j in range(seismic_data.shape[2]):
            S = seismic_data[i, :, j]
            N = noise_data[i, :, j]

            rms_S = np.sqrt(np.mean(S ** 2))
            rms_N = np.sqrt(np.mean(N ** 2))

            scale_factor = np.random.uniform(scaling_factor_range[0], scaling_factor_range[1])

            if rms_N == 0:
                scale = 0.0
            else:
                scale = (scale_factor * rms_S) / rms_N

            noisy_data[i, :, j] = S + scale * N

    return seismic_data, noisy_data


# --------------------------
# Data Loading and Processing Functions
# --------------------------
def load_noise_data(h5fname, npyfname, num_noise=10000, exclude_ids=None):
    """Load noise data with exclusion capability for specific IDs"""
    print(f"Loading noise data: {h5fname}")
    f = h5py.File(h5fname, 'r')
    allid = np.load(npyfname)
    exclude_ids = exclude_ids if exclude_ids is not None else []

    # Filter noise IDs while excluding specified IDs
    noiseid = [ii for ii in allid if ii.split("_")[-1] == 'NO' and ii not in exclude_ids]
    print(f"Found {len(noiseid)} noise samples (after excluding specified IDs)")

    if len(noiseid) < num_noise:
        print(f"Warning: Available noise samples ({len(noiseid)}) are fewer than requested {num_noise}")
        num_noise = len(noiseid)

    # Random selection of noise
    random.shuffle(noiseid)
    selected_noise_ids = noiseid[:num_noise]

    # Load noise data
    noise_data = []
    for i, idx in enumerate(selected_noise_ids):
        if i % 1000 == 0:
            print(f"Loading noise {i + 1}/{len(selected_noise_ids)}")
        dataset = f.get(idx)
        if dataset is None:
            continue
        data = np.array(dataset['data'])   # Shape: (6000, 3)
        noise_data.append(data)

    f.close()
    return np.array(noise_data), selected_noise_ids


def load_event_data(h5fname, npyfname, num_event=10000, exclude_ids=None):
    """Load event data with exclusion capability for specific IDs"""
    print(f"Loading event data: {h5fname}")
    f = h5py.File(h5fname, 'r')
    allid = np.load(npyfname)
    exclude_ids = exclude_ids if exclude_ids is not None else []

    # Filter event IDs while excluding specified IDs
    eventids = [ii for ii in allid if ii.split("_")[-1] == 'EV' and ii not in exclude_ids]
    print(f"Found {len(eventids)} event samples (after excluding specified IDs)")

    if len(eventids) < num_event:
        print(f"Warning: Available event samples ({len(eventids)}) are fewer than requested {num_event}")
        num_event = len(eventids)

    # Random selection of events
    random.seed(42)
    random.shuffle(eventids)
    selected_event_ids = eventids[:num_event]

    # Load event data
    event_data = []
    for i, idx in enumerate(selected_event_ids):
        if i % 1000 == 0:
            print(f"Loading event {i + 1}/{len(selected_event_ids)}")
        dataset = f.get(idx)
        if dataset is None:
            continue
        data = np.array(dataset['data'])  # Shape: (6000, 3)
        event_data.append(data)

    f.close()
    return np.array(event_data), selected_event_ids

def create_validation_data(h5_path, npy_path, val_size=2000):
    """Create validation dataset: Extract P/S wave arrival times from event attributes"""
    print(f"Creating validation data (size: {val_size})...")
    h5_file = h5py.File(h5_path, 'r')
    all_ids = np.load(npy_path)

    # Filter event IDs
    event_ids = [eid for eid in all_ids if eid.split("_")[-1] == 'EV']
    random.shuffle(event_ids)  # Random shuffle

    val_waveforms = []
    val_p_arrivals = []
    val_s_arrivals = []
    val_ids = []

    for eid in event_ids:
        # Get event group
        event_group = h5_file.get(eid)
        if event_group is None:
            continue

        # Get P and S wave arrival times from attributes
        if 'p_arrival_sample' in event_group.attrs and 's_arrival_sample' in event_group.attrs:
            p_time = event_group.attrs['p_arrival_sample']
            s_time = event_group.attrs['s_arrival_sample']

            # Validate times
            if p_time > 0 and s_time > p_time:  # Ensure P-wave precedes S-wave
                # Get waveform data
                waveform = np.array(event_group['data'])  # Shape: (6000, 3)
                val_waveforms.append(waveform)
                val_p_arrivals.append(p_time)
                val_s_arrivals.append(s_time)
                val_ids.append(eid)

                # Stop when reaching target size
                if len(val_waveforms) >= val_size:
                    break

    h5_file.close()

    # Check results
    if len(val_waveforms) < val_size:
        print(f"Warning: Only found {len(val_waveforms)} events with valid P/S wave info")
        val_size = len(val_waveforms)

    if val_size == 0:
        print("Error: No valid event data found")
        return None, None, None, []

    print(f"Validation dataset created: {val_size} samples")
    return (np.array(val_waveforms),
            np.array(val_p_arrivals),
            np.array(val_s_arrivals),
            val_ids)


# --------------------------
# Model Training Function
# --------------------------
def train_model(train_clean, train_noisy, model_path, log_path):
    """Model training routine"""
    # Process training data
    train_noisy_processed = process_3componentN(train_noisy.transpose((0, 2, 1)))
    train_clean_processed = process_3componentC(train_clean.transpose((0, 2, 1)))

    train_noisy_tensor = torch.FloatTensor(train_noisy_processed).permute(0, 1, 2, 3)
    train_clean_tensor = torch.FloatTensor(train_clean_processed).permute(0, 1, 2, 3)

    # Create DataLoader
    train_dataset = TensorDataset(train_noisy_tensor, train_clean_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Initialize model and training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TriComponentProcessor().to(device)
    criterion = TriComponent_MaskLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    epochs = 50

    # Early stopping
    es = EarlyStopping(patience=15, verbose=True, path=model_path)

    # Record start time
    start_time = time.time()
    run_losses = []
    val_losses = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for noisy_batch, clean_batch in train_loader:
            noisy_batch, clean_batch = noisy_batch.to(device), clean_batch.to(device)
            outputs, _ = model(noisy_batch)
            loss = criterion(clean_batch, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        run_losses.append(avg_train_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}')

        # Early stopping check
        es(avg_train_loss, model)

    # Calculate training time
    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')

    # Save training log
    with open(log_path, 'w') as f:
        f.write(f'Training Time: {training_time:.2f} seconds\n')
        f.write('Epoch,Train Loss,Val Loss\n')
        for e, (t, v) in enumerate(zip(run_losses, val_losses)):
            f.write(f'{e + 1},{t:.6f},{v:.6f}\n')

    return training_time


# --------------------------
# Main Function
# --------------------------
if __name__ == '__main__':
    # File paths
    h5fname = "E:/download/TXED_20231111.h5"
    npyfname = "E:/download/ID_20231111.npy"

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)

    # Step 1: Create validation dataset (with P/S wave info, excluded from training)
    val_waveforms, val_p_arrivals, val_s_arrivals, val_ids = create_validation_data(
        h5fname, npyfname, val_size=2000
    )

    if val_waveforms is None:
        print("Failed to create validation set. Exiting program.")
        exit()

    # Save validation data (to exclude from training)
    val_data = {
        'waveforms': val_waveforms,
        'p_arrivals': val_p_arrivals,
        's_arrivals': val_s_arrivals,
        'ids': val_ids
    }
    np.save('../datasets/validation_data.npy', val_data)
    print(f"Validation data saved to datasets/validation_data.npy ({len(val_waveforms)} samples)")

    # Define different training set sizes
    train_sizes = [10000]
    training_times = {}

    # Train models for each size
    for size in train_sizes:
        print(f"\n===== Training model with {size} samples =====")

        # Load data (excluding validation IDs)
        noise, _ = load_noise_data(h5fname, npyfname, size, exclude_ids=val_ids)
        data, _ = load_event_data(h5fname, npyfname, size, exclude_ids=val_ids)

        if len(data) < size or len(noise) < size:
            print(f"Warning: Insufficient data ({len(data)} events, {len(noise)} noise). Skipping size {size}")
            continue

        # Create training data with noise
        train_clean, train_noisy = add_noise_with_scaling(data, noise)

        # Configure paths
        model_path = f'../models/model_{size}.pth'
        log_path = f'../logs/training_{size}.log'

        # Train model
        time_taken = train_model(train_clean, train_noisy, model_path, log_path)
        training_times[size] = time_taken

        print(f"{size}-sample model training complete. Time: {time_taken:.2f} seconds")

    # Save training time summary
    with open('../logs/training_times_summary.txt', 'w') as f:
        f.write("Training Size,Training Time (seconds)\n")
        for size, time_taken in training_times.items():
            f.write(f"{size},{time_taken:.2f}\n")