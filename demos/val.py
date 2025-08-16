import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import signal, ndimage
from net.DL import TriComponentProcessor
from scipy.signal import butter, lfilter_zi, lfilter
import h5py

nperseg = 20
noverlap = 10
nfft = 20


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TriComponentProcessor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


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


def process_3component(data):
    processed = []
    Zxx_list = []
    for sample in data:
        features = []
        for ch in range(3):
            dn = butter_bandpass_filter_zi(sample[ch], 1, 45, 100)
            _, _, Zxx = signal.stft(dn, fs=100,
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    nfft=nfft)
            Zxx_list.append(Zxx)
            if np.max(np.abs(Zxx)) > 1:
                max_val = np.abs(Zxx).max()
                features.extend([
                    np.real(Zxx) / max_val,
                    np.imag(Zxx) / max_val
                ])

        processed.append(np.stack(features))
    return processed, Zxx_list


def denoise(model, raw_data):
    device = next(model.parameters()).device

    stft_data, Zxx_list = process_3component(raw_data[np.newaxis])  # Add sample dimension
    input_tensor = torch.from_numpy(np.array(stft_data)).float().to(device)

    with torch.no_grad():
        output, _ = model(input_tensor)
    output = output.cpu().numpy()[0]

    mask = create_adaptive_mask(output, Zxx_list)
    denoised_signals = apply_mask_and_istft(Zxx_list, mask)
    return denoised_signals


def create_adaptive_mask(model_output, Zxx_list, yy=1, sizmedian=10):
    combined_energy = np.sum(np.array(model_output), axis=0)

    outAM = combined_energy * np.max(np.abs(np.abs(Zxx_list)))
    outAM = outAM / np.max(np.abs(outAM))

    for iu in range(outAM.shape[1]):
        tmp = np.copy(outAM[:, iu])
        med = np.median(tmp)
        mad = np.median(np.abs(tmp - med))
        threshold = med - 1.5 * mad
        tmp[tmp < threshold] = 1e-10
        outAM[:, iu] = tmp

    for iu in range(outAM.shape[0]):
        tmp = np.copy(outAM[iu, :])
        med = np.median(tmp)
        mad = np.median(np.abs(tmp - med))
        threshold = med - 1.5 * mad
        tmp[tmp < threshold] = 1e-10
        outAM[iu, :] = tmp

    mea = np.mean(outAM)
    outAM[outAM >= yy * mea] = 1
    outAM[outAM < yy * mea] = 1e-10

    outAM = ndimage.median_filter(outAM, size=sizmedian)

    return outAM


def apply_mask_and_istft(Zxx_list, masks):
    """Apply mask and inverse STFT, with denormalization"""
    denoised = []
    for ch in range(3):
        dn_mask = np.real(Zxx_list[ch]) * masks + 1j * np.imag(Zxx_list[ch]) * masks
        _, reconstructed = signal.istft(dn_mask, fs=100, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        denoised.append(reconstructed)
    return np.array(denoised)


def calculate_snr(component_data, tp_idx, ts_idx, window_size=100):
    """
    Calculate SNR for each component
    Args:
        component_data: 2D array of shape (3, N) containing data for three components
        tp_idx: P-wave arrival time index
        ts_idx: S-wave arrival time index
        window_size: size of the noise window before P-wave and signal window after S-wave
    """
    if component_data.ndim != 2 or component_data.shape[0] != 3:
        raise ValueError("component_data must be a 2D array of shape (3, N) containing three components of data")

    snr_list = []
    for i in range(3):
        comp_data = component_data[i]

        noise_window = comp_data[:tp_idx]
        signal_window = comp_data[tp_idx:ts_idx + window_size]

        signal_energy = np.sum(signal_window ** 2)
        noise_energy = np.sum(noise_window ** 2)

        snr_db = np.log10(signal_energy / noise_energy) if noise_energy != 0 else np.inf
        snr_db = round(snr_db, 2)

        snr_list.append(snr_db)

    return snr_list


# --------------------------
# Validation Main Function
# --------------------------
def validate_and_save_snr(model_sizes, val_data_path, save_dir="snr_results"):
    """
    Perform validation using a validation set with P-wave and S-wave arrival times
    and save SNR results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load validation data
    val_data = np.load(val_data_path, allow_pickle=True).item()
    print(f"Successfully loaded validation data with {len(val_data['waveforms'])} samples")

    # Extract waveforms and arrival times
    waveforms = val_data['waveforms']  # (n_samples, 6000, 3)
    p_arrivals = val_data['p_arrivals']  # (n_samples,)
    s_arrivals = val_data['s_arrivals']  # (n_samples,)
    sample_ids = val_data.get('ids', [f"sample_{i}" for i in range(len(waveforms))])

    n_val = len(waveforms)
    print(f"Using validation set data with {n_val} samples")

    # Store SNR results for all models
    all_snr_results = {}

    for size in model_sizes:
        print(f"\n===== Processing model: ../models/model_{size}.pth =====")
        model_path = f"../models/model_{size}.pth"
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found, skipping")
            continue

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TriComponentProcessor().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Record SNR results for current model
        snr_original = []  # SNR for original noisy data
        snr_denoised = []  # SNR after denoising
        snr_improvement = []  # SNR improvement

        # Store indices of failed samples
        failed_samples = []

        # Batch size (adjust based on available memory)
        batch_size = 100
        num_batches = (n_val + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_val)

            # Progress indicator
            print(f"Processing batch {batch_idx + 1}/{num_batches} ({start_idx}-{end_idx - 1}/{n_val})")

            for i in range(start_idx, end_idx):
                try:
                    # Get waveform and arrival times
                    waveform = waveforms[i]
                    tp_idx = int(p_arrivals[i])
                    ts_idx = int(s_arrivals[i])
                    sample_id = sample_ids[i] if i < len(sample_ids) else f"sample_{i}"

                    # Validate arrival times
                    if tp_idx <= 0 or ts_idx <= tp_idx or ts_idx >= waveform.shape[0]:
                        print(
                            f"Sample {i} ({sample_id}) has invalid arrival times (tp={tp_idx}, ts={ts_idx}), skipping")
                        failed_samples.append(i)
                        continue

                    # Transpose to (3, 6000)
                    noisy_data = waveform.transpose((1, 0))

                    # Denoise data
                    denoised_data = denoise(model, noisy_data)  # Shape (3, 6000)

                    # Calculate SNR metrics
                    snr_raw = calculate_snr(noisy_data, tp_idx, ts_idx)
                    snr_den = calculate_snr(denoised_data, tp_idx, ts_idx)
                    snr_imp = [den - raw for den, raw in zip(snr_den, snr_raw)]

                    # Store results
                    snr_original.append(snr_raw)
                    snr_denoised.append(snr_den)
                    snr_improvement.append(snr_imp)

                except Exception as e:
                    print(f"Failed to process sample {i} ({sample_id}): {str(e)}, skipping")
                    failed_samples.append(i)
                    continue

        # Save results for current model
        model_results = {
            "original_snr": np.array(snr_original),
            "denoised_snr": np.array(snr_denoised),
            "snr_improvement": np.array(snr_improvement),
            "avg_original": np.mean(snr_original, axis=0) if snr_original else np.array([0, 0, 0]),
            "avg_denoised": np.mean(snr_denoised, axis=0) if snr_denoised else np.array([0, 0, 0]),
            "avg_improvement": np.mean(snr_improvement, axis=0) if snr_improvement else np.array([0, 0, 0]),
            "failed_samples": failed_samples
        }

        # Add to results dictionary
        all_snr_results[size] = model_results

        # Save model-specific results
        np.save(os.path.join(save_dir, f"snr_model_{size}.npy"), model_results)
        print(f"SNR results for model {size} saved to {save_dir}/snr_model_{size}.npy")

        # Print average results
        print(f"Average SNR improvement for model {size}:")
        print(f"Z-component: {model_results['avg_improvement'][0]:.2f} dB")
        print(f"N-component: {model_results['avg_improvement'][1]:.2f} dB")
        print(f"E-component: {model_results['avg_improvement'][2]:.2f} dB")
        print(f"Failed samples: {len(failed_samples)}/{n_val}\n")


# --------------------------
# Execute Validation
# --------------------------
if __name__ == "__main__":
    # Configuration parameters
    MODEL_SIZES = [10000]  # Training set sizes
    VAL_DATA_PATH = "../datasets/validation_data.npy"  # Validation data path
    SAVE_DIR = "snr_results_with_ps"  # Results save directory

    # Execute validation
    validate_and_save_snr(
        model_sizes=MODEL_SIZES,
        val_data_path=VAL_DATA_PATH,
        save_dir=SAVE_DIR
    )