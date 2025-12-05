"""
COMPLETE AUTOMATIC SPEECH RECOGNITION (ASR) SYSTEM
With Noise Reduction, Performance Analysis, and Multiple Recognition Engines

Features:
- Real-time and file-based speech recognition
- Advanced noise reduction algorithms
- Multiple ASR engines (Google, Sphinx, Wit.ai)
- Word Error Rate (WER) calculation
- Audio preprocessing and enhancement
- Comprehensive visualization and analysis
"""

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import scipy.fft
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
import json
import threading
import queue
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core Libraries
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    print("Warning: Audio libraries not available. Install with: pip install sounddevice soundfile")
    AUDIO_AVAILABLE = False

try:
    import speech_recognition as sr
    ASR_AVAILABLE = True
except ImportError:
    print("Warning: Speech recognition not available. Install with: pip install speechrecognition")
    ASR_AVAILABLE = False

try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    print("Warning: Noise reduction not available. Install with: pip install noisereduce")
    NOISE_REDUCE_AVAILABLE = False

try:
    import jiwer
    WER_AVAILABLE = True
except ImportError:
    print("Warning: WER calculation not available. Install with: pip install jiwer")
    WER_AVAILABLE = False

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    print("Warning: Advanced audio processing not available. Install with: pip install librosa")
    LIBROSA_AVAILABLE = False

class AdvancedASRSystem:
    """
    Complete ASR System with advanced features
    """
    
    def __init__(self, sample_rate=44100, channels=1):
        """
        Initialize the ASR system
        
        Args:
            sample_rate (int): Audio sample rate
            channels (int): Number of audio channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        
        # Initialize speech recognizer
        if ASR_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone() if AUDIO_AVAILABLE else None
        
        # Audio processing parameters
        self.frame_length = 2048
        self.hop_length = 512
        
        # Recognition engines
        self.engines = {
            'google': self._recognize_google,
            'sphinx': self._recognize_sphinx,
            'wit': self._recognize_wit,
            'azure': self._recognize_azure,
            'watson': self._recognize_watson
        }
        
        # Results storage
        self.results = {
            'transcriptions': [],
            'wer_scores': [],
            'processing_times': [],
            'confidence_scores': []
        }
        
        print("Advanced ASR System Initialized")
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"Channels: {self.channels}")
        print("="*60)
    
    def install_dependencies(self):
        """Install all required dependencies"""
        dependencies = [
            "numpy", "scipy", "matplotlib", "seaborn", "pandas",
            "sounddevice", "soundfile", "speech-recognition", 
            "noisereduce", "jiwer", "librosa", "pyaudio"
        ]
        
        print("Installing dependencies...")
        for dep in dependencies:
            try:
                __import__(dep.replace("-", "_"))
                print(f"✓ {dep}")
            except ImportError:
                print(f"Installing {dep}...")
                os.system(f"pip install {dep}")
    
    # =========================
    # AUDIO RECORDING FUNCTIONS
    # =========================
    
    def record_audio(self, duration=5, filename=None, real_time_plot=False):
        """
        Record audio from microphone
        
        Args:
            duration (float): Recording duration in seconds
            filename (str): Output filename (optional)
            real_time_plot (bool): Show real-time waveform
            
        Returns:
            numpy.ndarray: Recorded audio data
        """
        if not AUDIO_AVAILABLE:
            print("Audio recording not available. Please install sounddevice.")
            return None
        
        print(f"Recording for {duration} seconds...")
        print("3... 2... 1... 🎤 RECORDING!")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float64'
        )
        
        # Real-time plotting
        if real_time_plot:
            self._plot_realtime_audio(audio_data)
        
        sd.wait()  # Wait for recording to finish
        
        # Save if filename provided
        if filename:
            self.save_audio(audio_data, filename)
        
        print("✓ Recording completed!")
        return audio_data.flatten() if self.channels == 1 else audio_data
    
    def _plot_realtime_audio(self, audio_data):
        """Plot real-time audio waveform"""
        plt.figure(figsize=(12, 4))
        plt.plot(np.linspace(0, len(audio_data)/self.sample_rate, len(audio_data)), audio_data)
        plt.title("Real-time Audio Recording")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
    
    def load_audio(self, filename):
        """
        Load audio file
        
        Args:
            filename (str): Path to audio file
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            if LIBROSA_AVAILABLE:
                audio_data, sr = librosa.load(filename, sr=self.sample_rate)
                return audio_data, sr
            else:
                sr, audio_data = wav.read(filename)
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float64) / 32767.0
                return audio_data, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def save_audio(self, audio_data, filename):
        """Save audio data to file"""
        try:
            if LIBROSA_AVAILABLE:
                sf.write(filename, audio_data, self.sample_rate)
            else:
                # Convert to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wav.write(filename, self.sample_rate, audio_int16)
            print(f"✓ Audio saved: {filename}")
        except Exception as e:
            print(f"Error saving audio: {e}")
    
    # ============================
    # AUDIO PREPROCESSING FUNCTIONS
    # ============================
    
    def preprocess_audio(self, audio_data, apply_filters=True):
        """
        Comprehensive audio preprocessing
        
        Args:
            audio_data (np.array): Input audio signal
            apply_filters (bool): Whether to apply filtering
            
        Returns:
            np.array: Preprocessed audio
        """
        processed_audio = audio_data.copy()
        
        # 1. Normalize audio
        processed_audio = self.normalize_audio(processed_audio)
        
        # 2. Remove DC offset
        processed_audio = processed_audio - np.mean(processed_audio)
        
        # 3. Apply filters if requested
        if apply_filters:
            processed_audio = self.apply_bandpass_filter(processed_audio)
        
        # 4. Trim silence
        processed_audio = self.trim_silence(processed_audio)
        
        return processed_audio
    
    def normalize_audio(self, audio_data):
        """Normalize audio to [-1, 1] range"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    def apply_bandpass_filter(self, audio_data, low_freq=80, high_freq=8000):
        """Apply bandpass filter for speech enhancement"""
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = scipy.signal.butter(4, [low, high], btype='band')
        filtered_audio = scipy.signal.filtfilt(b, a, audio_data)
        return filtered_audio
    
    def trim_silence(self, audio_data, threshold=0.01):
        """Remove silence from beginning and end"""
        # Find non-silent regions
        non_silent = np.where(np.abs(audio_data) > threshold)[0]
        
        if len(non_silent) > 0:
            start_idx = non_silent[0]
            end_idx = non_silent[-1]
            return audio_data[start_idx:end_idx+1]
        return audio_data
    
    # ============================
    # NOISE REDUCTION FUNCTIONS
    # ============================
    
    def reduce_noise(self, audio_data, method='noisereduce', **kwargs):
        """
        Apply noise reduction using various methods
        
        Args:
            audio_data (np.array): Input audio
            method (str): Noise reduction method
            **kwargs: Additional parameters
            
        Returns:
            np.array: Noise-reduced audio
        """
        if method == 'noisereduce' and NOISE_REDUCE_AVAILABLE:
            return self._noisereduce_method(audio_data, **kwargs)
        elif method == 'spectral_subtraction':
            return self._spectral_subtraction(audio_data, **kwargs)
        elif method == 'wiener_filter':
            return self._wiener_filter(audio_data, **kwargs)
        elif method == 'median_filter':
            return self._median_filter(audio_data, **kwargs)
        else:
            print(f"Noise reduction method '{method}' not available")
            return audio_data
    
    def _noisereduce_method(self, audio_data, stationary=False, prop_decrease=0.8):
        """Noise reduction using noisereduce library"""
        return nr.reduce_noise(
            y=audio_data,
            sr=self.sample_rate,
            stationary=stationary,
            prop_decrease=prop_decrease
        )
    
    def _spectral_subtraction(self, audio_data, alpha=2.0, beta=0.01):
        """Spectral subtraction noise reduction"""
        # Compute STFT
        f, t, stft = scipy.signal.stft(audio_data, self.sample_rate, 
                                      nperseg=self.frame_length, 
                                      noverlap=self.hop_length)
        
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise (first 10 frames)
        noise_magnitude = np.mean(magnitude[:, :10], axis=1, keepdims=True)
        
        # Spectral subtraction
        enhanced_magnitude = magnitude - alpha * noise_magnitude
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
        
        # Reconstruct signal
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        _, enhanced_audio = scipy.signal.istft(enhanced_stft, self.sample_rate,
                                             nperseg=self.frame_length,
                                             noverlap=self.hop_length)
        
        return enhanced_audio
    
    def _wiener_filter(self, audio_data, noise_factor=0.1):
        """Wiener filter for noise reduction"""
        # Simple Wiener filter implementation
        signal_fft = np.fft.fft(audio_data)
        power_spectrum = np.abs(signal_fft) ** 2
        
        # Estimate noise power (assuming first 10% is noise)
        noise_power = np.mean(power_spectrum[:len(power_spectrum)//10])
        
        # Wiener filter
        wiener_filter = power_spectrum / (power_spectrum + noise_factor * noise_power)
        filtered_fft = signal_fft * wiener_filter
        
        return np.real(np.fft.ifft(filtered_fft))
    
    def _median_filter(self, audio_data, kernel_size=5):
        """Median filter for impulsive noise reduction"""
        return scipy.signal.medfilt(audio_data, kernel_size=kernel_size)
    
    # ============================
    # SPEECH RECOGNITION FUNCTIONS
    # ============================
    
    def transcribe_audio(self, audio_data, engine='google', language='en-US'):
        """
        Transcribe audio using specified engine
        
        Args:
            audio_data (np.array): Input audio
            engine (str): Recognition engine to use
            language (str): Language code
            
        Returns:
            dict: Transcription results with metadata
        """
        if not ASR_AVAILABLE:
            return {"error": "Speech recognition not available"}
        
        start_time = time.time()
        
        try:
            # Convert numpy array to AudioData
            audio_data_int16 = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_data_int16.tobytes()
            
            audio_data_sr = sr.AudioData(audio_bytes, self.sample_rate, 2)
            
            # Use specified engine
            if engine in self.engines:
                result = self.engines[engine](audio_data_sr, language)
            else:
                result = self._recognize_google(audio_data_sr, language)
            
            processing_time = time.time() - start_time
            
            return {
                'transcription': result.get('transcription', ''),
                'confidence': result.get('confidence', 0.0),
                'engine': engine,
                'processing_time': processing_time,
                'language': language,
                'audio_length': len(audio_data) / self.sample_rate
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'engine': engine,
                'processing_time': time.time() - start_time
            }
    
    def _recognize_google(self, audio_data, language='en-US'):
        """Google Speech Recognition"""
        try:
            transcription = self.recognizer.recognize_google(audio_data, language=language)
            return {'transcription': transcription, 'confidence': 1.0}
        except sr.UnknownValueError:
            return {'transcription': '', 'confidence': 0.0}
        except Exception as e:
            return {'error': str(e)}
    
    def _recognize_sphinx(self, audio_data, language='en-US'):
        """CMU Sphinx Recognition"""
        try:
            transcription = self.recognizer.recognize_sphinx(audio_data)
            return {'transcription': transcription, 'confidence': 0.8}
        except sr.UnknownValueError:
            return {'transcription': '', 'confidence': 0.0}
        except Exception as e:
            return {'error': str(e)}
    
    def _recognize_wit(self, audio_data, language='en-US'):
        """Wit.ai Recognition (requires API key)"""
        try:
            # Replace with your Wit.ai API key
            WIT_AI_KEY = "YOUR_WIT_AI_KEY"
            transcription = self.recognizer.recognize_wit(audio_data, key=WIT_AI_KEY)
            return {'transcription': transcription, 'confidence': 0.9}
        except Exception as e:
            return {'error': 'Wit.ai API key required'}
    
    def _recognize_azure(self, audio_data, language='en-US'):
        """Azure Speech Recognition (requires API key)"""
        return {'error': 'Azure API key required'}
    
    def _recognize_watson(self, audio_data, language='en-US'):
        """IBM Watson Recognition (requires API key)"""
        return {'error': 'Watson API key required'}
    
    def transcribe_file(self, filename, engines=['google'], preprocess=True):
        """
        Transcribe audio file using multiple engines
        
        Args:
            filename (str): Path to audio file
            engines (list): List of engines to use
            preprocess (bool): Whether to preprocess audio
            
        Returns:
            dict: Results from all engines
        """
        # Load audio
        audio_data, sr = self.load_audio(filename)
        if audio_data is None:
            return {'error': 'Could not load audio file'}
        
        # Preprocess if requested
        if preprocess:
            audio_data = self.preprocess_audio(audio_data)
        
        # Transcribe with each engine
        results = {}
        for engine in engines:
            print(f"Transcribing with {engine}...")
            result = self.transcribe_audio(audio_data, engine)
            results[engine] = result
        
        return results
    
    # ============================
    # EVALUATION FUNCTIONS
    # ============================
    
    def calculate_wer(self, reference, hypothesis):
        """
        Calculate Word Error Rate
        
        Args:
            reference (str): Ground truth text
            hypothesis (str): Recognized text
            
        Returns:
            dict: WER metrics
        """
        if not WER_AVAILABLE:
            return {'error': 'jiwer not available'}
        
        try:
            # Basic WER
            wer = jiwer.wer(reference, hypothesis)
            
            # Additional metrics
            mer = jiwer.mer(reference, hypothesis)  # Match Error Rate
            wil = jiwer.wil(reference, hypothesis)  # Word Information Lost
            wip = jiwer.wip(reference, hypothesis)  # Word Information Preserved
            
            # Character-level metrics
            cer = jiwer.cer(reference, hypothesis)  # Character Error Rate
            
            return {
                'wer': wer * 100,  # Convert to percentage
                'mer': mer * 100,
                'wil': wil * 100,
                'wip': wip * 100,
                'cer': cer * 100,
                'reference': reference,
                'hypothesis': hypothesis
            }
        except Exception as e:
            return {'error': str(e)}
    
    def evaluate_asr_performance(self, test_cases):
        """
        Evaluate ASR performance across multiple test cases
        
        Args:
            test_cases (list): List of (audio_file, reference_text) tuples
            
        Returns:
            pandas.DataFrame: Performance metrics
        """
        results = []
        
        for i, (audio_file, reference_text) in enumerate(test_cases):
            print(f"Evaluating test case {i+1}/{len(test_cases)}: {audio_file}")
            
            # Transcribe with different preprocessing options
            configs = [
                {'preprocess': False, 'noise_reduction': None},
                {'preprocess': True, 'noise_reduction': None},
                {'preprocess': True, 'noise_reduction': 'noisereduce'},
                {'preprocess': True, 'noise_reduction': 'spectral_subtraction'}
            ]
            
            for config in configs:
                # Load and process audio
                audio_data, _ = self.load_audio(audio_file)
                if audio_data is None:
                    continue
                
                if config['preprocess']:
                    audio_data = self.preprocess_audio(audio_data)
                
                if config['noise_reduction']:
                    audio_data = self.reduce_noise(audio_data, config['noise_reduction'])
                
                # Transcribe
                transcription_result = self.transcribe_audio(audio_data)
                if 'error' not in transcription_result:
                    hypothesis = transcription_result['transcription']
                    
                    # Calculate metrics
                    wer_metrics = self.calculate_wer(reference_text, hypothesis)
                    
                    if 'error' not in wer_metrics:
                        results.append({
                            'test_case': i+1,
                            'audio_file': audio_file,
                            'preprocessing': config['preprocess'],
                            'noise_reduction': config['noise_reduction'],
                            'wer': wer_metrics['wer'],
                            'cer': wer_metrics['cer'],
                            'processing_time': transcription_result['processing_time'],
                            'reference': reference_text,
                            'hypothesis': hypothesis
                        })
        
        return pd.DataFrame(results)
    
    # ============================
    # VISUALIZATION FUNCTIONS
    # ============================
    
    def plot_audio_waveform(self, audio_data, title="Audio Waveform", save_path=None):
        """Plot audio waveform"""
        plt.figure(figsize=(12, 4))
        time_axis = np.linspace(0, len(audio_data) / self.sample_rate, len(audio_data))
        plt.plot(time_axis, audio_data)
        plt.title(title)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_spectrogram(self, audio_data, title="Spectrogram", save_path=None):
        """Plot audio spectrogram"""
        plt.figure(figsize=(12, 6))
        
        if LIBROSA_AVAILABLE:
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
            librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=self.sample_rate)
            plt.colorbar(format='%+2.0f dB')
        else:
            f, t, Sxx = scipy.signal.spectrogram(audio_data, self.sample_rate)
            plt.pcolormesh(t, f, 10 * np.log10(Sxx))
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.colorbar(label='Power Spectral Density [dB]')
        
        plt.title(title)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_noise_comparison(self, original_audio, cleaned_audio, save_path=None):
        """Compare original and noise-reduced audio"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time domain comparison
        time_axis_orig = np.linspace(0, len(original_audio) / self.sample_rate, len(original_audio))
        time_axis_clean = np.linspace(0, len(cleaned_audio) / self.sample_rate, len(cleaned_audio))
        
        axes[0, 0].plot(time_axis_orig, original_audio)
        axes[0, 0].set_title("Original Audio (Time Domain)")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True)
        
        axes[1, 0].plot(time_axis_clean, cleaned_audio)
        axes[1, 0].set_title("Noise-Reduced Audio (Time Domain)")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Amplitude")
        axes[1, 0].grid(True)
        
        # Frequency domain comparison
        freqs_orig, psd_orig = scipy.signal.welch(original_audio, self.sample_rate)
        freqs_clean, psd_clean = scipy.signal.welch(cleaned_audio, self.sample_rate)
        
        axes[0, 1].semilogy(freqs_orig, psd_orig)
        axes[0, 1].set_title("Original Audio (Frequency Domain)")
        axes[0, 1].set_xlabel("Frequency (Hz)")
        axes[0, 1].set_ylabel("PSD")
        axes[0, 1].grid(True)
        
        axes[1, 1].semilogy(freqs_clean, psd_clean)
        axes[1, 1].set_title("Noise-Reduced Audio (Frequency Domain)")
        axes[1, 1].set_xlabel("Frequency (Hz)")
        axes[1, 1].set_ylabel("PSD")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_wer_comparison(self, results_df, save_path=None):
        """Plot WER comparison across different configurations"""
        plt.figure(figsize=(12, 8))
        
        # Group by configuration
        grouped = results_df.groupby(['preprocessing', 'noise_reduction'])['wer'].mean().reset_index()
        
        # Create bar plot
        sns.barplot(data=grouped, x='noise_reduction', y='wer', hue='preprocessing')
        plt.title("WER Comparison Across Different Configurations")
        plt.xlabel("Noise Reduction Method")
        plt.ylabel("Word Error Rate (%)")
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    # ============================
    # REAL-TIME ASR FUNCTIONS
    # ============================
    
    def real_time_transcription(self, duration=30, chunk_duration=2):
        """
        Real-time speech transcription
        
        Args:
            duration (float): Total recording duration
            chunk_duration (float): Duration of each processing chunk
        """
        if not AUDIO_AVAILABLE or not ASR_AVAILABLE:
            print("Real-time transcription requires sounddevice and speech_recognition")
            return
        
        print(f"Starting real-time transcription for {duration} seconds...")
        print("Speak now...")
        
        # Audio buffer
        audio_buffer = []
        transcriptions = []
        
        chunk_samples = int(chunk_duration * self.sample_rate)
        total_chunks = int(duration / chunk_duration)
        
        for chunk_num in range(total_chunks):
            print(f"Chunk {chunk_num + 1}/{total_chunks}")
            
            # Record chunk
            chunk_data = sd.rec(chunk_samples, samplerate=self.sample_rate, channels=1)
            sd.wait()
            
            # Process chunk
            chunk_data = chunk_data.flatten()
            chunk_data = self.preprocess_audio(chunk_data)
            
            # Transcribe
            result = self.transcribe_audio(chunk_data)
            if 'transcription' in result and result['transcription']:
                transcriptions.append(result['transcription'])
                print(f"  -> {result['transcription']}")
            
            audio_buffer.extend(chunk_data)
        
        # Final transcription
        print("\n" + "="*50)
        print("FINAL TRANSCRIPTION:")
        print("="*50)
        full_transcription = " ".join(transcriptions)
        print(full_transcription)
        
        return {
            'full_audio': np.array(audio_buffer),
            'chunk_transcriptions': transcriptions,
            'full_transcription': full_transcription
        }
    
    # ============================
    # MAIN DEMO FUNCTIONS
    # ============================
    
    def run_complete_demo(self):
        """Run complete ASR demonstration"""
        print("COMPLETE ASR SYSTEM DEMONSTRATION")
        print("="*60)
        
        # Test sentence
        reference_sentence = "The quick brown fox jumps over the lazy dog"
        
        print(f"Reference Sentence: '{reference_sentence}'")
        print()
        
        # 1. Record clean audio
        print("STEP 1: Recording clean audio...")
        clean_audio = self.record_audio(duration=5, filename="clean_recording.wav")
        
        # 2. Record noisy audio (or simulate noise)
        print("\nSTEP 2: Recording noisy audio...")
        print("(Try to add background noise or speak from farther away)")
        noisy_audio = self.record_audio(duration=5, filename="noisy_recording.wav")
        
        # Add artificial noise if needed
        if np.std(noisy_audio) < np.std(clean_audio) * 0.8:
            print("Adding artificial noise for demonstration...")
            noise = np.random.normal(0, 0.1, len(noisy_audio))
            noisy_audio = noisy_audio + noise
            self.save_audio(noisy_audio, "noisy_recording.wav")
        
        # 3. Apply noise reduction
        print("\nSTEP 3: Applying noise reduction...")
        cleaned_audio = self.reduce_noise(noisy_audio, method='noisereduce')
        self.save_audio(cleaned_audio, "cleaned_recording.wav")
        
        # 4. Transcribe all versions
        print("\nSTEP 4: Transcribing all audio versions...")
        
        transcriptions = {}
        audio_versions = {
            'Clean': clean_audio,
            'Noisy': noisy_audio,
            'Cleaned': cleaned_audio
        }
        
        for version_name, audio_data in audio_versions.items():
            result = self.transcribe_audio(audio_data)
            transcriptions[version_name] = result
            print(f"{version_name} Audio: '{result.get('transcription', 'ERROR')}'")
        
        # 5. Calculate WER
        print("\nSTEP 5: Calculating Word Error Rates...")
        wer_results = {}
        
        for version_name, result in transcriptions.items():
            if 'transcription' in result:
                wer_metrics = self.calculate_wer(reference_sentence, result['transcription'])
                wer_results[version_name] = wer_metrics['wer']
                print(f"{version_name} WER: {wer_metrics['wer']:.2f}%")
        
        # 6. Visualizations
        print("\nSTEP 6: Creating visualizations...")
        
        # Audio comparison
        self.plot_noise_comparison(noisy_audio, cleaned_audio, "noise_comparison.png")
        
        # Spectrograms
        self.plot_spectrogram(clean_audio, "Clean Audio Spectrogram", "clean_spectrogram.png")
        self.plot_spectrogram(noisy_audio, "Noisy Audio Spectrogram", "noisy_spectrogram.png")
        self.plot_spectrogram(cleaned_audio, "Cleaned Audio Spectrogram", "cleaned_spectrogram.png")
        
        # 7. Summary Report
        print("\n" + "="*60)
        print("FINAL ANALYSIS REPORT")
        print("="*60)
        
        print(f"Reference: '{reference_sentence}'")
        print()
        
        for version_name in ['Clean', 'Noisy', 'Cleaned']:
            if version_name in transcriptions and version_name in wer_results:
                transcription = transcriptions[version_name].get('transcription', 'ERROR')
                wer = wer_results[version_name]
                processing_time = transcriptions[version_name].get('processing_time', 0)
                
                print(f"{version_name} Audio:")
                print(f"  Transcription: '{transcription}'")
                print(f"  WER: {wer:.2f}%")
                print(f"  Processing Time: {processing_time:.3f}s")
                print()
        
        # Improvement analysis
        if 'Noisy' in wer_results and 'Cleaned' in wer_results:
            improvement = wer_results['Noisy'] - wer_results['Cleaned']
            print(f"Noise Reduction Improvement: {improvement:.2f} percentage points")
            if improvement > 0:
                print("✓ Noise reduction IMPROVED ASR accuracy")
            elif improvement < 0:
                print("✗ Noise reduction DECREASED ASR accuracy") 
            else:
                print("○ No change in ASR accuracy")
        
        print("\nGenerated Files:")
        print("- clean_recording.wav")
        print("- noisy_recording.wav") 
        print("- cleaned_recording.wav")
        print("- noise_comparison.png")
        print("- clean_spectrogram.png")
        print("- noisy_spectrogram.png")
        print("- cleaned_spectrogram.png")
        
        return {
            'transcriptions': transcriptions,
            'wer_results': wer_results,
            'audio_data': audio_versions,
            'reference': reference_sentence
        }
    
    def run_advanced_evaluation(self, test_files=None):
        """Run advanced ASR evaluation with multiple configurations"""
        print("ADVANCED ASR EVALUATION")
        print("="*60)
        
        if test_files is None:
            # Use recorded files if no test files provided
            test_files = [
                ("clean_recording.wav", "The quick brown fox jumps over the lazy dog"),
                ("noisy_recording.wav", "The quick brown fox jumps over the lazy dog")
            ]
        
        # Test different configurations
        configurations = [
            {'name': 'Baseline', 'preprocess': False, 'noise_reduction': None},
            {'name': 'Preprocessed', 'preprocess': True, 'noise_reduction': None},
            {'name': 'NoiseReduce', 'preprocess': True, 'noise_reduction': 'noisereduce'},
            {'name': 'SpectralSub', 'preprocess': True, 'noise_reduction': 'spectral_subtraction'},
            {'name': 'WienerFilter', 'preprocess': True, 'noise_reduction': 'wiener_filter'},
        ]
        
        results = []
        
        for test_file, reference in test_files:
            if not os.path.exists(test_file):
                continue
                
            print(f"\nEvaluating: {test_file}")
            print(f"Reference: '{reference}'")
            print("-" * 40)
            
            # Load original audio
            original_audio, _ = self.load_audio(test_file)
            if original_audio is None:
                continue
            
            for config in configurations:
                print(f"Testing {config['name']}...")
                
                # Process audio according to configuration
                processed_audio = original_audio.copy()
                
                if config['preprocess']:
                    processed_audio = self.preprocess_audio(processed_audio)
                
                if config['noise_reduction']:
                    processed_audio = self.reduce_noise(processed_audio, config['noise_reduction'])
                
                # Transcribe
                transcription_result = self.transcribe_audio(processed_audio)
                
                if 'error' not in transcription_result:
                    hypothesis = transcription_result['transcription']
                    
                    # Calculate metrics
                    wer_metrics = self.calculate_wer(reference, hypothesis)
                    
                    if 'error' not in wer_metrics:
                        result = {
                            'file': test_file,
                            'configuration': config['name'],
                            'transcription': hypothesis,
                            'wer': wer_metrics['wer'],
                            'cer': wer_metrics['cer'],
                            'processing_time': transcription_result['processing_time'],
                            'reference': reference
                        }
                        results.append(result)
                        
                        print(f"  WER: {wer_metrics['wer']:.2f}%")
                        print(f"  Transcription: '{hypothesis}'")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            # Summary statistics
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            
            summary = results_df.groupby('configuration').agg({
                'wer': ['mean', 'std', 'min', 'max'],
                'cer': ['mean', 'std'],
                'processing_time': ['mean', 'std']
            }).round(3)
            
            print(summary)
            
            # Best configuration
            best_config = results_df.loc[results_df['wer'].idxmin()]
            print(f"\nBest Configuration: {best_config['configuration']}")
            print(f"Best WER: {best_config['wer']:.2f}%")
            
            # Save results
            results_df.to_csv('asr_evaluation_results.csv', index=False)
            print("\n✓ Results saved to 'asr_evaluation_results.csv'")
            
            # Plot results
            if len(results_df) > 1:
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=results_df, x='configuration', y='wer')
                plt.title('WER Distribution Across Configurations')
                plt.ylabel('Word Error Rate (%)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('wer_comparison.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        return results_df
    
    def benchmark_engines(self, audio_file, reference_text):
        """Benchmark different ASR engines"""
        print("ASR ENGINE BENCHMARK")
        print("="*60)
        
        engines_to_test = ['google', 'sphinx']  # Add more if API keys available
        
        # Load audio
        audio_data, _ = self.load_audio(audio_file)
        if audio_data is None:
            print(f"Could not load {audio_file}")
            return None
        
        # Preprocess audio
        audio_data = self.preprocess_audio(audio_data)
        
        results = {}
        
        for engine in engines_to_test:
            print(f"\nTesting {engine.upper()} engine...")
            
            result = self.transcribe_audio(audio_data, engine=engine)
            
            if 'error' not in result:
                transcription = result['transcription']
                processing_time = result['processing_time']
                
                # Calculate WER
                wer_metrics = self.calculate_wer(reference_text, transcription)
                
                results[engine] = {
                    'transcription': transcription,
                    'wer': wer_metrics.get('wer', float('inf')),
                    'processing_time': processing_time,
                    'confidence': result.get('confidence', 0.0)
                }
                
                print(f"  Transcription: '{transcription}'")
                print(f"  WER: {wer_metrics.get('wer', 'N/A'):.2f}%")
                print(f"  Time: {processing_time:.3f}s")
            else:
                print(f"  Error: {result['error']}")
                results[engine] = {'error': result['error']}
        
        return results

# ============================
# UTILITY FUNCTIONS
# ============================

def create_test_dataset():
    """Create a test dataset with various audio samples"""
    test_sentences = [
        "The quick brown fox jumps over the lazy dog",
        "Speech recognition technology has advanced significantly", 
        "Machine learning enables computers to understand human speech",
        "Noise reduction improves transcription accuracy",
        "Automatic speech recognition systems process audio signals"
    ]
    
    print("Creating test dataset...")
    print("Please record the following sentences:")
    
    test_files = []
    
    for i, sentence in enumerate(test_sentences):
        filename = f"test_sample_{i+1}.wav"
        print(f"\nSentence {i+1}: '{sentence}'")
        
        input(f"Press Enter to record {filename}...")
        
        # Initialize ASR system for recording
        asr_system = AdvancedASRSystem()
        audio_data = asr_system.record_audio(duration=5, filename=filename)
        
        test_files.append((filename, sentence))
        print(f"✓ Recorded {filename}")
    
    return test_files

def run_noise_analysis_study():
    """Run comprehensive noise analysis study"""
    print("COMPREHENSIVE NOISE ANALYSIS STUDY")
    print("="*60)
    
    # Initialize system
    asr_system = AdvancedASRSystem()
    
    # Test sentence
    reference = "The quick brown fox jumps over the lazy dog"
    
    # Record clean audio
    print("Step 1: Record clean reference audio")
    input("Press Enter when ready to record clean audio...")
    clean_audio = asr_system.record_audio(duration=5, filename="study_clean.wav")
    
    # Create different noise conditions
    noise_conditions = [
        {'name': 'White Noise (Low)', 'type': 'white', 'level': 0.05},
        {'name': 'White Noise (Medium)', 'type': 'white', 'level': 0.15}, 
        {'name': 'White Noise (High)', 'type': 'white', 'level': 0.30},
        {'name': 'Pink Noise', 'type': 'pink', 'level': 0.15},
        {'name': 'Brown Noise', 'type': 'brown', 'level': 0.15}
    ]
    
    results = []
    
    for condition in noise_conditions:
        print(f"\nTesting: {condition['name']}")
        
        # Add synthetic noise
        if condition['type'] == 'white':
            noise = np.random.normal(0, condition['level'], len(clean_audio))
        elif condition['type'] == 'pink':
            # Pink noise (1/f spectrum)
            noise = asr_system._generate_pink_noise(len(clean_audio), condition['level'])
        elif condition['type'] == 'brown':
            # Brown noise (1/f^2 spectrum) 
            noise = asr_system._generate_brown_noise(len(clean_audio), condition['level'])
        
        noisy_audio = clean_audio + noise
        noisy_filename = f"study_noisy_{condition['name'].lower().replace(' ', '_')}.wav"
        asr_system.save_audio(noisy_audio, noisy_filename)
        
        # Test different noise reduction methods
        reduction_methods = ['noisereduce', 'spectral_subtraction', 'wiener_filter']
        
        for method in reduction_methods:
            # Apply noise reduction
            cleaned_audio = asr_system.reduce_noise(noisy_audio, method=method)
            cleaned_filename = f"study_cleaned_{condition['name'].lower().replace(' ', '_')}_{method}.wav"
            asr_system.save_audio(cleaned_audio, cleaned_filename)
            
            # Transcribe both noisy and cleaned
            noisy_result = asr_system.transcribe_audio(noisy_audio)
            cleaned_result = asr_system.transcribe_audio(cleaned_audio)
            
            if 'transcription' in noisy_result and 'transcription' in cleaned_result:
                noisy_wer = asr_system.calculate_wer(reference, noisy_result['transcription'])
                cleaned_wer = asr_system.calculate_wer(reference, cleaned_result['transcription'])
                
                result = {
                    'noise_condition': condition['name'],
                    'noise_level': condition['level'],
                    'reduction_method': method,
                    'noisy_wer': noisy_wer.get('wer', float('inf')),
                    'cleaned_wer': cleaned_wer.get('wer', float('inf')),
                    'improvement': noisy_wer.get('wer', float('inf')) - cleaned_wer.get('wer', float('inf')),
                    'noisy_transcription': noisy_result['transcription'],
                    'cleaned_transcription': cleaned_result['transcription']
                }
                results.append(result)
                
                print(f"  {method}: {result['noisy_wer']:.1f}% -> {result['cleaned_wer']:.1f}% (Δ{result['improvement']:+.1f}%)")
    
    # Save and analyze results
    results_df = pd.DataFrame(results)
    results_df.to_csv('noise_analysis_study.csv', index=False)
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # WER by noise condition
    sns.boxplot(data=results_df, x='noise_condition', y='cleaned_wer', ax=axes[0,0])
    axes[0,0].set_title('WER by Noise Condition (After Cleaning)')
    axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
    
    # Improvement by method
    sns.boxplot(data=results_df, x='reduction_method', y='improvement', ax=axes[0,1])
    axes[0,1].set_title('WER Improvement by Noise Reduction Method')
    
    # Heatmap of improvements
    pivot_table = results_df.pivot_table(values='improvement', 
                                       index='noise_condition', 
                                       columns='reduction_method', 
                                       aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, fmt='.1f', ax=axes[1,0], cmap='RdYlGn')
    axes[1,0].set_title('Average WER Improvement Heatmap')
    
    # Noise level vs performance
    sns.scatterplot(data=results_df, x='noise_level', y='improvement', 
                   hue='reduction_method', ax=axes[1,1])
    axes[1,1].set_title('Noise Level vs WER Improvement')
    
    plt.tight_layout()
    plt.savefig('noise_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df

def _generate_pink_noise(self, length, amplitude):
    """Generate pink noise (1/f spectrum)"""
    # Simple pink noise approximation
    white_noise = np.random.normal(0, 1, length)
    # Apply 1/f filter (simplified)
    freqs = np.fft.fftfreq(length)
    freqs[0] = 1  # Avoid division by zero
    filter_response = 1 / np.sqrt(np.abs(freqs))
    pink_noise_fft = np.fft.fft(white_noise) * filter_response
    pink_noise = np.real(np.fft.ifft(pink_noise_fft))
    return pink_noise * amplitude / np.std(pink_noise)

def _generate_brown_noise(self, length, amplitude):
    """Generate brown noise (1/f^2 spectrum)"""
    white_noise = np.random.normal(0, 1, length)
    freqs = np.fft.fftfreq(length)
    freqs[0] = 1  # Avoid division by zero
    filter_response = 1 / np.abs(freqs)
    brown_noise_fft = np.fft.fft(white_noise) * filter_response
    brown_noise = np.real(np.fft.ifft(brown_noise_fft))
    return brown_noise * amplitude / np.std(brown_noise)

# Add noise generation methods to the class
AdvancedASRSystem._generate_pink_noise = _generate_pink_noise
AdvancedASRSystem._generate_brown_noise = _generate_brown_noise

# ============================
# MAIN EXECUTION
# ============================

def main():
    """Main function to run the complete ASR system"""
    print("ADVANCED ASR SYSTEM - MAIN MENU")
    print("="*60)
    print("1. Complete Demo (Record + Process + Analyze)")
    print("2. Advanced Evaluation (Multiple Configurations)")  
    print("3. Engine Benchmark (Compare ASR Engines)")
    print("4. Real-time Transcription")
    print("5. Noise Analysis Study")
    print("6. Create Test Dataset")
    print("0. Exit")
    
    asr_system = AdvancedASRSystem()
    
    while True:
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '1':
            try:
                results = asr_system.run_complete_demo()
                print("✓ Complete demo finished successfully!")
            except Exception as e:
                print(f"Error in demo: {e}")
        
        elif choice == '2':
            try:
                results_df = asr_system.run_advanced_evaluation()
                print("✓ Advanced evaluation completed!")
            except Exception as e:
                print(f"Error in evaluation: {e}")
        
        elif choice == '3':
            audio_file = input("Enter audio file path: ").strip()
            reference = input("Enter reference text: ").strip()
            try:
                results = asr_system.benchmark_engines(audio_file, reference)
                print("✓ Engine benchmark completed!")
            except Exception as e:
                print(f"Error in benchmark: {e}")
        
        elif choice == '4':
            try:
                duration = float(input("Enter duration (seconds, default 30): ") or "30")
                results = asr_system.real_time_transcription(duration=duration)
                print("✓ Real-time transcription completed!")
            except Exception as e:
                print(f"Error in real-time transcription: {e}")
        
        elif choice == '5':
            try:
                results_df = run_noise_analysis_study()
                print("✓ Noise analysis study completed!")
            except Exception as e:
                print(f"Error in noise study: {e}")
        
        elif choice == '6':
            try:
                test_files = create_test_dataset()
                print("✓ Test dataset created!")
            except Exception as e:
                print(f"Error creating dataset: {e}")
        
        elif choice == '0':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()