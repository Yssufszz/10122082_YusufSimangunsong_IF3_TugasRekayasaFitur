import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte
import warnings
warnings.filterwarnings('ignore')

class VideoFeatureExtractor:
    """Ini Kelas buat ngekstrak fitur dari video"""
    
    def __init__(self, data_folder='data/', output_folder='output/'):
        """
        Inisialisasi ekstrator fitur video
        
        Args:
            data_folder (str): Path ke folder yang berisi video
            output_folder (str): Path ke folder output
        """
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.features = []
        self.video_files = []
        
        # Buat Folder Outputnya kalo belum ada
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Ini parameter buat LBP
        self.lbp_radius = 1
        self.lbp_n_points = 8 * self.lbp_radius
        
        print("Video featurenya udah berhasil diload")
        print(f" Data folder: {self.data_folder}")
        print(f" Output folder: {self.output_folder}")
    
    def get_video_files(self):
        """Lagi ambil daftar video dari folder data"""
        if not os.path.exists(self.data_folder):
            raise FileNotFoundError(f"Folder {self.data_folder} Ga Ketemu")
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        self.video_files = []
        
        for file in os.listdir(self.data_folder):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                self.video_files.append(os.path.join(self.data_folder, file))
        
        if not self.video_files:
            raise ValueError(f"Tidak ada file video ditemukan di folder {self.data_folder}")
        
        print(f"Nah Ada {len(self.video_files)} file video:")
        for i, file in enumerate(self.video_files, 1):
            print(f"   {i}. {os.path.basename(file)}")
        
        return self.video_files
    
    def preprocess_frame(self, frame, target_size=(640, 480)):
        """
        Langkah awal pra-pemrosesan: resize frame dan ubah jadi hitam putih (grayscale)

        Args:
            frame (numpy.ndarray): Frame video yang akan diproses
            target_size (tuple): Ukuran yang ditargetkan (width, height)

        Returns:
            tuple: (frame yang sudah di-resize, frame dalam bentuk grayscale)
        """
        # Resize frame
        frame_resized = cv2.resize(frame, target_size)
        
        # Ngonversi ke grayscale
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        return frame_resized, frame_gray
    
    def extract_color_histogram(self, frame):
        """
        Ngambil fitur warna dari frame dengan histogram RGB

        Args:
            frame (numpy.ndarray): Frame video dalam format BGR (standar OpenCV)

        Returns:
            numpy.ndarray: Histogram gabungan dari warna merah, hijau, dan biru (R+G+B)
        """
        # Konversi BGR ke RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ngitung histogram untuk setiap channel
        hist_r = cv2.calcHist([frame_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([frame_rgb], [2], None, [256], [0, 256])
        
        # Ngegabungin histogram sama normalisasi
        combined_hist = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
        combined_hist = combined_hist / np.sum(combined_hist)
        
        return combined_hist
    
    def extract_optical_flow(self, prev_gray, curr_gray):
        """
        Ngambil fitur optical flow pakai metode Farnebäck

        Args:
            prev_gray (numpy.ndarray): Frame grayscale sebelumnya
            curr_gray (numpy.ndarray): Frame grayscale yang sekarang

        Returns:
            float: Nilai rata-rata dari magnitudo pergerakan (optical flow)
        """
        
        # Pakai metode Farnebäck untuk dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Hitung magnitudo dan sudut
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Rata-rata magnitudo
        avg_magnitude = np.mean(magnitude)
        
        return avg_magnitude
    
    def extract_edge_density(self, frame_gray):
        """
        Ekstraksi fitur kepadatan tepi menggunakan Canny edge detection
        
        Args:
            frame_gray (numpy.ndarray): Frame grayscale
        
        Returns:
            float: Kepadatan tepi (proporsi piksel tepi)
        """
        # Deteksi tepi dengan Canny
        edges = cv2.Canny(frame_gray, 50, 150)
        
        # Hitung kepadatan tepi
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return edge_density
    
    def extract_lbp_histogram(self, frame_gray):
        """
        Ngambil fitur tekstur dari frame pakai histogram LBP (Local Binary Patterns)

        Args:
            frame_gray (numpy.ndarray): Frame dalam format grayscale

        Returns:
            numpy.ndarray: Histogram LBP yang udah dinormalisasi
        """
        # Hitung LBP
        lbp = local_binary_pattern(frame_gray, self.lbp_n_points, self.lbp_radius, method='uniform')
        
        # Hitung histogram LBP
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=self.lbp_n_points + 2, range=(0, self.lbp_n_points + 2))
        
        # Normalisasi histogram
        hist_lbp = hist_lbp.astype(float)
        hist_lbp /= (hist_lbp.sum() + 1e-7)
        
        return hist_lbp
    
    def extract_features_from_video(self, video_path):
        """
        Ngambil semua fitur dari satu file video

        Args:
            video_path (str): Lokasi/path ke file video

        Returns:
            dict: Dictionary yang berisi semua fitur hasil ekstraksi
        """
        print(f"Memproses: {os.path.basename(video_path)}")
        
        # Buka video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Tidak dapat membuka video: {video_path}")
        
        # Inisialisasi variabel untuk menyimpan fitur
        color_histograms = []
        optical_flows = []
        edge_densities = []
        lbp_histograms = []
        
        # Baca frame pertama
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError(f"Tidak dapat membaca frame dari video: {video_path}")
        
        prev_frame_resized, prev_gray = self.preprocess_frame(prev_frame)
        
        # Dapatkan informasi video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"    Total frames: {total_frames}, FPS: {fps:.2f}")
        
        # Progress bar untuk frame
        frame_pbar = tqdm(total=total_frames-1, desc="   Ekstraksi frame", leave=False)
        
        frame_count = 0
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            # Pra-pemrosesan frame
            curr_frame_resized, curr_gray = self.preprocess_frame(curr_frame)
            
            # Ekstraksi fitur
            try:
                # 1. Histogram warna
                color_hist = self.extract_color_histogram(curr_frame_resized)
                color_histograms.append(color_hist)
                
                # 2. Optical flow
                if frame_count > 0:  # Jadinya optical flow cuma dihitung setelah frame pertama
                    flow_magnitude = self.extract_optical_flow(prev_gray, curr_gray)
                    optical_flows.append(flow_magnitude)
                
                # 3. Edge density
                edge_density = self.extract_edge_density(curr_gray)
                edge_densities.append(edge_density)
                
                # 4. LBP histogram
                lbp_hist = self.extract_lbp_histogram(curr_gray)
                lbp_histograms.append(lbp_hist)
                
            except Exception as e:
                print(f"    Error processing frame {frame_count}: {e}")
                continue
            
            # Update frame sebelumnya
            prev_gray = curr_gray.copy()
            frame_count += 1
            frame_pbar.update(1)
        
        frame_pbar.close()
        cap.release()
        
        # Hitung rata-rata fitur
        features = {
            'filename': os.path.basename(video_path),
            'total_frames': total_frames,
            'fps': fps,
            'duration_seconds': total_frames / fps if fps > 0 else 0,
            
            # Fitur warna: rata-rata histogram RGB
            'avg_color_histogram': np.mean(color_histograms, axis=0) if color_histograms else np.zeros(768),
            
            # Fitur gerakan: rata-rata magnitudo optical flow
            'avg_flow_magnitude': np.mean(optical_flows) if optical_flows else 0,
            
            # Fitur kompleksitas: rata-rata kepadatan tepi
            'avg_edge_density': np.mean(edge_densities) if edge_densities else 0,
            
            # Fitur tekstur: rata-rata histogram LBP
            'avg_lbp_histogram': np.mean(lbp_histograms, axis=0) if lbp_histograms else np.zeros(self.lbp_n_points + 2)
        }
        
        print(f"    Berhasil diekstrak {len(color_histograms)} frame")
        return features
    
    def process_all_videos(self):
        """Proses semua video dan ekstrak fiturnya"""
        print("\n Memulai ekstraksi fitur dari semua video...")
        
        # Dapatkan daftar file video
        video_files = self.get_video_files()
        
        # Progress bar untuk video
        video_pbar = tqdm(video_files, desc=" Memproses video")
        
        for video_path in video_pbar:
            video_pbar.set_postfix({"Current": os.path.basename(video_path)})
            
            try:
                features = self.extract_features_from_video(video_path)
                self.features.append(features)
                
            except Exception as e:
                print(f" Error processing {video_path}: {e}")
                continue
        
        video_pbar.close()
        print(f"\n Selesai memproses {len(self.features)} video")
    
    def save_features_to_csv(self):
        """Simpan fitur ke file CSV"""
        if not self.features:
            print(" Tidak ada fitur untuk disimpan")
            return
        
        print("\n Menyimpan fitur ke CSV...")
        
        # Konversi ke DataFrame
        df_data = []
        for feature in self.features:
            row = {
                'filename': feature['filename'],
                'total_frames': feature['total_frames'],
                'fps': feature['fps'],
                'duration_seconds': feature['duration_seconds'],
                'avg_flow_magnitude': feature['avg_flow_magnitude'],
                'avg_edge_density': feature['avg_edge_density']
            }
            
            # Tambahkan fitur histogram warna
            for i, val in enumerate(feature['avg_color_histogram']):
                row[f'color_hist_{i}'] = val
            
            # Tambahkan fitur histogram LBP
            for i, val in enumerate(feature['avg_lbp_histogram']):
                row[f'lbp_hist_{i}'] = val
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Simpan ke CSV
        csv_path = os.path.join(self.output_folder, 'features.csv')
        df.to_csv(csv_path, index=False)
        
        print(f" Fitur disimpan ke: {csv_path}")
        print(f" DataFrame shape: {df.shape}")
        
        return df
    
    def create_visualizations(self):
        """Buat visualisasi fitur"""
        if not self.features:
            print(" Tidak ada fitur untuk divisualisasikan")
            return
        
        print("\n Membuat visualisasi...")
        
        # Ntiapin data buat visualisasinya
        filenames = [f['filename'] for f in self.features]
        flow_magnitudes = [f['avg_flow_magnitude'] for f in self.features]
        edge_densities = [f['avg_edge_density'] for f in self.features]
        
        # Set style matplotlib
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Visualisasi Gerakan (Bar Chart)
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(filenames)), flow_magnitudes, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.xlabel('Video Files')
        plt.ylabel('Average Optical Flow Magnitude')
        plt.title('Perbandingan Tingkat Gerakan Antar Video', fontsize=14, fontweight='bold')
        plt.xticks(range(len(filenames)), [f.replace('.mp4', '') for f in filenames], rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Tambahkan nilai di atas bar
        for bar, val in zip(bars, flow_magnitudes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'visualisasi_gerakan.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Visualisasi Kompleksitas (Bar Chart)
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(filenames)), edge_densities, color='lightcoral', edgecolor='darkred', alpha=0.7)
        plt.xlabel('Video Files')
        plt.ylabel('Average Edge Density')
        plt.title('Perbandingan Kompleksitas Visual Antar Video', fontsize=14, fontweight='bold')
        plt.xticks(range(len(filenames)), [f.replace('.mp4', '') for f in filenames], rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Tambahkan nilai di atas bar
        for bar, val in zip(bars, edge_densities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'visualisasi_kompleksitas.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Visualisasi Warna (Line Plot)
        plt.figure(figsize=(15, 8))
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        
        for i, feature in enumerate(self.features):
            hist = feature['avg_color_histogram']
            # Ambil histogram R, G, B (masing-masing 256 bins)
            hist_r = hist[:256]
            hist_g = hist[256:512]
            hist_b = hist[512:768]
            
            # Plot histogram gabungan
            combined = (hist_r + hist_g + hist_b) / 3
            plt.plot(combined, label=feature['filename'].replace('.mp4', ''), 
                    color=colors[i % len(colors)], linewidth=2, alpha=0.8)
        
        plt.xlabel('Intensity Bins')
        plt.ylabel('Normalized Frequency')
        plt.title('Perbandingan Histogram Warna Antar Video', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'visualisasi_warna.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Visualisasi Tekstur LBP (Line Plot)
        plt.figure(figsize=(12, 8))
        
        for i, feature in enumerate(self.features):
            lbp_hist = feature['avg_lbp_histogram']
            plt.plot(lbp_hist, label=feature['filename'].replace('.mp4', ''), 
                    color=colors[i % len(colors)], linewidth=2, alpha=0.8, marker='o', markersize=4)
        
        plt.xlabel('LBP Pattern Bins')
        plt.ylabel('Normalized Frequency')
        plt.title('Perbandingan Histogram LBP (Tekstur) Antar Video', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'visualisasi_tekstur.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualisasi selesai dibuat:")
        print(f"    visualisasi_gerakan.png")
        print(f"    visualisasi_kompleksitas.png")
        print(f"    visualisasi_warna.png")
        print(f"    visualisasi_tekstur.png")
    
    def run(self):
        """Jalankan seluruh proses ekstraksi fitur"""
        print("="*60)
        print(" PROGRAM EKSTRAKSI FITUR VIDEO")
        print("="*60)
        
        try:
            # 1. Proses semua video
            self.process_all_videos()
            
            # 2. Simpan fitur ke CSV
            df = self.save_features_to_csv()
            
            # 3. Buat visualisasi
            self.create_visualizations()
            
            print("\n" + "="*60)
            print(" PROGRAM SELESAI BERHASIL!")
            print("="*60)
            print(f" Hasil tersimpan di folder: {self.output_folder}")
            print(f" Total video diproses: {len(self.features)}")
            
            # Tampilkan ringkasan fitur
            if self.features:
                print("\n RINGKASAN FITUR:")
                print("-" * 40)
                for feature in self.features:
                    print(f" {feature['filename']}")
                    print(f"   Frames: {feature['total_frames']}")
                    print(f"   Duration: {feature['duration_seconds']:.2f}s")
                    print(f"   Flow Magnitude: {feature['avg_flow_magnitude']:.4f}")
                    print(f"   Edge Density: {feature['avg_edge_density']:.4f}")
                    print()
        
        except Exception as e:
            print(f" Error: {e}")
            return False
        
        return True

def main():
    """Fungsi utama program"""
    # Inisialisasi ekstrator fitur
    extractor = VideoFeatureExtractor(
        data_folder='data/',
        output_folder='output/'
    )
    
    # Jalankan program
    success = extractor.run()
    
    if success:
        print(" Program berhasil dijalankan!")
    else:
        print(" Program gagal dijalankan!")

if __name__ == "__main__":
    main()