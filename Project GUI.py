import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import Counter
import heapq
import struct
import numpy as np
import bitarray
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import scipy.fftpack
import pickle
import time
import os
import io


class JpegCompressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("JPEG Compression Tool")
        self.root.geometry("1100x750")
        
        self.input_image = None
        self.encoded_data = None
        self.decoded_image = None
        self.quality = tk.IntVar(value=50)
        self.force_grayscale = tk.BooleanVar(value=False)
        self.original_file_size = 0
        
        # Create the main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.tab_control = ttk.Notebook(self.main_frame)
        
        self.encode_tab = ttk.Frame(self.tab_control)
        self.decode_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.encode_tab, text="Encode")
        self.tab_control.add(self.decode_tab, text="Decode")
        self.tab_control.pack(expand=1, fill=tk.BOTH)
        
        # Setup encode tab
        self.setup_encode_tab()
        
        # Setup decode tab
        self.setup_decode_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_encode_tab(self):
        # Create frames
        control_frame = ttk.LabelFrame(self.encode_tab, text="Controls", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        image_frame = ttk.LabelFrame(self.encode_tab, text="Images", padding="10")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Controls
        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=5)
        
        # Quality control
        quality_frame = ttk.Frame(control_frame)
        quality_frame.pack(fill=tk.X, pady=10)
        ttk.Label(quality_frame, text="Quality:").pack(side=tk.LEFT)
        ttk.Label(quality_frame, textvariable=self.quality).pack(side=tk.RIGHT)
        
        quality_scale = ttk.Scale(control_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                                 variable=self.quality, command=lambda s: self.quality.set(int(float(s))))
        quality_scale.pack(fill=tk.X, pady=5)
        
        # Grayscale option
        ttk.Checkbutton(control_frame, text="Force Grayscale", variable=self.force_grayscale).pack(anchor=tk.W, pady=5)
        
        # Action buttons
        ttk.Button(control_frame, text="Encode", command=self.encode_action).pack(fill=tk.X, pady=10)
        ttk.Button(control_frame, text="Save Encoded Data", command=self.save_encoded_data).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Save Decoded Image", command=self.save_decoded_image).pack(fill=tk.X, pady=5)
        
        # Information frame
        info_frame = ttk.LabelFrame(control_frame, text="File Information")
        info_frame.pack(fill=tk.X, pady=10)
        
        self.original_size_var = tk.StringVar(value="Original Size: N/A")
        self.encoded_size_var = tk.StringVar(value="Encoded Size: N/A")
        self.compression_ratio_var = tk.StringVar(value="Compression Ratio: N/A")
        self.psnr_var = tk.StringVar(value="PSNR: N/A")
        
        ttk.Label(info_frame, textvariable=self.original_size_var).pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, textvariable=self.encoded_size_var).pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, textvariable=self.compression_ratio_var).pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, textvariable=self.psnr_var).pack(anchor=tk.W, pady=2)
        
        # Image display area
        self.image_display_frame = ttk.Frame(image_frame)
        self.image_display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Original image
        self.original_image_frame = ttk.LabelFrame(self.image_display_frame, text="Original Image")
        self.original_image_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.original_canvas = tk.Canvas(self.original_image_frame)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Decoded image
        self.decoded_image_frame = ttk.LabelFrame(self.image_display_frame, text="Decoded Image")
        self.decoded_image_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        self.decoded_canvas = tk.Canvas(self.decoded_image_frame)
        self.decoded_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.image_display_frame.grid_rowconfigure(0, weight=1)
        self.image_display_frame.grid_columnconfigure(0, weight=1)
        self.image_display_frame.grid_columnconfigure(1, weight=1)

    def setup_decode_tab(self):
        # Create frames
        control_frame = ttk.LabelFrame(self.decode_tab, text="Controls", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        image_frame = ttk.LabelFrame(self.decode_tab, text="Decoded Image", padding="10")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Controls
        ttk.Button(control_frame, text="Load Encoded Data", command=self.load_encoded_data).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Decode", command=self.decode_action).pack(fill=tk.X, pady=10)
        ttk.Button(control_frame, text="Save Decoded Image", command=self.save_decoded_image).pack(fill=tk.X, pady=5)
        
        # Information frame
        info_frame = ttk.LabelFrame(control_frame, text="File Information")
        info_frame.pack(fill=tk.X, pady=10)
        
        self.decode_encoded_size_var = tk.StringVar(value="Encoded Size: N/A")
        self.decode_decoded_size_var = tk.StringVar(value="Decoded Size: N/A")
        self.decode_type_var = tk.StringVar(value="Image Type: N/A")
        self.decode_quality_var = tk.StringVar(value="Quality: N/A")
        
        ttk.Label(info_frame, textvariable=self.decode_encoded_size_var).pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, textvariable=self.decode_decoded_size_var).pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, textvariable=self.decode_type_var).pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, textvariable=self.decode_quality_var).pack(anchor=tk.W, pady=2)
        
        # Image display
        self.decode_canvas = tk.Canvas(image_frame)
        self.decode_canvas.pack(fill=tk.BOTH, expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                self.status_var.set(f"Loading image: {file_path}")
                self.root.update()
                
                # Load image
                self.input_image = self.load_image_from_path(file_path)
                
                # Display image
                self.display_image_on_canvas(self.input_image, self.original_canvas)
                
                # Store the original file size
                self.original_file_size = os.path.getsize(file_path)
                self.original_size_var.set(f"Original Size: {self.format_file_size(self.original_file_size)}")
                
                # Clear decoded image
                self.decoded_image = None
                self.decoded_canvas.delete("all")
                
                # Reset stats
                self.encoded_size_var.set("Encoded Size: N/A")
                self.compression_ratio_var.set("Compression Ratio: N/A")
                self.psnr_var.set("PSNR: N/A")
                
                # Auto-detect grayscale
                if self.is_grayscale_image(self.input_image):
                    self.force_grayscale.set(True)
                
                self.status_var.set(f"Image loaded: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_var.set("Failed to load image")

    def encode_action(self):
        if self.input_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            self.status_var.set("Encoding image...")
            self.root.update()
            
            # Prepare image
            img = self.input_image.copy()
            
            # Convert to grayscale if requested
            if self.force_grayscale.get() and not self.is_grayscale_image(img):
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # Convert RGB to grayscale
                    Y, _, _ = rgb_to_ycbcr(img)
                    img = Y.astype(np.uint8)
            
            # Start timing
            start_time = time.time()
            
            # Encode the image
            self.encoded_data = jpeg_compress(img, quality=self.quality.get())
            
            # End timing
            encode_time = time.time() - start_time
            
            # Decode for display
            start_time = time.time()
            self.decoded_image = jpeg_decompress(self.encoded_data)
            decode_time = time.time() - start_time
            
            # Display decoded image
            self.display_image_on_canvas(self.decoded_image, self.decoded_canvas)
            
            # Calculate stats
            # Use the stored original file size instead of recalculating
            original_size = self.original_file_size
            encoded_size = estimate_compressed_size(self.encoded_data)
            compression_ratio = original_size / encoded_size if encoded_size > 0 else 0
            
            # Calculate PSNR
            psnr = calculate_psnr(img, self.decoded_image)
            
            # Update info using stored original size
            self.original_size_var.set(f"Original Size: {self.format_file_size(original_size)}")
            self.encoded_size_var.set(f"Encoded Size: {self.format_file_size(encoded_size)}")
            self.compression_ratio_var.set(f"Compression Ratio: {compression_ratio:.2f}x")
            self.psnr_var.set(f"PSNR: {psnr:.2f} dB")
            
            self.status_var.set(f"Image encoded in {encode_time:.2f}s and decoded in {decode_time:.2f}s")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to encode image: {str(e)}")
            self.status_var.set("Failed to encode image")
            import traceback
            traceback.print_exc()

    def save_encoded_data(self):
        if self.encoded_data is None:
            messagebox.showwarning("Warning", "Please encode an image first")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Encoded Data",
            defaultextension=".jpgc",
            filetypes=[("JPEG Compressed", "*.jpgc"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                self.status_var.set(f"Saving encoded data to: {file_path}")
                self.root.update()
                
                # Use our new binary format instead of pickle
                save_encoded_data(self.encoded_data, file_path)
                
                # Update file size info
                file_size = os.path.getsize(file_path)
                self.encoded_size_var.set(f"Encoded Size: {self.format_file_size(file_size)}")
                
                # Update compression ratio using stored original size
                compression_ratio = self.original_file_size / file_size
                self.compression_ratio_var.set(f"Compression Ratio: {compression_ratio:.2f}x")
                
                self.status_var.set(f"Encoded data saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save encoded data: {str(e)}")
                self.status_var.set("Failed to save encoded data")
                import traceback
                traceback.print_exc()

    def load_encoded_data(self):
        file_path = filedialog.askopenfilename(
            title="Select Encoded Data File",
            filetypes=[("JPEG Compressed", "*.jpgc"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                self.status_var.set(f"Loading encoded data: {file_path}")
                self.root.update()
                
                # Use our new binary loading function
                self.encoded_data = load_encoded_data(file_path)
                
                # Clear any existing decoded image
                self.decoded_image = None
                
                # Clear canvas in both tabs
                self.decoded_canvas.delete("all")
                self.decode_canvas.delete("all")
                
                # Update file size info
                file_size = os.path.getsize(file_path)
                self.decode_encoded_size_var.set(f"Encoded Size: {self.format_file_size(file_size)}")
                
                # Update metadata info
                is_gray = 'is_grayscale' in self.encoded_data and self.encoded_data['is_grayscale']
                self.decode_type_var.set(f"Image Type: {'Grayscale' if is_gray else 'Color'}")
                
                if 'quality' in self.encoded_data:
                    self.decode_quality_var.set(f"Quality: {self.encoded_data['quality']}")
                
                # Reset decoded size info
                self.decode_decoded_size_var.set("Decoded Size: N/A")
                
                self.status_var.set(f"Encoded data loaded: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load encoded data: {str(e)}")
                self.status_var.set("Failed to load encoded data")
                import traceback
                traceback.print_exc()

    def decode_action(self):
        if self.encoded_data is None:
            messagebox.showwarning("Warning", "Please load encoded data first")
            return
        
        try:
            self.status_var.set("Decoding image...")
            self.root.update()
            
            # Start timing
            start_time = time.time()
            
            # Debug info
            print(f"Decoding data: {'grayscale' if 'is_grayscale' in self.encoded_data and self.encoded_data['is_grayscale'] else 'color'}")
            if 'is_grayscale' in self.encoded_data and self.encoded_data['is_grayscale']:
                print(f"Gray RLE blocks: {len(self.encoded_data['gray_rle'])}")
            else:
                print(f"Y RLE blocks: {len(self.encoded_data['Y_rle'])}")
                print(f"Cb RLE blocks: {len(self.encoded_data['Cb_rle'])}")
                print(f"Cr RLE blocks: {len(self.encoded_data['Cr_rle'])}")
        
            # Decode the image
            self.decoded_image = jpeg_decompress(self.encoded_data)
            
            # End timing
            decode_time = time.time() - start_time
            
            # Display decoded image - use correct canvas based on active tab
            if self.tab_control.select() == str(self.encode_tab):
                self.display_image_on_canvas(self.decoded_image, self.decoded_canvas)
            else:
                self.display_image_on_canvas(self.decoded_image, self.decode_canvas)
            
            # Update decoded size
            decoded_size = self.decoded_image.size * self.decoded_image.itemsize
            self.decode_decoded_size_var.set(f"Decoded Size: {self.format_file_size(decoded_size)}")
            
            self.status_var.set(f"Image decoded in {decode_time:.2f}s")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to decode image: {str(e)}")
            self.status_var.set("Failed to decode image")
            import traceback
            traceback.print_exc()

    def save_decoded_image(self):
        if self.decoded_image is None:
            messagebox.showwarning("Warning", "Please encode or decode an image first")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Decoded Image",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                self.status_var.set(f"Saving decoded image to: {file_path}")
                self.root.update()
                
                # Convert to PIL image
                if len(self.decoded_image.shape) == 2:
                    # Grayscale
                    pil_image = Image.fromarray(self.decoded_image.astype(np.uint8), 'L')
                else:
                    # RGB
                    pil_image = Image.fromarray(self.decoded_image.astype(np.uint8), 'RGB')
                
                pil_image.save(file_path)
                
                self.status_var.set(f"Decoded image saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save decoded image: {str(e)}")
                self.status_var.set("Failed to save decoded image")

    def load_image_from_path(self, path):
        img = Image.open(path)
        return np.array(img)

    def is_grayscale_image(self, img):
        return is_grayscale(img)

    def display_image_on_canvas(self, img, canvas):
        canvas.delete("all")
        
        if img is None:
            return
        
        # Convert to PIL image
        if len(img.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(img.astype(np.uint8), 'L')
        else:
            # RGB
            pil_image = Image.fromarray(img.astype(np.uint8), 'RGB')
        
        # Resize image for display if needed
        canvas_width = canvas.winfo_width() or 400
        canvas_height = canvas.winfo_height() or 400
        
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        photo_image = ImageTk.PhotoImage(pil_image)
        
        # Keep a reference to avoid garbage collection
        canvas.image = photo_image
        
        # Display image
        canvas.create_image(canvas_width//2, canvas_height//2, image=photo_image, anchor=tk.CENTER)

    @staticmethod
    def format_file_size(size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0 or unit == 'GB':
                break
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} {unit}"

# ====================================
# JPEG Compression Algorithm Functions
# ====================================

def is_grayscale(img):
    if len(img.shape) == 2:
        return True
    elif len(img.shape) == 3 and img.shape[2] == 1:
        return True
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # Check if all color channels are equal
        return np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,0], img[:,:,2])
    return False

def rgb_to_ycbcr(img):
    # Make sure we're working with float values for precision
    img = img.astype(np.float32)
    
    # RGB to YCbCr conversion matrix
    transform = np.array([
        [ 0.299,  0.587,  0.114],
        [-0.169, -0.331,  0.500],
        [ 0.500, -0.419, -0.081]
    ])
    
    # Offset for Cb and Cr components
    offset = np.array([0, 128, 128])
    
    # Reshape the image to a 2D array of pixels
    height, width, _ = img.shape
    pixels = img.reshape(-1, 3)
    
    # Apply the transformation
    ycbcr_pixels = np.dot(pixels, transform.T) + offset
    
    # Reshape back to the original image shape
    ycbcr = ycbcr_pixels.reshape(height, width, 3)
    
    # Extract Y, Cb, Cr components
    Y = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 1]
    Cr = ycbcr[:, :, 2]
    
    return Y, Cb, Cr

def ycbcr_to_rgb(Y, Cb, Cr):
    # Reshape components to have the same dimensions
    height, width = Y.shape
    Y_reshaped = Y.reshape(height, width, 1)
    Cb_reshaped = Cb.reshape(height, width, 1)
    Cr_reshaped = Cr.reshape(height, width, 1)
    
    # Combine into a single array
    ycbcr = np.concatenate([Y_reshaped, Cb_reshaped, Cr_reshaped], axis=2)
    
    # YCbCr to RGB inverse transformation
    transform = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344, -0.714],
        [1.0, 1.772, 0.0]
    ])
    
    # Offset for Cb and Cr components
    offset = np.array([0, -128, -128])
    
    # Reshape to apply transformation
    pixels = ycbcr.reshape(-1, 3)
    
    # Apply the inverse transformation
    rgb_pixels = np.dot(pixels + offset, transform.T)
    
    # Clip values to stay in 0-255 range
    rgb_pixels = np.clip(rgb_pixels, 0, 255)
    
    # Reshape back to original dimensions and convert to uint8
    rgb = rgb_pixels.reshape(height, width, 3).astype(np.uint8)
    
    return rgb

def downsample_chroma(cb, cr):
    height, width = cb.shape
    # Ensure even dimensions
    height_even = height - (height % 2)
    width_even = width - (width % 2)
    
    # Reshape to group 2x2 blocks and take their average
    cb_view = cb[:height_even, :width_even].reshape(height_even//2, 2, width_even//2, 2)
    cr_view = cr[:height_even, :width_even].reshape(height_even//2, 2, width_even//2, 2)
    
    cb_downsampled = cb_view.mean(axis=(1, 3))
    cr_downsampled = cr_view.mean(axis=(1, 3))
    
    return cb_downsampled, cr_downsampled

def upsample_chroma(cb_downsampled, cr_downsampled, original_shape):
    # Create output arrays
    height, width = original_shape
    cb_upsampled = np.zeros(original_shape, dtype=np.float32)
    cr_upsampled = np.zeros(original_shape, dtype=np.float32)
    
    # Simple nearest-neighbor upsampling
    for i in range(height):
        for j in range(width):
            # Find corresponding position in downsampled image
            ds_i = min(i // 2, cb_downsampled.shape[0] - 1)
            ds_j = min(j // 2, cb_downsampled.shape[1] - 1)
            
            cb_upsampled[i, j] = cb_downsampled[ds_i, ds_j]
            cr_upsampled[i, j] = cr_downsampled[ds_i, ds_j]
    
    return cb_upsampled, cr_upsampled

def pad_to_multiple_of_8(channel):
    height, width = channel.shape
    new_height = height + (8 - height % 8) % 8
    new_width = width + (8 - width % 8) % 8
    
    padded = np.zeros((new_height, new_width), dtype=channel.dtype)
    padded[:height, :width] = channel
    
    return padded, (height, width)

def split_into_blocks(channel):
    # Pad the channel to have dimensions that are multiples of 8
    padded_channel, original_shape = pad_to_multiple_of_8(channel)
    
    # Get the padded dimensions
    height, width = padded_channel.shape
    
    # Calculate number of blocks in each dimension
    num_blocks_v = height // 8
    num_blocks_h = width // 8
    
    # Create a 4D array view of the image, where each element is an 8x8 block
    blocks = padded_channel.reshape(num_blocks_v, 8, num_blocks_h, 8)
    blocks = blocks.transpose(0, 2, 1, 3)
    
    return blocks, original_shape

def combine_blocks(blocks, original_shape):
    # Get number of blocks in each dimension
    num_blocks_v, num_blocks_h = blocks.shape[0:2]
    
    # Calculate padded dimensions
    height_padded = num_blocks_v * 8
    width_padded = num_blocks_h * 8
    
    # Reshape blocks back to a 2D array
    blocks_transposed = blocks.transpose(0, 2, 1, 3)
    channel_padded = blocks_transposed.reshape(height_padded, width_padded)
    
    # Crop to original dimensions
    height, width = original_shape
    channel = channel_padded[:height, :width]
    
    return channel

def apply_dct(block):
    # Shift values from [0, 255] to [-128, 127] before DCT
    shifted_block = block - 128.0
    
    # Apply 2D DCT
    dct_block = scipy.fftpack.dct(scipy.fftpack.dct(shifted_block, axis=0, norm='ortho'), 
                            axis=1, norm='ortho')
    
    return dct_block

def apply_idct(dct_block):
    # Apply 2D IDCT
    idct_block = scipy.fftpack.idct(scipy.fftpack.idct(dct_block, axis=0, norm='ortho'), 
                              axis=1, norm='ortho')
    
    # Shift values back to [0, 255] range
    block = idct_block + 128.0
    
    # Clip values to valid range
    block = np.clip(block, 0, 255)
    
    return block

def apply_dct_to_blocks(blocks):
    num_blocks_v, num_blocks_h = blocks.shape[0:2]
    dct_blocks = np.zeros_like(blocks, dtype=np.float32)
    
    for v in range(num_blocks_v):
        for h in range(num_blocks_h):
            dct_blocks[v, h] = apply_dct(blocks[v, h])
    
    return dct_blocks

def apply_idct_to_blocks(dct_blocks):
    num_blocks_v, num_blocks_h = dct_blocks.shape[0:2]
    blocks = np.zeros_like(dct_blocks, dtype=np.float32)
    
    for v in range(num_blocks_v):
        for h in range(num_blocks_h):
            blocks[v, h] = apply_idct(dct_blocks[v, h])
    
    return blocks

def get_quantization_tables(quality=50):
    # Standard JPEG quantization tables
    y_table_base = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    c_table_base = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])
    
    if quality < 50:
        scaling_factor = 5000 / quality
    else:
        scaling_factor = 200 - 2 * quality
    
    # Scale the tables according to quality
    y_table = np.floor((y_table_base * scaling_factor + 50) / 100)
    c_table = np.floor((c_table_base * scaling_factor + 50) / 100)
    
    # Ensure minimum value of 1
    y_table = np.clip(y_table, 1, 255).astype(np.uint8)
    c_table = np.clip(c_table, 1, 255).astype(np.uint8)
    
    return y_table, c_table

def quantize_block(dct_block, q_table):

    return np.round(dct_block / q_table).astype(np.int32)

def dequantize_block(quantized_block, q_table):
    return quantized_block * q_table

def quantize_channel_blocks(dct_blocks, q_table):
    num_blocks_v, num_blocks_h = dct_blocks.shape[0:2]
    quantized_blocks = np.zeros_like(dct_blocks, dtype=np.int32)
    
    for v in range(num_blocks_v):
        for h in range(num_blocks_h):
            quantized_blocks[v, h] = quantize_block(dct_blocks[v, h], q_table)
    
    return quantized_blocks

def dequantize_channel_blocks(quantized_blocks, q_table):
    num_blocks_v, num_blocks_h = quantized_blocks.shape[0:2]
    dct_blocks = np.zeros_like(quantized_blocks, dtype=np.float32)
    
    for v in range(num_blocks_v):
        for h in range(num_blocks_h):
            dct_blocks[v, h] = dequantize_block(quantized_blocks[v, h], q_table)
    
    return dct_blocks

def zigzag_scan(block):
    # Create a zigzag pattern of indices
    zigzag_indices = np.array([
        0,  1,  8, 16,  9,  2,  3, 10,
       17, 24, 32, 25, 18, 11,  4,  5,
       12, 19, 26, 33, 40, 48, 41, 34,
       27, 20, 13,  6,  7, 14, 21, 28,
       35, 42, 49, 56, 57, 50, 43, 36,
       29, 22, 15, 23, 30, 37, 44, 51,
       58, 59, 52, 45, 38, 31, 39, 46,
       53, 60, 61, 54, 47, 55, 62, 63
    ])
    
    # Flatten the block
    flat_block = block.flatten()
    
    # Create the zigzag sequence
    zigzag = np.zeros_like(flat_block)
    for i, idx in enumerate(zigzag_indices):
        zigzag[i] = flat_block[idx]
    
    return zigzag

def inverse_zigzag_scan(zigzag):

    # Create a zigzag pattern of indices
    zigzag_indices = np.array([
        0,  1,  8, 16,  9,  2,  3, 10,
       17, 24, 32, 25, 18, 11,  4,  5,
       12, 19, 26, 33, 40, 48, 41, 34,
       27, 20, 13,  6,  7, 14, 21, 28,
       35, 42, 49, 56, 57, 50, 43, 36,
       29, 22, 15, 23, 30, 37, 44, 51,
       58, 59, 52, 45, 38, 31, 39, 46,
       53, 60, 61, 54, 47, 55, 62, 63
    ])
    
    # Create an empty flattened block
    flat_block = np.zeros_like(zigzag)
    
    # Fill the flattened block using the zigzag sequence
    for i, idx in enumerate(zigzag_indices):
        flat_block[idx] = zigzag[i]
    
    # Reshape to an 8x8 block
    block = flat_block.reshape(8, 8)
    
    return block

def rle_encode(zigzag):
    rle = []
    
    # First coefficient (DC) is directly stored
    rle.append((0, zigzag[0]))
    
    # Encode AC coefficients (all the rest)
    i = 1
    while i < len(zigzag):
        # Count zeros
        run_length = 0
        while i < len(zigzag) and zigzag[i] == 0:
            run_length += 1
            i += 1
        
        # If we reached the end, add EOB marker
        if i >= len(zigzag):
            rle.append((0, 0))  # End of block marker
            break
        
        # Store the run length and the non-zero value
        rle.append((run_length, zigzag[i]))
        i += 1
    
    return rle

def rle_decode(rle, length=64):
    zigzag = np.zeros(length, dtype=int)
    
    # First coefficient is DC
    zigzag[0] = rle[0][1]
    
    # Decode AC coefficients
    pos = 1
    for i in range(1, len(rle)):
        run_length, value = rle[i]
        
        # If EOB marker, break
        if run_length == 0 and value == 0:
            break
        
        # Skip zeros
        pos += run_length
        
        # Add non-zero value
        if pos < length:
            zigzag[pos] = value
        
        pos += 1
    
    return zigzag

def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def estimate_compressed_size(encoded_data):
    if 'is_grayscale' in encoded_data and encoded_data['is_grayscale']:
        # Handle grayscale case
        gray_pairs = sum(len(block) for block in encoded_data['gray_rle'])
        # Extract data for Huffman coding simulation
        flattened_data = []
        dc_coeffs = [block[0][1] for block in encoded_data['gray_rle']]
        diff_dc = dc_differential_encode(dc_coeffs)
        
        # Apply differential coding to DC coefficients
        for i, block in enumerate(encoded_data['gray_rle']):
            for j, (run, val) in enumerate(block):
                if j == 0:  # DC coefficient
                    flattened_data.append(run)
                    flattened_data.append(diff_dc[i])
                else:  # AC coefficient
                    flattened_data.append(run)
                    flattened_data.append(val)
        
        # Estimate Huffman coding size
        _, encoded_bits, _ = huffman_encode(flattened_data)
        bit_size = len(encoded_bits)
        byte_size = (bit_size + 7) // 8
        
        # Add header size: signature + grayscale flag + quality + shape + qtable + huffman table
        header_size = 4 + 1 + 1 + 5 + 4 + 64 + (4 + 8 * len(set(flattened_data))) + 5
        
        estimated_size = byte_size + header_size
    else:
        # Handle color case - similar approach for each channel
        # Extract data for Huffman coding simulation
        y_flattened = []
        cb_flattened = []
        cr_flattened = []
        
        # Y channel with differential coding
        y_dc = [block[0][1] for block in encoded_data['Y_rle']]
        y_diff = dc_differential_encode(y_dc)
        
        for i, block in enumerate(encoded_data['Y_rle']):
            for j, (run, val) in enumerate(block):
                if j == 0:  # DC coefficient
                    y_flattened.append(run)
                    y_flattened.append(y_diff[i])
                else:  # AC coefficient
                    y_flattened.append(run)
                    y_flattened.append(val)
        
        # Cb channel with differential coding
        cb_dc = [block[0][1] for block in encoded_data['Cb_rle']]
        cb_diff = dc_differential_encode(cb_dc)
        
        for i, block in enumerate(encoded_data['Cb_rle']):
            for j, (run, val) in enumerate(block):
                if j == 0:  # DC coefficient
                    cb_flattened.append(run)
                    cb_flattened.append(cb_diff[i])
                else:  # AC coefficient
                    cb_flattened.append(run)
                    cb_flattened.append(val)
        
        # Cr channel with differential coding
        cr_dc = [block[0][1] for block in encoded_data['Cr_rle']]
        cr_diff = dc_differential_encode(cr_dc)
        
        for i, block in enumerate(encoded_data['Cr_rle']):
            for j, (run, val) in enumerate(block):
                if j == 0:  # DC coefficient
                    cr_flattened.append(run)
                    cr_flattened.append(cr_diff[i])
                else:  # AC coefficient
                    cr_flattened.append(run)
                    cr_flattened.append(val)
        
        # Estimate Huffman coding size for each channel
        _, y_bits, _ = huffman_encode(y_flattened)
        _, cb_bits, _ = huffman_encode(cb_flattened)
        _, cr_bits, _ = huffman_encode(cr_flattened)
        
        y_byte_size = (len(y_bits) + 7) // 8
        cb_byte_size = (len(cb_bits) + 7) // 8
        cr_byte_size = (len(cr_bits) + 7) // 8
        
        # Add header size
        header_size = 4 + 1 + 1 + 5 + 12 + 128  # Signature, flags, shape, quantization tables
        
        # Add Huffman tables size
        y_symbols = len(set(y_flattened))
        cb_symbols = len(set(cb_flattened))
        cr_symbols = len(set(cr_flattened))
        
        huffman_tables_size = (4 + 8 * y_symbols + 5) + (4 + 8 * cb_symbols + 5) + (4 + 8 * cr_symbols + 5)
        
        estimated_size = y_byte_size + cb_byte_size + cr_byte_size + header_size + huffman_tables_size
    
    return estimated_size

def grayscale_jpeg_encode(image, quality=50):
    # Ensure the image is grayscale
    gray = image
        
    # Make sure we have a 2D array
    if len(gray.shape) == 3:
        gray = gray[:,:,0]
    
    # Step 1: Split into 8x8 blocks
    gray_blocks, gray_shape = split_into_blocks(gray)
    
    # Step 2: Apply DCT to each block
    gray_dct_blocks = apply_dct_to_blocks(gray_blocks)
    
    # Step 3: Quantization (using luminance quantization table)
    y_qtable, _ = get_quantization_tables(quality)
    gray_quantized_blocks = quantize_channel_blocks(gray_dct_blocks, y_qtable)
    
    # Step 4: Zigzag scan and RLE
    gray_rle = []
    
    # Process grayscale channel
    for v in range(gray_quantized_blocks.shape[0]):
        for h in range(gray_quantized_blocks.shape[1]):
            zigzagged = zigzag_scan(gray_quantized_blocks[v, h])
            gray_rle.append(rle_encode(zigzagged))
    
    # Collect metadata for decoding
    encoded_data = {
        'gray_rle': gray_rle,
        'gray_shape': gray_shape,
        'y_qtable': y_qtable,
        'quality': quality,
        'original_shape': image.shape,
        'is_grayscale': True
    }
    
    return encoded_data

def grayscale_jpeg_decode(encoded_data):
    # Extract data and metadata
    gray_rle = encoded_data['gray_rle']
    gray_shape = encoded_data['gray_shape']
    y_qtable = encoded_data['y_qtable']
    original_shape = encoded_data['original_shape']
    
    # Step 1: RLE decode and inverse zigzag scan
    gray_blocks_shape = ((gray_shape[0] + 7) // 8, (gray_shape[1] + 7) // 8, 8, 8)
    gray_quantized_blocks = np.zeros(gray_blocks_shape, dtype=np.int32)
    
    # Process grayscale channel
    block_idx = 0
    for v in range(gray_blocks_shape[0]):
        for h in range(gray_blocks_shape[1]):
            rle_data = gray_rle[block_idx]
            zigzagged = rle_decode(rle_data)
            gray_quantized_blocks[v, h] = inverse_zigzag_scan(zigzagged)
            block_idx += 1
    
    # Step 2: Dequantization
    gray_dct_blocks = dequantize_channel_blocks(gray_quantized_blocks, y_qtable)
    
    # Step 3: Apply IDCT
    gray_blocks = apply_idct_to_blocks(gray_dct_blocks)
    
    # Step 4: Combine blocks
    gray_reconstructed = combine_blocks(gray_blocks, gray_shape)
    
    # Ensure the output has the correct shape
    if len(original_shape) == 3 and original_shape[2] == 1:
        # Original was 3D with single channel
        decoded_image = gray_reconstructed[:original_shape[0], :original_shape[1]].reshape(original_shape)
    else:
        # Original was 2D
        decoded_image = gray_reconstructed[:original_shape[0], :original_shape[1]]
    
    return decoded_image

def jpeg_encode(image, quality=50):
    # Step 1: Convert RGB to YCbCr
    Y, Cb, Cr = rgb_to_ycbcr(image)
    
    # Step 2: Chroma downsampling
    Cb_downsampled, Cr_downsampled = downsample_chroma(Cb, Cr)
    
    # Step 3: Split into 8x8 blocks
    Y_blocks, Y_shape = split_into_blocks(Y)
    Cb_blocks, Cb_shape = split_into_blocks(Cb_downsampled)
    Cr_blocks, Cr_shape = split_into_blocks(Cr_downsampled)
    
    # Step 4: Apply DCT to each block
    Y_dct_blocks = apply_dct_to_blocks(Y_blocks)
    Cb_dct_blocks = apply_dct_to_blocks(Cb_blocks)
    Cr_dct_blocks = apply_dct_to_blocks(Cr_blocks)
    
    # Step 5: Quantization
    y_qtable, c_qtable = get_quantization_tables(quality)
    Y_quantized_blocks = quantize_channel_blocks(Y_dct_blocks, y_qtable)
    Cb_quantized_blocks = quantize_channel_blocks(Cb_dct_blocks, c_qtable)
    Cr_quantized_blocks = quantize_channel_blocks(Cr_dct_blocks, c_qtable)
    
    # Step 6 & 7: Zigzag scan and RLE
    Y_rle = []
    Cb_rle = []
    Cr_rle = []
    
    # Process Y channel
    for v in range(Y_quantized_blocks.shape[0]):
        for h in range(Y_quantized_blocks.shape[1]):
            zigzagged = zigzag_scan(Y_quantized_blocks[v, h])
            Y_rle.append(rle_encode(zigzagged))
    
    # Process Cb channel
    for v in range(Cb_quantized_blocks.shape[0]):
        for h in range(Cb_quantized_blocks.shape[1]):
            zigzagged = zigzag_scan(Cb_quantized_blocks[v, h])
            Cb_rle.append(rle_encode(zigzagged))
    
    # Process Cr channel
    for v in range(Cr_quantized_blocks.shape[0]):
        for h in range(Cr_quantized_blocks.shape[1]):
            zigzagged = zigzag_scan(Cr_quantized_blocks[v, h])
            Cr_rle.append(rle_encode(zigzagged))
    
    # Collect metadata for decoding
    encoded_data = {
        'Y_rle': Y_rle,
        'Cb_rle': Cb_rle,
        'Cr_rle': Cr_rle,
        'Y_shape': Y_shape,
        'Cb_shape': Cb_shape,
        'Cr_shape': Cr_shape,
        'y_qtable': y_qtable,
        'c_qtable': c_qtable,
        'quality': quality,
        'original_shape': image.shape
    }
    
    return encoded_data

def jpeg_decode(encoded_data):
    # Extract data and metadata
    Y_rle = encoded_data['Y_rle']
    Cb_rle = encoded_data['Cb_rle']
    Cr_rle = encoded_data['Cr_rle']
    Y_shape = encoded_data['Y_shape']
    Cb_shape = encoded_data['Cb_shape']
    Cr_shape = encoded_data['Cr_shape']
    y_qtable = encoded_data['y_qtable']
    c_qtable = encoded_data['c_qtable']
    original_shape = encoded_data['original_shape']
    
    # Step 1: RLE decode and inverse zigzag scan
    Y_blocks_shape = (Y_shape[0] + 7) // 8, (Y_shape[1] + 7) // 8, 8, 8
    Cb_blocks_shape = (Cb_shape[0] + 7) // 8, (Cb_shape[1] + 7) // 8, 8, 8
    Cr_blocks_shape = (Cr_shape[0] + 7) // 8, (Cr_shape[1] + 7) // 8, 8, 8
    
    Y_quantized_blocks = np.zeros(Y_blocks_shape, dtype=np.int32)
    Cb_quantized_blocks = np.zeros(Cb_blocks_shape, dtype=np.int32)
    Cr_quantized_blocks = np.zeros(Cr_blocks_shape, dtype=np.int32)
    
    # Process Y channel
    block_idx = 0
    for v in range(Y_blocks_shape[0]):
        for h in range(Y_blocks_shape[1]):
            rle_data = Y_rle[block_idx]
            zigzagged = rle_decode(rle_data)
            Y_quantized_blocks[v, h] = inverse_zigzag_scan(zigzagged)
            block_idx += 1
    
    # Process Cb channel
    block_idx = 0
    for v in range(Cb_blocks_shape[0]):
        for h in range(Cb_blocks_shape[1]):
            rle_data = Cb_rle[block_idx]
            zigzagged = rle_decode(rle_data)
            Cb_quantized_blocks[v, h] = inverse_zigzag_scan(zigzagged)
            block_idx += 1
    
    # Process Cr channel
    block_idx = 0
    for v in range(Cr_blocks_shape[0]):
        for h in range(Cr_blocks_shape[1]):
            rle_data = Cr_rle[block_idx]
            zigzagged = rle_decode(rle_data)
            Cr_quantized_blocks[v, h] = inverse_zigzag_scan(zigzagged)
            block_idx += 1
    
    # Step 2: Dequantization
    Y_dct_blocks = dequantize_channel_blocks(Y_quantized_blocks, y_qtable)
    Cb_dct_blocks = dequantize_channel_blocks(Cb_quantized_blocks, c_qtable)
    Cr_dct_blocks = dequantize_channel_blocks(Cr_quantized_blocks, c_qtable)
    
    # Step 3: Apply IDCT
    Y_blocks = apply_idct_to_blocks(Y_dct_blocks)
    Cb_blocks = apply_idct_to_blocks(Cb_dct_blocks)
    Cr_blocks = apply_idct_to_blocks(Cr_dct_blocks)
    
    # Step 4: Combine blocks
    Y_reconstructed = combine_blocks(Y_blocks, Y_shape)
    Cb_downsampled = combine_blocks(Cb_blocks, Cb_shape)
    Cr_downsampled = combine_blocks(Cr_blocks, Cr_shape)
    
    # Step 5: Chroma upsampling
    Cb_upsampled, Cr_upsampled = upsample_chroma(Cb_downsampled, Cr_downsampled, Y_shape)
    
    # Step 6: Convert YCbCr to RGB
    decoded_image = ycbcr_to_rgb(Y_reconstructed, Cb_upsampled, Cr_upsampled)
    
    # Crop to original dimensions
    decoded_image = decoded_image[:original_shape[0], :original_shape[1], :original_shape[2]]
    
    return decoded_image

def jpeg_compress(image, quality=50):

    if is_grayscale(image):
        return grayscale_jpeg_encode(image, quality)
    else:
        return jpeg_encode(image, quality)

def jpeg_decompress(encoded_data):
    if 'is_grayscale' in encoded_data and encoded_data['is_grayscale']:
        return grayscale_jpeg_decode(encoded_data)
    else:
        return jpeg_decode(encoded_data)

def convert_to_grayscale(img):
    if is_grayscale(img):
        if len(img.shape) == 3:
            return img[:,:,0]  # Take just one channel if it's already grayscale
        return img
    else:
        # Use the Y component of YCbCr as grayscale representation
        Y, _, _ = rgb_to_ycbcr(img)
        return Y


def build_huffman_tree(frequencies):
    heap = [[freq, i, [sym, ""]] for i, (sym, freq) in enumerate(frequencies.items())]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        
        for pair in lo[2:]:
            pair[1] = '0' + pair[1]
        for pair in hi[2:]:
            pair[1] = '1' + pair[1]
        
        node = [lo[0] + hi[0], lo[1] + hi[1]] + lo[2:] + hi[2:]
        heapq.heappush(heap, node)
    
    return sorted(heapq.heappop(heap)[2:], key=lambda p: (len(p[-1]), p))

def huffman_encode(data):
    # Count frequencies
    frequencies = Counter(data)
    
    # Edge case for single symbol
    if len(frequencies) == 1:
        symbol = list(frequencies.keys())[0]
        return {symbol: '0'}, bitarray.bitarray('0') * len(data), frequencies
    
    # Build Huffman tree and get codes
    huffman_tree = build_huffman_tree(frequencies)
    huffman_codes = {sym: code for sym, code in huffman_tree}
    
    # Encode the data
    ba = bitarray.bitarray()
    for symbol in data:
        ba.extend(bitarray.bitarray(huffman_codes[symbol]))
    
    return huffman_codes, ba, frequencies

def huffman_decode(encoded_bits, huffman_codes):
    # Create reverse lookup dictionary
    reverse_codes = {code: sym for sym, code in huffman_codes.items()}
    
    decoded_data = []
    code_so_far = ""
    
    for bit in encoded_bits:
        code_so_far += '1' if bit else '0'
        if code_so_far in reverse_codes:
            decoded_data.append(reverse_codes[code_so_far])
            code_so_far = ""
    
    return decoded_data

def dc_differential_encode(dc_coefficients):
    if not dc_coefficients:
        return []
    
    diff_coeffs = [dc_coefficients[0]]  # First coefficient is unchanged
    for i in range(1, len(dc_coefficients)):
        diff_coeffs.append(dc_coefficients[i] - dc_coefficients[i-1])
    
    return diff_coeffs

def dc_differential_decode(diff_coeffs):
    if not diff_coeffs:
        return []
    
    dc_coeffs = [diff_coeffs[0]]  # First coefficient is unchanged
    for i in range(1, len(diff_coeffs)):
        dc_coeffs.append(dc_coeffs[i-1] + diff_coeffs[i])
    
    return dc_coeffs

def save_encoded_data(encoded_data, file_path):
    with open(file_path, 'wb') as f:
        # Write file signature
        f.write(b'JPGC')
        
        # Write metadata flags
        is_grayscale = 'is_grayscale' in encoded_data and encoded_data['is_grayscale']
        f.write(struct.pack('B', 1 if is_grayscale else 0))
        
        # Write quality
        f.write(struct.pack('B', encoded_data['quality']))
        
        # Write original shape
        height, width = encoded_data['original_shape'][:2]
        channels = 1 if is_grayscale else encoded_data['original_shape'][2]
        f.write(struct.pack('>HHB', height, width, channels))
        
        if is_grayscale:
            # 1. Extract DC coefficients
            dc_coeffs = [block[0][1] for block in encoded_data['gray_rle']]
            
            # 2. Apply differential coding
            diff_dc = dc_differential_encode(dc_coeffs)
            
            # 3. Replace DC values in RLE data
            modified_rle = []
            for i, block in enumerate(encoded_data['gray_rle']):
                block_mod = list(block)
                block_mod[0] = (block_mod[0][0], diff_dc[i])
                modified_rle.append(block_mod)
            
            # 4. Prepare data for huffman coding
            flattened_data = []
            for block in modified_rle:
                for run, value in block:
                    # Store run length followed by value
                    flattened_data.append(run)
                    flattened_data.append(value)
            
            # 5. Apply Huffman coding
            huffman_codes, encoded_bits, freq = huffman_encode(flattened_data)
            
            # Write shape and quantization table
            f.write(struct.pack('>HH', *encoded_data['gray_shape']))
            qt_flattened = encoded_data['y_qtable'].flatten().astype(np.uint8)
            f.write(qt_flattened.tobytes())
            
            # Write number of Huffman table entries
            f.write(struct.pack('>I', len(freq)))
            
            # Write Huffman table
            for symbol, frequency in freq.items():
                f.write(struct.pack('>ii', symbol, frequency))
            
            # Write encoded data length (in bits)
            f.write(struct.pack('>I', len(encoded_bits)))
            
            # Write padding bits count
            padding_bits = (8 - len(encoded_bits) % 8) % 8
            f.write(struct.pack('B', padding_bits))
            
            # Write encoded bit data
            f.write(encoded_bits.tobytes())
            
        else:
            # Handle color channels separately with the same approach
            # 1. Y channel
            y_dc = [block[0][1] for block in encoded_data['Y_rle']]
            y_dc_diff = dc_differential_encode(y_dc)
            
            y_modified = []
            for i, block in enumerate(encoded_data['Y_rle']):
                block_mod = list(block)
                block_mod[0] = (block_mod[0][0], y_dc_diff[i])
                y_modified.append(block_mod)
            
            y_flattened = []
            for block in y_modified:
                for run, value in block:
                    y_flattened.append(run)
                    y_flattened.append(value)
            
            y_codes, y_bits, y_freq = huffman_encode(y_flattened)
            
            # 2. Cb channel
            cb_dc = [block[0][1] for block in encoded_data['Cb_rle']]
            cb_dc_diff = dc_differential_encode(cb_dc)
            
            cb_modified = []
            for i, block in enumerate(encoded_data['Cb_rle']):
                block_mod = list(block)
                block_mod[0] = (block_mod[0][0], cb_dc_diff[i])
                cb_modified.append(block_mod)
            
            cb_flattened = []
            for block in cb_modified:
                for run, value in block:
                    cb_flattened.append(run)
                    cb_flattened.append(value)
            
            cb_codes, cb_bits, cb_freq = huffman_encode(cb_flattened)
            
            # 3. Cr channel
            cr_dc = [block[0][1] for block in encoded_data['Cr_rle']]
            cr_dc_diff = dc_differential_encode(cr_dc)
            
            cr_modified = []
            for i, block in enumerate(encoded_data['Cr_rle']):
                block_mod = list(block)
                block_mod[0] = (block_mod[0][0], cr_dc_diff[i])
                cr_modified.append(block_mod)
            
            cr_flattened = []
            for block in cr_modified:
                for run, value in block:
                    cr_flattened.append(run)
                    cr_flattened.append(value)
            
            cr_codes, cr_bits, cr_freq = huffman_encode(cr_flattened)
            
            # Write channel shapes
            f.write(struct.pack('>HH', *encoded_data['Y_shape']))
            f.write(struct.pack('>HH', *encoded_data['Cb_shape']))
            f.write(struct.pack('>HH', *encoded_data['Cr_shape']))
            
            # Write quantization tables
            y_qt = encoded_data['y_qtable'].flatten().astype(np.uint8)
            c_qt = encoded_data['c_qtable'].flatten().astype(np.uint8)
            f.write(y_qt.tobytes())
            f.write(c_qt.tobytes())
            
            # Write Y channel data
            f.write(struct.pack('>I', len(y_freq)))
            for symbol, frequency in y_freq.items():
                f.write(struct.pack('>ii', symbol, frequency))
            
            y_padding = (8 - len(y_bits) % 8) % 8
            f.write(struct.pack('>IB', len(y_bits), y_padding))
            f.write(y_bits.tobytes())
            
            # Write Cb channel data
            f.write(struct.pack('>I', len(cb_freq)))
            for symbol, frequency in cb_freq.items():
                f.write(struct.pack('>ii', symbol, frequency))
            
            cb_padding = (8 - len(cb_bits) % 8) % 8
            f.write(struct.pack('>IB', len(cb_bits), cb_padding))
            f.write(cb_bits.tobytes())
            
            # Write Cr channel data
            f.write(struct.pack('>I', len(cr_freq)))
            for symbol, frequency in cr_freq.items():
                f.write(struct.pack('>ii', symbol, frequency))
            
            cr_padding = (8 - len(cr_bits) % 8) % 8
            f.write(struct.pack('>IB', len(cr_bits), cr_padding))
            f.write(cr_bits.tobytes())

def load_encoded_data(file_path):
    with open(file_path, 'rb') as f:
        # Check signature
        signature = f.read(4)
        if signature != b'JPGC':
            raise ValueError("Invalid file format")
        
        # Read metadata
        is_grayscale = bool(struct.unpack('B', f.read(1))[0])
        quality = struct.unpack('B', f.read(1))[0]
        
        # Read original shape
        height, width, channels = struct.unpack('>HHB', f.read(5))
        original_shape = (height, width) if is_grayscale else (height, width, channels)
        
        if is_grayscale:
            # Read grayscale data
            gray_height, gray_width = struct.unpack('>HH', f.read(4))
            gray_shape = (gray_height, gray_width)
            
            # Read quantization table
            y_qtable = np.frombuffer(f.read(64), dtype=np.uint8).reshape(8, 8)
            
            # Read Huffman table
            n_symbols = struct.unpack('>I', f.read(4))[0]
            freq = {}
            for _ in range(n_symbols):
                symbol, frequency = struct.unpack('>ii', f.read(8))
                freq[symbol] = frequency
            
            # Read encoded data
            bit_length = struct.unpack('>I', f.read(4))[0]
            padding = struct.unpack('B', f.read(1))[0]
            
            # Calculate number of bytes needed
            byte_length = (bit_length + 7) // 8
            encoded_bytes = f.read(byte_length)
            
            # Convert to bits
            ba = bitarray.bitarray()
            ba.frombytes(encoded_bytes)
            
            # Remove padding bits
            encoded_bits = ba[:bit_length]
            
            # Rebuild Huffman codes
            huffman_tree = build_huffman_tree(freq)
            huffman_codes = {sym: code for sym, code in huffman_tree}
            
            # Decode data
            flattened_data = huffman_decode(encoded_bits, huffman_codes)
            
            # Convert back to RLE format
            blocks_count = ((gray_height + 7) // 8) * ((gray_width + 7) // 8)
            gray_rle = []
            
            i = 0
            for _ in range(blocks_count):
                block_rle = []
                dc_run = flattened_data[i]
                i += 1
                dc_value = flattened_data[i]
                i += 1
                
                # Add DC coefficient
                block_rle.append((dc_run, dc_value))
                
                # Process AC coefficients
                total_coeffs = 1  # Start with 1 for DC
                while i < len(flattened_data) - 1 and total_coeffs < 64:
                    run = flattened_data[i]
                    i += 1
                    value = flattened_data[i]
                    i += 1
                    
                    block_rle.append((run, value))
                    total_coeffs += run + 1
                    
                    # Break on EOB marker
                    if run == 0 and value == 0:
                        break
                
                # Ensure we have EOB marker
                if (len(block_rle) == 0 or block_rle[-1] != (0, 0)) and total_coeffs < 64:
                    block_rle.append((0, 0))
                    
                gray_rle.append(block_rle)
            
            # Apply inverse differential coding to DC coefficients
            dc_diff = [block[0][1] for block in gray_rle]
            dc_coeffs = dc_differential_decode(dc_diff)
            
            # Replace DC values
            for i, block in enumerate(gray_rle):
                block_mod = list(block)
                block_mod[0] = (block_mod[0][0], dc_coeffs[i])
                gray_rle[i] = block_mod
            
            # Create the full encoded data dictionary
            encoded_data = {
                'gray_rle': gray_rle,
                'gray_shape': gray_shape,
                'y_qtable': y_qtable,
                'quality': quality,
                'original_shape': original_shape,
                'is_grayscale': True
            }
            
        else:
            # Read color channel shapes
            y_height, y_width = struct.unpack('>HH', f.read(4))
            cb_height, cb_width = struct.unpack('>HH', f.read(4))
            cr_height, cr_width = struct.unpack('>HH', f.read(4))
            
            Y_shape = (y_height, y_width)
            Cb_shape = (cb_height, cb_width)
            Cr_shape = (cr_height, cr_width)
            
            # Read quantization tables
            y_qtable = np.frombuffer(f.read(64), dtype=np.uint8).reshape(8, 8)
            c_qtable = np.frombuffer(f.read(64), dtype=np.uint8).reshape(8, 8)
            
            # Read Y channel data
            y_symbols = struct.unpack('>I', f.read(4))[0]
            y_freq = {}
            for _ in range(y_symbols):
                symbol, frequency = struct.unpack('>ii', f.read(8))
                y_freq[symbol] = frequency
                
            y_bit_length, y_padding = struct.unpack('>IB', f.read(5))
            y_byte_length = (y_bit_length + 7) // 8
            y_bytes = f.read(y_byte_length)
            
            y_bits = bitarray.bitarray()
            y_bits.frombytes(y_bytes)
            y_bits = y_bits[:y_bit_length]
            
            # Read Cb channel data
            cb_symbols = struct.unpack('>I', f.read(4))[0]
            cb_freq = {}
            for _ in range(cb_symbols):
                symbol, frequency = struct.unpack('>ii', f.read(8))
                cb_freq[symbol] = frequency
                
            cb_bit_length, cb_padding = struct.unpack('>IB', f.read(5))
            cb_byte_length = (cb_bit_length + 7) // 8
            cb_bytes = f.read(cb_byte_length)
            
            cb_bits = bitarray.bitarray()
            cb_bits.frombytes(cb_bytes)
            cb_bits = cb_bits[:cb_bit_length]
            
            # Read Cr channel data
            cr_symbols = struct.unpack('>I', f.read(4))[0]
            cr_freq = {}
            for _ in range(cr_symbols):
                symbol, frequency = struct.unpack('>ii', f.read(8))
                cr_freq[symbol] = frequency
                
            cr_bit_length, cr_padding = struct.unpack('>IB', f.read(5))
            cr_byte_length = (cr_bit_length + 7) // 8
            cr_bytes = f.read(cr_byte_length)
            
            cr_bits = bitarray.bitarray()
            cr_bits.frombytes(cr_bytes)
            cr_bits = cr_bits[:cr_bit_length]
            
            # Decode all channels
            # Rebuild Huffman codes
            y_tree = build_huffman_tree(y_freq)
            y_codes = {sym: code for sym, code in y_tree}
            
            cb_tree = build_huffman_tree(cb_freq)
            cb_codes = {sym: code for sym, code in cb_tree}
            
            cr_tree = build_huffman_tree(cr_freq)
            cr_codes = {sym: code for sym, code in cr_tree}
            
            # Decode data
            y_data = huffman_decode(y_bits, y_codes)
            cb_data = huffman_decode(cb_bits, cb_codes)
            cr_data = huffman_decode(cr_bits, cr_codes)
            
            # Convert back to RLE format
            Y_blocks_count = ((y_height + 7) // 8) * ((y_width + 7) // 8)
            Cb_blocks_count = ((cb_height + 7) // 8) * ((cb_width + 7) // 8)
            Cr_blocks_count = ((cr_height + 7) // 8) * ((cr_width + 7) // 8)
            
            # Process Y channel RLE
            Y_rle = []
            i = 0
            for _ in range(Y_blocks_count):
                block_rle = []
                
                # DC coefficient
                dc_run = y_data[i]
                i += 1
                dc_value = y_data[i]
                i += 1
                block_rle.append((dc_run, dc_value))
                
                # AC coefficients
                total_coeffs = 1  # Start with 1 for DC
                while i < len(y_data) - 1 and total_coeffs < 64:
                    run = y_data[i]
                    i += 1
                    value = y_data[i]
                    i += 1
                    
                    block_rle.append((run, value))
                    total_coeffs += run + 1
                    
                    # Break on EOB marker
                    if run == 0 and value == 0:
                        break
                
                # Ensure EOB marker
                if (len(block_rle) == 0 or block_rle[-1] != (0, 0)) and total_coeffs < 64:
                    block_rle.append((0, 0))
                    
                Y_rle.append(block_rle)
            
            # Process Cb channel RLE
            Cb_rle = []
            i = 0
            for _ in range(Cb_blocks_count):
                block_rle = []
                
                # DC coefficient
                dc_run = cb_data[i]
                i += 1
                dc_value = cb_data[i]
                i += 1
                block_rle.append((dc_run, dc_value))
                
                # AC coefficients
                total_coeffs = 1  # Start with 1 for DC
                while i < len(cb_data) - 1 and total_coeffs < 64:
                    run = cb_data[i]
                    i += 1
                    value = cb_data[i]
                    i += 1
                    
                    block_rle.append((run, value))
                    total_coeffs += run + 1
                    
                    # Break on EOB marker
                    if run == 0 and value == 0:
                        break
                
                # Ensure EOB marker
                if (len(block_rle) == 0 or block_rle[-1] != (0, 0)) and total_coeffs < 64:
                    block_rle.append((0, 0))
                    
                Cb_rle.append(block_rle)
            
            # Process Cr channel RLE
            Cr_rle = []
            i = 0
            for _ in range(Cr_blocks_count):
                block_rle = []
                
                # DC coefficient
                dc_run = cr_data[i]
                i += 1
                dc_value = cr_data[i]
                i += 1
                block_rle.append((dc_run, dc_value))
                
                # AC coefficients
                total_coeffs = 1  # Start with 1 for DC
                while i < len(cr_data) - 1 and total_coeffs < 64:
                    run = cr_data[i]
                    i += 1
                    value = cr_data[i]
                    i += 1
                    
                    block_rle.append((run, value))
                    total_coeffs += run + 1
                    
                    # Break on EOB marker
                    if run == 0 and value == 0:
                        break
                
                # Ensure EOB marker
                if (len(block_rle) == 0 or block_rle[-1] != (0, 0)) and total_coeffs < 64:
                    block_rle.append((0, 0))
                    
                Cr_rle.append(block_rle)
            
            # Apply inverse differential coding
            y_dc_diff = [block[0][1] for block in Y_rle]
            y_dc = dc_differential_decode(y_dc_diff)
            
            cb_dc_diff = [block[0][1] for block in Cb_rle]
            cb_dc = dc_differential_decode(cb_dc_diff)
            
            cr_dc_diff = [block[0][1] for block in Cr_rle]
            cr_dc = dc_differential_decode(cr_dc_diff)
            
            # Replace DC values
            for i, block in enumerate(Y_rle):
                block_mod = list(block)
                block_mod[0] = (block_mod[0][0], y_dc[i])
                Y_rle[i] = block_mod
            
           
            for i, block in enumerate(Cb_rle):
                block_mod = list(block)
                block_mod[0] = (block_mod[0][0], cb_dc[i])
                Cb_rle[i] = block_mod
                
            for i, block in enumerate(Cr_rle):
                block_mod = list(block)
                block_mod[0] = (block_mod[0][0], cr_dc[i])
                Cr_rle[i] = block_mod
            
            # Create the full encoded data dictionary
            encoded_data = {
                'Y_rle': Y_rle,
                'Cb_rle': Cb_rle,
                'Cr_rle': Cr_rle,
                'Y_shape': Y_shape,
                'Cb_shape': Cb_shape,
                'Cr_shape': Cr_shape,
                'y_qtable': y_qtable,
                'c_qtable': c_qtable,
                'quality': quality,
                'original_shape': original_shape
            }
        
        return encoded_data



# Main application function
def main():
    root = tk.Tk()
    app = JpegCompressionApp(root)
    
    # Update canvases after window is drawn
    root.update()
    
    # Canvas resize handler
    def on_resize(event):
        if event.widget == app.original_canvas and app.input_image is not None:
            app.display_image_on_canvas(app.input_image, app.original_canvas)
        elif event.widget == app.decoded_canvas and app.decoded_image is not None:
            app.display_image_on_canvas(app.decoded_image, app.decoded_canvas)
        elif event.widget == app.decode_canvas and app.decoded_image is not None:
            app.display_image_on_canvas(app.decoded_image, app.decode_canvas)
    
    app.original_canvas.bind("<Configure>", on_resize)
    app.decoded_canvas.bind("<Configure>", on_resize)
    app.decode_canvas.bind("<Configure>", on_resize)
    
    root.mainloop()

if __name__ == "__main__":
    main()
