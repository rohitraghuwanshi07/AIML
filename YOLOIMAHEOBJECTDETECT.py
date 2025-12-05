import cv2
import numpy as np
import os
import urllib.request
from pathlib import Path
import matplotlib.pyplot as plt

class YOLOv3ObjectDetector:
    """
    A complete YOLOv3 object detection implementation using OpenCV
    """
    
    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Initialize the YOLOv3 detector
        
        Args:
            confidence_threshold (float): Minimum confidence for detections
            nms_threshold (float): Threshold for Non-Maximum Suppression
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.classes = []
        self.colors = []
        
    def download_yolo_files(self):
        """
        Download YOLOv3 configuration, weights, and class names files
        """
        print("Downloading YOLOv3 files...")
        
        # URLs for YOLOv3 files
        files_to_download = {
            'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
            'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
            'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
        }
        
        for filename, url in files_to_download.items():
            if not os.path.exists(filename):
                print(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filename)
                    print(f"✓ Downloaded {filename}")
                except Exception as e:
                    print(f"✗ Failed to download {filename}: {e}")
                    print(f"Please manually download from: {url}")
            else:
                print(f"✓ {filename} already exists")
    
    def load_model(self):
        """
        Load the YOLOv3 model and class names
        """
        print("Loading YOLOv3 model...")
        
        # Load class names
        try:
            with open('coco.names', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            print(f"✓ Loaded {len(self.classes)} class names")
        except FileNotFoundError:
            print("✗ coco.names file not found!")
            return False
        
        # Generate random colors for each class
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Load YOLO network
        try:
            self.net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
            print("✓ YOLOv3 model loaded successfully")
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            print(f"✓ Output layers: {self.output_layers}")
            
            return True
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess the input image
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            tuple: (original_image, blob, height, width)
        """
        print(f"Loading image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        height, width, channels = image.shape
        print(f"Image dimensions: {width}x{height}x{channels}")
        
        # Create blob from image
        # YOLOv3 expects 416x416 input size, normalize to [0,1], and swap RGB channels
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
        print("✓ Image preprocessed and converted to blob")
        
        return image, blob, height, width
    
    def detect_objects(self, blob):
        """
        Run forward pass through the network to detect objects
        
        Args:
            blob: Preprocessed image blob
            
        Returns:
            list: Network outputs from each output layer
        """
        print("Running object detection...")
        
        # Set input to the network
        self.net.setInput(blob)
        
        # Run forward pass
        outputs = self.net.forward(self.output_layers)
        print("✓ Forward pass completed")
        
        return outputs
    
    def process_detections(self, outputs, width, height):
        """
        Process network outputs to extract bounding boxes, confidences, and class IDs
        
        Args:
            outputs: Raw network outputs
            width (int): Original image width
            height (int): Original image height
            
        Returns:
            tuple: (boxes, confidences, class_ids)
        """
        print("Processing detections...")
        
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each output layer
        for output in outputs:
            # Each detection is a vector of 85 values:
            # [center_x, center_y, width, height, objectness, class_1_prob, ..., class_80_prob]
            for detection in output:
                # Extract class probabilities (skip first 5 values)
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter out weak detections
                if confidence > self.confidence_threshold:
                    # YOLO returns center coordinates, convert to corner coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        print(f"✓ Found {len(boxes)} detections above confidence threshold")
        return boxes, confidences, class_ids
    
    def apply_nms(self, boxes, confidences, class_ids):
        """
        Apply Non-Maximum Suppression to remove overlapping boxes
        
        Args:
            boxes: List of bounding boxes
            confidences: List of confidence scores
            class_ids: List of class IDs
            
        Returns:
            tuple: (filtered_boxes, filtered_confidences, filtered_class_ids)
        """
        print("Applying Non-Maximum Suppression...")
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                filtered_boxes.append(boxes[i])
                filtered_confidences.append(confidences[i])
                filtered_class_ids.append(class_ids[i])
        
        print(f"✓ After NMS: {len(filtered_boxes)} final detections")
        return filtered_boxes, filtered_confidences, filtered_class_ids
    
    def draw_detections(self, image, boxes, confidences, class_ids):
        """
        Draw bounding boxes and labels on the image
        
        Args:
            image: Original image
            boxes: Filtered bounding boxes
            confidences: Filtered confidence scores
            class_ids: Filtered class IDs
            
        Returns:
            numpy.ndarray: Image with drawn detections
        """
        print("Drawing detections on image...")
        
        # Create a copy to avoid modifying the original
        result_image = image.copy()
        
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]
            
            # Get class name and color
            label = self.classes[class_id]
            color = self.colors[class_id]
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Create label text
            label_text = f"{label}: {confidence:.2f}"
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                result_image, 
                (x, y - text_height - baseline), 
                (x + text_width, y), 
                color, 
                -1
            )
            
            # Draw text
            cv2.putText(
                result_image, 
                label_text, 
                (x, y - baseline), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2
            )
        
        print("✓ Detections drawn on image")
        return result_image
    
    def detect_and_visualize(self, image_path, output_path=None):
        """
        Complete detection pipeline: load image, detect objects, and visualize results
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save output image (optional)
            
        Returns:
            numpy.ndarray: Image with detections
        """
        # Preprocess image
        image, blob, height, width = self.preprocess_image(image_path)
        
        # Detect objects
        outputs = self.detect_objects(blob)
        
        # Process detections
        boxes, confidences, class_ids = self.process_detections(outputs, width, height)
        
        # Apply NMS
        final_boxes, final_confidences, final_class_ids = self.apply_nms(boxes, confidences, class_ids)
        
        # Draw detections
        result_image = self.draw_detections(image, final_boxes, final_confidences, final_class_ids)
        
        # Save output image if path provided
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"✓ Output saved to: {output_path}")
        
        # Print detection summary
        print("\n" + "="*50)
        print("DETECTION SUMMARY:")
        print("="*50)
        
        detected_objects = {}
        for i, class_id in enumerate(final_class_ids):
            class_name = self.classes[class_id]
            confidence = final_confidences[i]
            
            if class_name not in detected_objects:
                detected_objects[class_name] = []
            detected_objects[class_name].append(confidence)
        
        for obj, confs in detected_objects.items():
            print(f"{obj}: {len(confs)} detections (avg confidence: {np.mean(confs):.3f})")
        
        print("="*50)
        
        return result_image

def main():
    """
    Main function to run the YOLOv3 object detection
    """
    print("YOLOv3 Object Detection System")
    print("="*50)
    
    # Initialize detector
    detector = YOLOv3ObjectDetector(confidence_threshold=0.5, nms_threshold=0.4)
    
    # Download required files
    detector.download_yolo_files()
    
    # Load model
    if not detector.load_model():
        print("Failed to load model. Exiting.")
        return
    
    # Example usage - you can modify this section
    image_path = "sample_image.jpg"  # Change this to your image path
    
    # Create a sample image if it doesn't exist (for demonstration)
    if not os.path.exists(image_path):
        print(f"Sample image not found at {image_path}")
        print("Please provide a valid image path or place an image named 'sample_image.jpg' in the current directory")
        
        # You can download a sample image for testing
        sample_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg"
        try:
            urllib.request.urlretrieve(sample_url, "dog.jpg")
            image_path = "dog.jpg"
            print("Downloaded sample image: dog.jpg")
        except:
            print("Could not download sample image. Please provide your own image.")
            return
    
    try:
        # Run detection
        result_image = detector.detect_and_visualize(image_path, "output_detections.jpg")
        
        # Display result using matplotlib (works better in different environments)
        plt.figure(figsize=(12, 8))
        # Convert BGR to RGB for matplotlib
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        plt.imshow(result_rgb)
        plt.title("YOLOv3 Object Detection Results")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("detection_visualization.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\nDetection completed successfully!")
        print("Output files created:")
        print("- output_detections.jpg (detected objects with bounding boxes)")
        print("- detection_visualization.png (matplotlib visualization)")
        
    except Exception as e:
        print(f"Error during detection: {e}")

if __name__ == "__main__":
    main()