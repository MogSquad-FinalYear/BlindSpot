import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import kornia.augmentation as K
from transformers import CLIPVisionModel
from peft import get_peft_model, LoraConfig, TaskType

# CLIP input image preprocessing
def discriminator_preprocess(image_tensor):
    """Preprocess images for CLIP model input."""
    # If the input is already a batched tensor, only apply normalization.
    if isinstance(image_tensor, torch.Tensor):
        # Normalize to the CLIP image space.
        normalize = K.AugmentationSequential(
            K.Normalize(
                mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
                std=torch.tensor([0.26862954, 0.26130258, 0.27577711]),
            ),
        )
        return normalize(image_tensor)
    else:
        raise TypeError("Expected the input to be a torch.Tensor.")

# CLIP-LoRA Detector model with lower rank
class CLIPLoRADetector(nn.Module):
    """CLIP-based generated-image detector with a LoRA adapter."""
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", lora_rank=4):
        super(CLIPLoRADetector, self).__init__()
        
        # Load the CLIP vision backbone.
        self.clip = CLIPVisionModel.from_pretrained(clip_model_name)
        
        # Configure LoRA for low-rank adaptation.
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        )
        
        # Wrap the CLIP vision model with LoRA layers.
        self.clip = get_peft_model(self.clip, peft_config)
        
        # Freeze the base model parameters.
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # Keep LoRA parameters trainable.
        for name, param in self.clip.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        
        # Add a lightweight linear classifier on top of the CLS token.
        self.classifier = nn.Linear(self.clip.config.hidden_size, 1)
    
    def forward(self, pixel_values):
        """Forward pass."""
        # Fetch CLIP outputs robustly in case the PEFT wrapper API differs.
        try:
            # First try the wrapped model directly.
            outputs = self.clip(pixel_values)
        except Exception:
            try:
                # Fall back to the underlying base model.
                outputs = self.clip.base_model(pixel_values=pixel_values)
            except Exception:
                # As a last resort, resolve the underlying model explicitly.
                if hasattr(self.clip, 'model'):
                    clip_model = self.clip.model
                elif hasattr(self.clip, 'base_model'):
                    clip_model = self.clip.base_model
                else:
                    clip_model = self.clip
                    
                outputs = clip_model.forward(pixel_values=pixel_values)
        
        # Classify from the CLS token at position 0.
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        return logits

# Helper: check whether a file is a valid image.
def is_image_file(filename):
    """Check whether a file is a supported image."""
    try:
        with Image.open(filename) as img:
            img.verify()
        return True
    except:
        return False

class CLIPLoRADetectorWrapper:
    """Convenience wrapper for loading and running the CLIP-LoRA detector."""
    
    def __init__(self, model_path, clip_model_name="openai/clip-vit-large-patch14", 
                 lora_rank=4, device=None):
        """
        Initialize the CLIP-LoRA detector.

        Args:
            model_path: Path to the detector checkpoint.
            clip_model_name: CLIP model identifier.
            lora_rank: LoRA adapter rank.
            device: Target runtime device such as ``cuda:0`` or ``cpu``.
        """
        # Select the runtime device.
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load the detector model.
        self.model = self._load_model(model_path, clip_model_name, lora_rank)
        
        # Image preprocessing for PIL inputs.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def _load_model(self, model_path, clip_model_name, lora_rank):
        """Load a pretrained CLIP-LoRA detector."""
        model = CLIPLoRADetector(
            clip_model_name=clip_model_name, 
            lora_rank=lora_rank
        ).to(self.device)
        
        # Load pretrained weights.
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Switch to evaluation mode.
        model.eval()
        return model
    
    def preprocess_image(self, image):
        """Preprocess a single image."""
        if isinstance(image, str):
            # Load from an image path.
            image = Image.open(image).convert('RGB')
        
        if isinstance(image, Image.Image):
            # Convert a PIL image to tensor form.
            image = self.transform(image)
        
        return image
    
    def predict(self, image, threshold=0.5):
        """
        Predict whether an image is real or AI-generated.

        Args:
            image: PIL image, image path, or a preprocessed tensor.
            threshold: Decision threshold, default is 0.5.

        Returns:
            A dictionary with the prediction result.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Preprocess the input image.
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:  # Single image tensor: add a batch dimension.
                    img_tensor = image.unsqueeze(0).to(self.device)
                else:  # Already batched.
                    img_tensor = image.to(self.device)
            else:
                # Convert non-tensor inputs to a batched tensor.
                img_tensor = self.preprocess_image(image)
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Apply detector-specific preprocessing.
            img_tensor = discriminator_preprocess(img_tensor)
            
            # Run the model.
            output = self.model(img_tensor)
            
            # Convert logits to a binary prediction.
            prob = torch.sigmoid(output).cpu().item()
            pred = int(prob >= threshold)
        
        return {
            'prediction': pred,  # 0 = real, 1 = AI-generated
            'probability': prob,
            'is_generated': bool(pred)
        }

if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Load and test the CLIP-LoRA detector.")
    parser.add_argument("--model_path", type=str, 
                        default="/pubdata/zhouyue/Adversarial_Training_AIGC/checkpoints_low_rank/best_model_low_rank.pt",
                        help="Path to the model checkpoint.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on.")
    parser.add_argument("--test_image", type=str, help="Optional test image path.")
    
    args = parser.parse_args()
    
    # Create the detector instance.
    print("Initializing the CLIP-LoRA detector...")
    detector = CLIPLoRADetectorWrapper(
        model_path=args.model_path,
        device=args.device
    )
    
    # Test the model with a random tensor.
    print("\nTesting the model with a random tensor...")
    random_tensor = torch.rand(1, 3, 224, 224)  # Random image tensor with batch size 1.
    print(f"Random tensor shape: {random_tensor.shape}")
    
    # Preprocess the tensor and inspect the output.
    processed_tensor = discriminator_preprocess(random_tensor)
    print(f"Processed tensor shape: {processed_tensor.shape}")
    print(f"First values before preprocessing: {random_tensor[0, 0, 0, :5]}")
    print(f"First values after preprocessing: {processed_tensor[0, 0, 0, :5]}")
    
    # Run a forward pass.
    result = detector.predict(random_tensor)
    print("\nModel output:")
    print(f"Prediction: {'AI-generated' if result['is_generated'] else 'Real'}")
    print(f"Probability: {result['probability']:.4f}")
    
    # Test a real image if one is provided.
    if args.test_image:
        print(f"\nTesting image: {args.test_image}")
        try:
            result = detector.predict(args.test_image)
            print(f"Prediction: {'AI-generated' if result['is_generated'] else 'Real'}")
            print(f"Probability: {result['probability']:.4f}")
        except Exception as e:
            print(f"Error while testing the image: {str(e)}")
    
    print("\nDone.")
