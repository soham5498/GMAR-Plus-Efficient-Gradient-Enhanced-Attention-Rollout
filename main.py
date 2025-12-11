"""
Main entry point for GMAR++ Explainability Analysis.

This module provides the main() function which orchestrates the complete
explainability workflow including method selection, model initialization,
dataset processing, and result aggregation.
"""

import traceback
from pathlib import Path

from src.managers import DatasetManager
from src.engine import ExplainabilityEngine
from src.explainers import ExplainerFactory

try:
    import kagglehub
except Exception:
    kagglehub = None


def main():
    """
    Main entry point orchestrating the complete explainability analysis workflow.
    
    This function implements the high-level user-facing workflow:
    1. Prompts user to select explainability method from available options
    2. Initializes ExplainabilityEngine with pre-trained Vision Transformer model
    3. Processes local image dataset with selected method
    4. Optionally downloads and processes TinyImageNet test set for validation
    5. Prints final recap comparing results across datasets
    
    The workflow is designed for interactive use with clear console feedback at
    each stage. The function handles errors gracefully - if TinyImageNet download
    fails, processing continues with local dataset results only.
    """
    try:
        # Step 1: Get available methods and prompt user
        try:
            available_methods = ExplainerFactory.available_methods()
            if not available_methods:
                raise ValueError("No explanation methods available in ExplainerFactory")
            methods_str = "/".join(available_methods)
        except Exception as e:
            print(f"[ERROR] Failed to retrieve available methods: {e}")
            return
        
        # Step 2: Get user input with validation
        try:
            choice = input(f"Which method? ({methods_str}): ").strip().lower()
            if not choice:
                raise ValueError("Method selection cannot be empty")
            if choice not in available_methods:
                raise ValueError(
                    f"Invalid choice '{choice}'. Available methods: {', '.join(available_methods)}"
                )
        except (EOFError, KeyboardInterrupt) as e:
            print(f"\n[INFO] User cancelled input: {type(e).__name__}")
            return
        except ValueError as e:
            print(f"[ERROR] Invalid input: {e}")
            return

        # Step 3: Initialize ExplainabilityEngine
        try:
            model_path = "checkpoints/vit_large_tinyimagenet/best/"
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Model checkpoint not found at '{model_path}'. "
                    f"Please ensure the checkpoint directory exists."
                )
            engine = ExplainabilityEngine(
                model_name=model_path,
                results_root=Path("./results")
            )
        except FileNotFoundError as e:
            print(f"[ERROR] Model initialization failed - {e}")
            return
        except (RuntimeError, ValueError) as e:
            print(f"[ERROR] Failed to initialize model: {e}")
            return
        except Exception as e:
            print(f"[ERROR] Unexpected error during engine initialization: {type(e).__name__}: {e}")
            return

        # Step 4: Process local dataset
        image_dir = Path("./images")
        mask_dir = Path("./masks")
        local_summary = None
        
        try:
            if not image_dir.exists():
                print(f"[WARN] Image directory '{image_dir}' not found. Skipping local dataset.")
                local_summary = {}
            else:
                try:
                    local_images = DatasetManager.list_images(image_dir)
                    if not local_images:
                        print(f"[WARN] No images found in '{image_dir}'. Skipping local dataset.")
                        local_summary = {}
                    else:
                        local_summary = engine.process_dataset(
                            dataset_tag="local",
                            image_files=local_images,
                            method=choice,
                            mask_dir=mask_dir if mask_dir.exists() else None,
                            overlay_limit=None
                        )
                except (OSError, IOError) as e:
                    print(f"[ERROR] Failed to read images from '{image_dir}': {e}")
                    local_summary = {}
                except Exception as e:
                    print(f"[ERROR] Unexpected error processing local dataset: {type(e).__name__}: {e}")
                    local_summary = {}
        except KeyboardInterrupt:
            print(f"\n[INFO] Local dataset processing interrupted by user")
            local_summary = {}
        except Exception as e:
            print(f"[ERROR] Unexpected error in local dataset processing: {type(e).__name__}: {e}")
            local_summary = {}

        # Step 5: Process TinyImageNet test set
        tiny_summary = {}
        if kagglehub is None:
            print("[WARN] kagglehub not available; skipping tiny test dataset.")
        else:
            try:
                try:
                    base = Path(kagglehub.dataset_download("akash2sharma/tiny-imagenet"))
                except (ConnectionError, TimeoutError) as e:
                    print(f"[WARN] Network error downloading TinyImageNet: {e}")
                    base = None
                except Exception as e:
                    print(f"[WARN] Failed to download TinyImageNet dataset: {e}")
                    base = None
                
                if base is not None:
                    try:
                        tiny_root = DatasetManager.find_tinyimagenet_root(base)
                        if tiny_root is None:
                            raise FileNotFoundError("TinyImageNet structure not found in downloaded data")
                        
                        test_images_dir = tiny_root / "test" / "images"
                        if not test_images_dir.exists():
                            raise FileNotFoundError(f"TinyImageNet test images directory not found: {test_images_dir}")
                        
                        try:
                            tiny_test_images = DatasetManager.list_images(test_images_dir)
                            if tiny_test_images:
                                tiny_summary = engine.process_dataset(
                                    dataset_tag="tiny_test",
                                    image_files=tiny_test_images,
                                    method=choice,
                                    mask_dir=None,
                                    overlay_limit=20
                                )
                            else:
                                print(f"[WARN] No images found in TinyImageNet test directory")
                                tiny_summary = {}
                        except (OSError, IOError) as e:
                            print(f"[WARN] Failed to read TinyImageNet images: {e}")
                            tiny_summary = {}
                        except Exception as e:
                            print(f"[WARN] Unexpected error processing TinyImageNet: {type(e).__name__}: {e}")
                            tiny_summary = {}
                    except FileNotFoundError as e:
                        print(f"[WARN] TinyImageNet structure error: {e}")
                        tiny_summary = {}
                    except Exception as e:
                        print(f"[WARN] Unexpected error locating TinyImageNet: {type(e).__name__}: {e}")
                        tiny_summary = {}
            except KeyboardInterrupt:
                print(f"\n[INFO] TinyImageNet processing interrupted by user")
                tiny_summary = {}
            except Exception as e:
                print(f"[WARN] Failed to process Tiny-ImageNet test: {type(e).__name__}: {e}")
                tiny_summary = {}

        # Step 6: Print final recap
        try:
            print("\n================ Final Recap ================ ")
            if local_summary:
                print(f"Local images summary: {local_summary}")
            else:
                print("Local images summary: (no data)")
            
            if tiny_summary:
                print(f"Tiny-ImageNet test summary: {tiny_summary}")
            else:
                print("Tiny-ImageNet test summary: (no data)")
            print("============================================ ")
        except Exception as e:
            print(f"[ERROR] Failed to print final recap: {e}")
    
    except KeyboardInterrupt:
        print(f"\n[INFO] Application interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"[CRITICAL ERROR] Unexpected error in main: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
