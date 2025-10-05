from classes.BrainTumorPredictor import BrainTumorPredictor
from pathlib import Path

def predict_single_image(args):
    """Predict a single image"""
    predictor = BrainTumorPredictor(
        model_path=args.model_path,
        model_type=args.model_type,
        class_names=args.class_names
    )
    
    results = predictor.visualize_prediction(
        args.image_path,
        save_path=args.output,
        show_gradcam=args.gradcam
    )
    
    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Image: {args.image_path}")
    print(f"Predicted Class: {results['predicted_class'].upper()}")
    print(f"Confidence: {results['confidence']:.2f}%")
    print("\nAll Class Probabilities:")
    for class_name, prob in results['probabilities'].items():
        print(f"  {class_name:15s}: {prob:6.2f}%")
    print("="*60)

def predict_directory(args):
    """Predict all images in a directory"""
    predictor = BrainTumorPredictor(
        model_path=args.model_path,
        model_type=args.model_type,
        class_names=args.class_names
    )
    
    # Get all images
    image_dir = Path(args.image_path)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir.glob(ext))
    
    print(f"\nFound {len(image_paths)} images in {image_dir}")
    
    # Predict
    results = predictor.predict_batch(image_paths)
    
    # Display results
    print("\n" + "="*80)
    print("BATCH PREDICTION RESULTS")
    print("="*80)
    for result in results:
        print(f"{Path(result['image_path']).name:40s} -> {result['predicted_class']:15s} ({result['confidence']:.1f}%)")
    print("="*80)
    
    # Save summary
    if args.output:
        import pandas as pd
        df = pd.DataFrame([{
            'image': Path(r['image_path']).name,
            'prediction': r['predicted_class'],
            'confidence': r['confidence'],
            **r['probabilities']
        } for r in results])
        df.to_csv(args.output, index=False)
        print(f"\n✓ Results saved to: {args.output}")

