from classes.BrainTumorPredictor import BrainTumorPredictor

# Configure your model
MODEL_PATH = "./models/best_ResNet50.pth"  # Your trained model
MODEL_TYPE = "ResNet50"           # Must match your training
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']


def start():
    # Load the predictor
    predictor = BrainTumorPredictor(
        model_path=MODEL_PATH,
        model_type=MODEL_TYPE,
        class_names=CLASS_NAMES
    )

    # Predict a single image
    results = predictor.visualize_prediction(
        image_path="./data/Testing/glioma_tumor/image.jpg",
        save_path="./results/prediction_output.png",
        show_gradcam=True
    )

    # Print results
    print(f"\nPrediction: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']:.2f}%")
