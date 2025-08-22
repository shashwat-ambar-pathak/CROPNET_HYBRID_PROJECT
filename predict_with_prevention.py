import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from preventions import prevention_tips
from class_labels import class_names

model = load_model("cropnet_hybrid_model_one.h5")

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = class_names[predicted_class_idx]
    
    confidence = float(np.max(prediction))

    preventions = prevention_tips.get(predicted_class, ["No data available."])
    return predicted_class, confidence, preventions

# Example usage:
if __name__ == "__main__":
    test_img = "sample.jpg"   # change this to your test image
    disease, conf, tips = predict_disease(test_img)
    
    img = image.load_img(test_img)
    plt.imshow(img)
    plt.axis("off")
    tips_text = "\n".join([f"- {t}" for t in tips])
    plt.title(f"Input Image\nPrediction: {disease} ({conf:.2f}%)\n{tips_text}", fontsize=10)
    plt.show()

    print(f"Predicted Disease: {disease}")
    print(f"Confidence: {conf:.2f}")
    print("Prevention Tips:")
    for t in tips:
        print(f"- {t}")
