import torch
from PIL import Image
from scripts.data_loader import get_transform, CLASS_NAMES

def predict_image(model, image_path, device):
    model.eval()
    transform = get_transform()
    
    try:
        image = Image.open(image_path).convert('L')
        image = image.resize((224, 224))
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            preds = torch.sigmoid(outputs)
            
        preds_np = preds.cpu().numpy().flatten()
        
        results = {}
        for i, class_name in enumerate(CLASS_NAMES):
            results[class_name] = float(preds_np[i])
            
        return results
    except Exception as e:
        return {"error": str(e)}
