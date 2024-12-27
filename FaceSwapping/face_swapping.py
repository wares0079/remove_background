import cv2
from insightface.app import FaceAnalysis
import insightface
import numpy as np
import gfpgan

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
model_output_path = f'model/inswapper_128.onnx'
swapper = insightface.model_zoo.get_model(model_output_path, download=False, download_zip=False)


def gfpgan_runner(img, model):
    _, imgs, _ = model.enhance(img, paste_back=True, has_aligned=False)
    return imgs[0]


def enhance(name='GFPGAN', device="cpu"):
    supported_enhancers = {"GFPGAN": ("model/GFPGANv1.4.pth", gfpgan_runner)}
    modelpath, model_runner = supported_enhancers.get(name)
    model = gfpgan.GFPGANer(model_path=modelpath, upscale=2, device=device)
    return (model, model_runner)


def list_faces():
    photo = cv2.imread(f'output/source.jpg')
    detected_faces = app.get(photo)

    for i, face in enumerate(detected_faces):
        x1, y1, x2, y2 = map(int, face.bbox)
        
        margin_x = int((x2 - x1) * 0.1)
        margin_y = int((y2 - y1) * 0.1)

        new_x1 = max(0, x1 - margin_x)
        new_y1 = max(0, y1 - margin_y)
        new_x2 = min(photo.shape[1], x2 + margin_x)
        new_y2 = min(photo.shape[0], y2 + margin_y)
        
        face_crop = photo[new_y1:new_y2, new_x1:new_x2]
        cv2.imwrite(f'output/{i}.jpg', face_crop)
    
    swap_face(bbox_=str([new_x1, new_y1, new_x2, new_y2]), kps_=str(face.kps.tolist()))




def swap_face(bbox_, kps_):
    bbox = np.array(eval(bbox_))
    kps = np.array(eval(kps_))
    original_image = cv2.imread('output/source.jpg')
    new_face_image = cv2.imread(f'output/target.jpg')
    new_face_detected = app.get(new_face_image)

    if len(new_face_detected) != 1:
        return 'Could not detect exactly one face in the new image.'

    new_face = new_face_detected[0]
    class Face:
        def __init__(self, bbox, kps):
            self.bbox = bbox
            self.kps = kps

    original_face = Face(bbox, kps)
    res = swapper.get(original_image, original_face, new_face, paste_back=True)
    cv2.imwrite('output/Swapped.jpg', res)

    model, model_runner = enhance('GFPGAN','cpu')
    img1_ = model_runner(res, model)
    cv2.imwrite('output/enhanced.jpg', img1_)

    return "Done"




list_faces()