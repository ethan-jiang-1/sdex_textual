import os
from PIL import Image

def prepare_images(save_path):
    def _load_local_images():
        name = save_path.split("_")[-1]
        dir_src = f"/content/drive/MyDrive/shared/sd_concept_data/{name}"
        if not os.path.isdir(dir_src):
            dir_src = f"/root/texture/sd_concept_data/{name}"
            if not os.path.isdir(dir_src):
                raise ValueError(f"no folder named {dir_src}")

        images = []
        fnames = os.listdir(dir_src)
        for fname in fnames:
            if fname.endswith(".jpg") or fname.endswith(".jpeg"):
                fullpath = f"{dir_src}/{fname}"
                if os.path.isfile(fullpath):
                    image = Image.open(fullpath).convert("RGB")
                    image = image.resize((512,512))
                    images.append(image)
        return images

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    images = _load_local_images()
    [image.save(f"{save_path}/{i}.jpeg") for i, image in enumerate(images)]
    #return image_grid(images, 1, len(images))
    return len(images)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid