from PIL import Image
import os
import glob

def compress_images(path:str, output_path:str, size):
    """ Compress all images in path to given size, output to path in output_path
        with same directory structure.
    """
    if path[-1] != '/':
        path = path + '/'
    if output_path[-1] != '/':
        output_path = output_path + '/'


    image_paths = []
    for (root, dirs, fl) in os.walk(path):
        for f in fl:
            if '.jpg' in f:
                full_path = root + '/' + f
                image_paths.append(full_path[len(path):])
    
    for ip in image_paths:
        img_path = path + ip
        out_path = output_path + ip
        out_dir = "/".join(out_path.split('/')[:-1])
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        img = Image.open(img_path)
        img = img.convert('RGB')
        # img.show()
        img = img.resize(size, Image.Resampling.LANCZOS)
        # img.show()
        img.save(out_path)



if __name__ == '__main__':
    path = input("Enter image path: ")
    op = input("Enter output path: ")
    i, j, = map(int, input("Enter new image size: ").split())
    compress_images(path, op, (i, j))
