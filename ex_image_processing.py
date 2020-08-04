from PIL import Image, ImageOps

dreamdir = '.\\img\\'
converted_dir = '.\\converted\\'

def convert_nbody(img):
    img = img.crop((8, 0, img.width, img.height))

    nbody_pantie = Image.new('RGBA', (img.width*2, img.height))
    nbody_pantie.paste(img, (img.width, 0))
    nbody_pantie.paste(ImageOps.mirror(img), (0, 0))
    return nbody_pantie

for num in range(10):
    pantie_name = f'{dreamdir}{num+1:04d}.png'
    img = Image.open(pantie_name)
    pantie = convert_nbody(img)
    pantie.save(f'{converted_dir}{num+1:04d}.png')

# ribbon.resize((ribbon.width*m, ribbon.height*m), resample=Image.BICUBIC)