from PIL import Image  # uses pillow
from shutil import copyfile
import array as arr

pathInput="flickr_logos_27_dataset_images/"
pathOutput="tmp/"

# Convert box to imagenet format based on image size
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
    

fh = open('flickr_logos_27_dataset_training_set_annotation.txt')
categories = set()
lista = {}

for line in fh:
    fn, c, id, x1,y1,x2,y2 = line.split()
    categories.add(c)
    datalist = [c,id,x1,y1,x2,y2]
    key=id+"_"+c+"_"+fn
    if key in lista:
        templist = lista[key]
        templist.append(datalist)
        lista[key]=templist
    else:
        lista[key]=[]
        lista[key].append(datalist)
fh.close()


cat = list(categories)
print (cat)


for label in lista:
    id,c,fn = label.split("_")
    cid = cat.index(c)
    im = Image.open(pathInput+fn)
    size=im.size
    fn2 = id+"_"+c+"_"+fn 
    w= int(im.size[0])
    h= int(im.size[1])
    a =  ""
    size2 = w,h
    for item in lista[label]:
        box = float(item[2]), float(item[4]), float(item[3]), float(item[5])
        x,y,w,h = convert(size2,box)
        a = a+ str(cid)+ " "  + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n"
    fn3 =  fn2[:-3] + 'txt'
    print (pathInput+fn+ "--->"+pathOutput+fn2+ "("+pathOutput+fn3+")" )
    copyfile(pathInput+fn, pathOutput+fn2)
    text_file = open(pathOutput+fn3, "w")
    text_file.write(a)
    text_file.close()

print ("=========== DONE")
print ("CATEGORIES LIST for LABELS")
print (cat)

