from duckduckgo_search import DDGS
from fastcore.all import * 
import time, json
from fastdownload import *
from fastai.vision.all import * 

def search_images(keywords, max_images=200): 
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')

def custom_image_classifier(classified_item): 
    urls = search_images(f'{classified_item} photos', max_images=1)
    dest = f'{classified_item}.jpg'
    download_url(urls[0], dest, show_progress=False)
    path = Path(classified_item + '_or_not')
    dest = (path/classified_item)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest,urls=search_images(f'{classified_item} photo'))
    time.sleep(5)
    resize_images(path/classified_item,max_size=400,dest=path/classified_item)

    # Training 
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    
    dls = DataBlock(
            blocks={ImageBlock, CategoryBlock}, 
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=parent_label,
            item_tfms=[Resize(192, method='crop')]).dataloaders(path, bs=32)
    
    # On resnet18
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)

    # Exporting the model 
    learn.export(f"{classified_item}_classifier.pk1")

    # Testing
    is_classified_item,_,probs = learn.predict(PILImage.create(f'{classified_item}.jpg'))

    print(f"This is a: {is_classified_item}.")
    print(f"Probability it's a {classified_item}: {probs[dls.vocab.o2i[classified_item]]:.4f}")
if __name__ == "__main__": 
    classified_item = input("Enter whatever you'd like to classify, don't end or start with a space ")
    custom_image_classifier(classified_item)
