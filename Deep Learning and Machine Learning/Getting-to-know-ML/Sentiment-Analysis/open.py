with open("/Users/15i330/Documents/Datasets/sentiment_labelled_sentences/amazon_cells_labelled.txt") as f1:
    lines = f1.readlines()

with open("/Users/15i330/Documents/Datasets/sentiment_labelled_sentences/imdb_labelled.txt") as f1:
    temp = f1.readlines()
    lines=lines+temp

with open("/Users/15i330/Documents/Datasets/sentiment_labelled_sentences/yelp_labelled.txt") as f1:
    temp = f1.readlines()
    lines=lines+temp
    
    
#change the data location with your location
