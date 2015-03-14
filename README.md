# t-SNE Guardian
This little project shows a t-SNE visualization of articles from [The Guardian](guardian.co.uk) published in 2014.

## What I've done here
- Used the Guardian API to grab the title and thumbnail of all articles published in 2014.
- Used the [spacy](https://honnibal.github.io/spaCy/) NLP code to extract nouns from titles and trailing text.  Each article is now a "bag-of-nouns".
- Calculated cosine distance between articles in the bag-of-noun space.
- Used the `scikit-learn` t-SNE implementation to embed the articles in a 2d space, based on those cosine distances.
- Made a big jpg image showing the thumbnails for the articles in this 2d space.
- Hacked some leaflet/javascript for browser visualization.

## Results
- [U.S. news](http://stanford.edu/~rkeisler/tsne_guardian/us/)
- [World news](http://stanford.edu/~rkeisler/tsne_guardian/world/)
- [Football/Soccer new](http://stanford.edu/~rkeisler/tsne_guardian/football/)
 
