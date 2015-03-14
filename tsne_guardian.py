import ipdb
import numpy as np
import matplotlib.pylab as pl
import cPickle as pickle
datadir = 'data/'
pl.ion()
global_seed = 1

def main(section='us-news'):
    # Downloads the data from the Guardian API.
    # you'll need to get your own API key and put 
    # it a file called my_api_key.
    x = load_raw_set(section=section, quick=False)

    # Vectorizes sentences, embeds into 2d based on 
    # bag-of-noun distance, creates relevant jpg 
    # and html pages.
    pipeline(section=section, quick_pos=False)


def pipeline(section='us-news', quick_pos=False):
    np.random.seed(global_seed)
    pos_savename = '%s/pos_%s%i.pkl'%(datadir, section, global_seed)
    x = load_raw_set(section=section, quick=True)
    xx = preprocess_set(x, include_trail_text=True)
    if not(quick_pos): 
        vecs, words = vectorize_sentences(xx.text.values)
        dist = get_distance_matrix(vecs, metric='cosine')
        pos = tsne_embed(dist)
        pickle.dump(pos, open(pos_savename,'w'))
    else: pos = pickle.load(open(pos_savename,'r'))
    kw = big_image_kwargs(section)
    #word_clusters = get_word_clusters(pos, vecs, words)
    word_clusters = None
    make_big_image(pos, xx.image.values, section, word_clusters=word_clusters,
                   headlines=xx.headline.values, **kw)


def big_image_kwargs(section):
    if section=='us-news':
        kw={'density':1.00, 'nx_small':60, 'ny_small':100}
    if section=='world':
        kw={'density':0.54, 'nx_small':42, 'ny_small':70}
    if section=='football':
        kw={'density':0.54, 'nx_small':42, 'ny_small':70}
    return kw


def load_data_for_words(section='us-news'):
    pos_savename = '%s/pos_%s%i.pkl'%(datadir, section, global_seed)
    x = load_raw_set(section=section, quick=True)
    xx = preprocess_set(x, include_trail_text=True)
    vecs, words = vectorize_sentences(xx.text.values)
    pos = pickle.load(open(pos_savename,'r'))
    return pos, vecs, words

def get_word_clusters(pos, vecs, words, 
                      thresh=0.07, n_use_max=4, 
                      n_clusters=20, doplot=False):
    np.random.seed(global_seed)
    from sklearn.cluster import DBSCAN, AffinityPropagation, KMeans
    db = KMeans(n_clusters=n_clusters)
    db.fit(pos)
    if doplot:
        pl.clf()
        pl.plot(pos[:, 0], pos[:,1], '.')
    ncluster = len(set(db.labels_))
    output = []
    sf_fontsize = 0.15*(pos.max()-pos.min())
    for icluster, cluster in enumerate(set(db.labels_)):
        print '%i/%i'%(icluster, ncluster)
        wh_cluster = np.where(db.labels_==cluster)[0]
        x_cluster = np.mean(pos[wh_cluster, 0])
        y_cluster = np.mean(pos[wh_cluster, 1])
        score = np.squeeze(np.asarray(vecs[wh_cluster,:].mean(0)))
        score *= len(wh_cluster)
        ind_sort = np.argsort(score)[::-1]
        words_sort = words[ind_sort]
        score_sort = score[ind_sort]
        wh_big = np.where(score_sort > thresh)[0]
        ind_use = ind_sort[wh_big]
        if len(ind_use)>n_use_max: ind_use = ind_use[0:n_use_max]
        words_use = words[ind_use]
        score_use = score[ind_use]
        x_use = np.random.randn(len(ind_use))*1.4 + x_cluster
        y_use = np.random.randn(len(ind_use))*1. + y_cluster
        fontsize_use = sf_fontsize*np.sqrt(score_use)
        for this_word, this_fontsize, this_x, this_y in zip(words_use, fontsize_use, x_use, y_use):
            if doplot: pl.text(this_x, this_y, this_word, size=this_fontsize, ha='center')
            output.append({'x':-this_x/16., 'y':this_y/16., 'text':this_word, 'size':this_fontsize})
    return output



def make_big_image(pos, images, section, 
                   headlines=None,
                   density=1.0, force_grid=True,
                   npix_horizontal=7500, npix_vertical=4000,
                   nx_small=60, ny_small=100,
                   word_clusters=None,
                   randomize_order=False):

    from scipy.misc import imresize
    
    # load and rescale the 2d embedding.
    x = pos[:,0].copy()
    y = pos[:,1].copy()
    x -= x.min(); x /= x.max(); x *= 0.9; x += 0.05
    y -= y.min(); y /= y.max(); y *= 0.9; y += 0.05
    rr = np.sqrt((x-np.median(x))**2. + (y-np.median(y))**2.)
    wh_outlier = np.where(rr>np.percentile(rr,99))[0]
    x[wh_outlier]=np.median(x)
    y[wh_outlier]=np.median(y)
    x -= x.min(); x /= x.max(); x *= 0.9; x += 0.05
    y -= y.min(); y /= y.max(); y *= 0.9; y += 0.05
    
    # stupid matplotlib axis convention...
    nx_big = npix_vertical
    ny_big = npix_horizontal

    # initialize the big image.
    big = np.zeros((nx_big, ny_big, 3))
    background_color = np.array([255, 255, 255])
    big += background_color

   
    n_obj = len(x)
    ind = np.arange(n_obj)
    if randomize_order:
        np.random.shuffle(ind)
    n_used = 0
    rect_lines = []
    # loop over small images and add them to the big image.
    #for counter, this_x, this_y, this_image in zip(range(n_obj), x, y, images):
    for counter in range(n_obj):
        if (counter % 100)==0: print '*** %i/%i ***'%(counter, n_obj)
        j = ind[counter]
        this_x = x[j]
        this_y = y[j]
        this_image = images[j]
        this_headline = headlines[j]

        # get the location of this image.
        a = np.ceil(this_x * (nx_big-nx_small)+1)
        b = np.ceil(this_y * (ny_big-ny_small)+1)
        if force_grid:
            a = a-np.mod(a-1,nx_small)+1
            b = b-np.mod(b-1,ny_small)+1
            # if there is already an image here, skip it.
            # (or random walk until you find one)
            while (big[a,b,1] != background_color[1]):
                a_proposal = a + nx_small*np.random.random_integers(-1,1)
                b_proposal = b + ny_small*np.random.random_integers(-1,1)
                while((a_proposal<nx_small) | 
                      (b_proposal<ny_small) | 
                      (a_proposal+nx_small>(nx_big-nx_small)) | 
                      (b_proposal+ny_small>(ny_big-ny_small))):
                    a_proposal = a + nx_small*np.random.random_integers(-1,1)
                    b_proposal = b + ny_small*np.random.random_integers(-1,1)
                a = a_proposal
                b = b_proposal

            #if (big[a,b,1] != background_color[1]): continue

        # randomly decide whether or not we keep this image.
        if np.random.random() > density: continue

        # make sure the new small image will fit in the big image.
        #if ((a<0) | (b<0) | (a+nx_small>nx_big) | (b+ny_small>ny_big)):
        if ((a<nx_small) | (b<ny_small) | (a+nx_small>(nx_big-nx_small)) | (b+ny_small>(ny_big-ny_small))):
            continue

        # load the new small image.
        this_small = imresize(this_image, (nx_small, ny_small, 3))
        if this_small.shape != (nx_small, ny_small, 3):
            continue

        # put the new image into the big image.
        big[a:a+nx_small, b:b+ny_small, :] = this_small
        n_used += 1

        # record the position and headline of this image.
        if headlines is not None:
            corner0 = [a,b]
            corner1 = [a+nx_small-1, b+ny_small-1]
            rect_lines.append(get_one_rect_line(corner0, corner1, this_headline))

    # make sure the output directory exists.
    import os
    outdir = '%s%i/'%(section2name(section), global_seed)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # make the html page.
    make_index_html(section, npix_horizontal, npix_vertical, rect_lines, word_clusters=word_clusters)

    # convert to 8-bit and save as jpg.
    print 'used %i images'%n_used
    savename = '%s%i/big.jpg'%(section2name(section), global_seed)
    pl.imsave(savename, big.astype('uint8'))


def view_random_headlines(pos, headlines, probability=0.01, xlim=None, ylim=None):
    x = pos[:,1]
    y = -pos[:,0]
    pl.clf()
    if xlim: pl.xlim(xlim)
    if ylim: pl.ylim(ylim)
    pl.plot(x, y, '.')
    for xx, yy, headline in zip(x, y, headlines):
        if (np.random.random()<probability):
            pl.text(xx, yy, headline)
        


def tsne_embed(distance_matrix, early_exaggeration=4.):
    print '...running TSNE...'
    from sklearn.manifold import TSNE
    model = TSNE(early_exaggeration=early_exaggeration)
    pos = model.fit_transform(distance_matrix)
    return pos


def get_distance_matrix(vecs, metric='cosine'):
    print '...computing distance matrix...'
    # VECS should be [n_docs, n_dim_word_space].
    from sklearn.metrics.pairwise import pairwise_distances
    return pairwise_distances(vecs, metric=metric)
    

def vectorize_sentences(corpus):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(min_df=1)
    vecs = vectorizer.fit_transform(corpus)
    vecs = vecs.astype(float)
    # force vectors to have unit length.
    norm = np.sqrt(vecs.multiply(vecs).sum(1))
    vecs = vecs.multiply(1./norm)
    # get the words corresponding to the dimensions of the vector.
    words = np.array(vectorizer.get_feature_names())
    return vecs, words


def preprocess_set(data, include_trail_text=False, only_nouns=True,
                   min_wordcount=200, max_wordcount=20000):
    print '...preprocessing...'
    newdata = []

    if only_nouns:
        from spacy.en import English
        nlp = English()

    for x in data:
        if 'wordcount' not in x:
            continue
        if ((int(x['wordcount'])<min_wordcount) | 
            (int(x['wordcount'])>max_wordcount)):
            continue

        # add the headline to text.
        this_head = remove_html(x['headline'])

        # add the trail text to text.
        if include_trail_text:
            this_trail = remove_html(x['trailText'])
            this_head = this_head + '.  ' + this_trail

        # if desired, get only nouns.
        # make sure it returns one big string.
        if only_nouns: this_head = get_only_nouns(this_head, nlp=nlp)

        # append to output.
        if len(this_head)==0: continue
        this_dict = {'text':this_head, 'image':x['image'], 'headline':x['headline']}
        newdata.append(this_dict)

    # convert to a DataFrame object.
    from pandas import DataFrame
    return DataFrame(newdata)


def get_only_nouns(line, nlp=None):
    if not(nlp):
        from spacy.en import English
        nlp = English()
    tokens = nlp(line)
    output = ' '.join([t.string for t in tokens if t.pos_=='NOUN'])
    return output


def remove_html(line):
    if '<' not in line: return line
    if '>' not in line: return line
    newline = ''
    removing = False
    for c in line:
        if removing:
            if c=='>':
                removing=False
                newline += ' '
                continue
        else:
            if c=='<':
                removing=True
                continue
            else:
                newline += c
    return newline

        
def download_many_raw():
    for section in ['world','football']:
        load_raw_set(section=section,
                     start_date='2014-01-01',
                     stop_date='2014-12-31', 
                     query=None, quick=False)


def load_raw_set(section='us-news', 
                 start_date='2014-01-01',
                 stop_date='2014-12-31', 
                 query=None, quick=False):

    print '...loading raw set...'
    from urllib import urlretrieve
    from scipy.ndimage import imread
    from subprocess import call

    # get the savename
    savename = '%s/%s_%s_%s'%(datadir, section, start_date, stop_date)
    if query: savename += '_'+query
    savename += '.pkl'
    if quick: return pickle.load(open(savename,'r'))

    api_key = load_api_key()
    data = call_api(api_key=api_key, section=section, 
                   start_date=start_date, stop_date=stop_date, 
                    page=1, query=query)
    ntotal = data['response']['total']
    print ' '
    print '%i TOTAL ARTICLES'%ntotal
    print ' '
    npages = data['response']['pages']
    output = []
    for page in np.arange(npages)+1:
        print '--- PAGE %i/%i ---'%(page, npages)
        data = call_api(api_key=api_key, section=section, 
                        start_date=start_date, stop_date=stop_date, 
                        page=page, query=query)
        for result in data['response']['results']:
            if 'thumbnail' not in result['fields']: continue
            if not(is_ascii(result['fields']['thumbnail'])): continue
            these_fields = result['fields']
            print these_fields['headline'].encode('utf-8')
            print these_fields['trailText'].encode('utf-8')
            this_thumb_url = result['fields']['thumbnail']
            urlretrieve(this_thumb_url,'tmp.jpg')
            these_fields['image'] = imread('tmp.jpg')
            output.append(these_fields)
    call(['rm','tmp.jpg'])

    # save
    pickle.dump(output, open(savename,'w'))
    return output


def call_api(api_key='test', section=None,
             start_date=None, stop_date=None, 
             page=None, query=None):
    import json
    import urllib2
    url = 'http://content.guardianapis.com/search?api-key='+api_key
    url += '&show-fields=thumbnail,headline,trailText,wordcount,commentable'
    url += '&page-size=100'
    if query: url += '&q=%s'%query
    if start_date: url += '&from-date=%s'%start_date
    if stop_date: url += '&to-date=%s'%stop_date
    if section: url += '&section=%s'%section
    if page: url += '&page=%i'%page
    print url
    return json.load(urllib2.urlopen(url))


def load_api_key():
    f=open('my_api_key','r')
    return f.read().rstrip().strip()

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def make_index_html(section, npix_horizontal, npix_vertical, rect_lines, 
                    word_clusters=None):
    fin = open('index_template.html','r')
    lines_out = []
    name = section2name(section)
    for line in fin:

        if 't-SNE' in line:
            lines_out.append('t-SNE visualization of %s news from The Guardian\n'%name.upper())
            continue
                    
        if 'var w =' in line:
            lines_out.append('var w = %i;\n'%npix_horizontal)
            continue

        if 'var h =' in line:
            lines_out.append('var h = %i;\n'%npix_vertical)
            continue

        if 'var url =' in line:
            this_url = "'http://stanford.edu/~rkeisler/tsne_guardian/%s/big.jpg'"%name
            lines_out.append('var url = %s;\n'%this_url)
            continue

        if 'add rectangles here' in line:
            lines_out = lines_out + rect_lines
            continue

        # otherwise just keep the line as is.
        lines_out.append(line)
    fin.close()

    # now write out.
    fout = open('%s%i/index.html'%(name, global_seed),'w')
    for line in lines_out:
        fout.write(line.encode('utf8'))
    fout.close()


def get_one_rect_line(corner0, corner1, headline):
    sf = 1./16.
    this_line = 'L.rectangle([ [%.3f,%.3f], [%0.3f, %0.3f] ], {opacity:myOpacity, fillOpacity:myOpacity}).bindPopup("%s",{autoPan:false}).on("mouseover", function (e) {if (map.getZoom() > 2) {this.openPopup();}}).on("mouseout", function (e) {this.closePopup();}).addTo(map);\n'%(-corner0[0]*sf, corner0[1]*sf, -corner1[0]*sf, corner1[1]*sf, headline.replace('"',"'"))
    return this_line


def section2name(section):
    name = {'us-news':'us', 
            'world':'world', 
            'football':'football'}[section]
    return name



if __name__ == "__main__":
    main()

