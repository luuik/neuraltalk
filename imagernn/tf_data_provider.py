import json
import os
import random
import scipy.io
import codecs
from collections import defaultdict
import argparse
import time
import numpy as np
import tensorflow as tf

class BasicDataProvider:
  def __init__(self, dataset):
    print 'Initializing data provider for dataset %s...' % (dataset, )

    # !assumptions on folder structure
    self.dataset_root = os.path.join('data', dataset)
    self.image_root = os.path.join('data', dataset, 'imgs')

    # load the dataset into memory
    dataset_path = os.path.join(self.dataset_root, 'dataset.json')
    print 'BasicDataProvider: reading %s' % (dataset_path, )
    self.dataset = json.load(open(dataset_path, 'r'))

    # load the image features into memory
    features_path = os.path.join(self.dataset_root, 'vgg_feats.mat')
    print 'BasicDataProvider: reading %s' % (features_path, )
    features_struct = scipy.io.loadmat(features_path)
    self.features = features_struct['feats']

    # group images by their train/val/test split into a dictionary -> list structure
    self.split = defaultdict(list)
    for img in self.dataset['images']:
      self.split[img['split']].append(img)

  # "PRIVATE" FUNCTIONS
  # in future we may want to create copies here so that we don't touch the 
  # data provider class data, but for now lets do the simple thing and 
  # just return raw internal img sent structs. This also has the advantage
  # that the driver could store various useful caching stuff in these structs
  # and they will be returned in the future with the cache present
  def _getImage(self, img):
    """ create an image structure for the driver """

    # lazily fill in some attributes
    if not 'local_file_path' in img: img['local_file_path'] = os.path.join(self.image_root, img['filename'])
    if not 'feat' in img: # also fill in the features
      feature_index = img['imgid'] # NOTE: imgid is an integer, and it indexes into features
      img['feat'] = self.features[:,feature_index]
    return img

  def _getSentence(self, sent):
    """ create a sentence structure for the driver """
    # NOOP for now
    return sent

  # PUBLIC FUNCTIONS

  def getSplitSize(self, split, ofwhat = 'sentences'):
    """ return size of a split, either number of sentences or number of images """
    if ofwhat == 'sentences': 
      return sum(len(img['sentences']) for img in self.split[split])
    else: # assume images
      return len(self.split[split])

  def sampleImageSentencePair(self, split = 'train'):
    """ sample image sentence pair from a split """
    images = self.split[split]

    img = random.choice(images)
    sent = random.choice(img['sentences'])

    out = {}
    out['image'] = self._getImage(img)
    out['sentence'] = self._getSentence(sent)
    return out

  def iterImageSentencePair(self, split = 'train', max_images = -1):
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        yield out

  def iterImageSentencePairBatch(self, split = 'train', max_images = -1, max_batch_size = 100):
    batch = []
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        batch.append(out)
        if len(batch) >= max_batch_size:
          yield batch
          batch = []
    if batch:
      yield batch

  def iterSentences(self, split = 'train'):
    for img in self.split[split]: 
      for sent in img['sentences']:
        yield self._getSentence(sent)

  def iterSentencesMultiSplits(self, splits):
    for mysplit in splits:
      for img in self.split[mysplit]: 
        for sent in img['sentences']:
          yield self._getSentence(sent)

  def iterImages(self, split = 'train', shuffle = False, max_images = -1):
    imglist = self.split[split]
    ix = range(len(imglist))
    if shuffle:
      random.shuffle(ix)
    if max_images > 0:
      ix = ix[:min(len(ix),max_images)] # crop the list
    for i in ix:
      yield self._getImage(imglist[i])

def getDataProvider(dataset):
  """ we could intercept a special dataset and return different data providers """
  assert dataset in ['flickr8k', 'flickr30k', 'coco'], 'dataset %s unknown' % (dataset, )
  return BasicDataProvider(dataset)

def preProBuildWordVocab(sentence_iterator, word_count_threshold):
  # count up all word counts so that we can threshold
  # this shouldnt be too expensive of an operation
  print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
  t0 = time.time()
  word_counts = {}
  nsents = 0
  for sent in sentence_iterator:
    nsents += 1
    for w in sent['tokens']:
      word_counts[w] = word_counts.get(w, 0) + 1
  vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
  print 'filtered words from %d to %d in %.2fs' % (len(word_counts), len(vocab), time.time() - t0)

  # with K distinct words:
  # - there are K+1 possible inputs (START token and all the words)
  # - there are K+1 possible outputs (END token and all the words)
  # we use ixtoword to take predicted indeces and map them to words for output visualization
  # we use wordtoix to take raw words and get their index in word vector matrix
  ixtoword = {}
  ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
  wordtoix = {}
  wordtoix['#START#'] = 0 # make first vector be the start token
  ix = 1
  for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

  # compute bias vector, which is related to the log probability of the distribution
  # of the labels (words) and how often they occur. We will use this vector to initialize
  # the decoder weights, so that the loss function doesnt show a huge increase in performance
  # very quickly (which is just the network learning this anyway, for the most part). This makes
  # the visualizations of the cost function nicer because it doesn't look like a hockey stick.
  # for example on Flickr8K, doing this brings down initial perplexity from ~2500 to ~170.
  word_counts['.'] = nsents
  bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
  bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
  bias_init_vector = np.log(bias_init_vector)
  bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
  return wordtoix, ixtoword, bias_init_vector

def make_example(img_feats, sentence):
  # The object we return
  img_feats = [a.item() for a in img_feats]
  #sentence = [a.item() for a in sentence]
  example = tf.train.Example(
    # Example contains a Features proto object
    features=tf.train.Features(
      # Features contains a map of string to Feature proto objects
      feature={
        # A Feature contains one of either a int64_list,
        # float_list, or bytes_list
        'img_feats': tf.train.Feature(
          float_list=tf.train.FloatList(value=img_feats)),
        'sentence': tf.train.Feature(
          int64_list=tf.train.Int64List(value=sentence)),
      }))
    # use the proto object to serialize the example to a string
  return example

def write_record(splitname, dp, misc):
  with codecs.open(splitname+".tfrec",'w',encoding='utf8') as f:
    writer = tf.python_io.TFRecordWriter(f.name)
    for img in dp.split[splitname]:
      feature_index = img['imgid'] # NOTE: imgid is an integer, and it indexes into features
      img_feats = dp.features[:,feature_index]
      for sent in img['sentences']:
        img_sent = dp._getSentence(sent)
        token_ids = []
        for w in sent['tokens']:
          token_ids.append(misc['wordtoix'][w])
        ex = make_example(img_feats, token_ids)
        serialized = ex.SerializeToString()
        writer.write(serialized)
    writer.close()
    print("Wrote to {}".format(f.name))
        
def main(params):
  # load data and dumps
  # 1) vocabulary index
  # 2) tensorflow record
  dp = getDataProvider(params["dataset"])

  # 1) write vocabulary file
  misc = {}
  misc['wordtoix'], misc['ixtoword'], bias_init_vector = preProBuildWordVocab(dp.iterSentencesMultiSplits(['train', 'val', 'test']), params["word_count_threshold"])
  with codecs.open("neuraltalk.vcb",'w',encoding='utf8') as f:
    for index in misc['ixtoword']:
      f.write(str(index)+"\t"+misc['ixtoword'][index]+"\n")

  splits = ['train', 'val', 'test'] 
  # 2) write tensorflow records
  for mysplit in splits:
    write_record(mysplit, dp, misc)
    
    
if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument('-d', '--dataset', dest='dataset', default='flickr8k', help='dataset: flickr8k/flickr30k')

  # data preprocessing parameters
  parser.add_argument('--word_count_threshold', dest='word_count_threshold', type=int, default=1, help='if a word occurs less than this number of times in training data, it is discarded')


  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)