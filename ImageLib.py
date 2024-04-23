## Standard library imports
import logging
import ssl
import time
import requests
import urllib
import string
import re

## Related third-party imports
import cv2
import pandas as pd ## only used in mode colors
import icecream as ic
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
# from scipy.spatial import distance
import skimage
# from skimage import color
# from skimage import io
import easyocr as eo
from spellchecker import SpellChecker

## Local application/library-specific imports
# from imagesc import imagesc ## imports Undouble which messes with logging
# from undouble import Undouble



def get_full_class_name(obj):
    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__
    return module + '.' + obj.__class__.__name__

def retry(method, kwargs, retries=5, max_backoff=2):
  for i in range(retries):
    try:
      logging.debug(f'Attempt {i} of {method.__name__}() with \
        \n\tKWARGS: {kwargs}')
      return method(**kwargs)
    except Exception as e:
      logging.warn(f'FAIL {i} of {method.__name__}() with \
        \n\tKWARGS: {kwargs}\
        \n\tError: {e}')
      time.sleep( max(0.3*retries, max_backoff) ) ## progressively longer backoff
      continue
    else:
      break
    print("wtf")
  logging.error(f'FAILED all ({retries}) attempts of {method.__name__}() with \
    \n\tKWARGS: {kwargs}')

def image_info(image, NUM_BINS=99):
    # calculate histogram
    try:
        hist = cv2.calcHist([image],[0],None,[NUM_BINS],[0,256])
    except:
        logging.error(f"failed to calculate hist for {str(image)}")
        hist = None
    # calculate image shape
    try:
            h, w, c = image.shape
    except:
        logging.error(f"failed to calculate shape for {str(image)}")
        h,w,c = None,None,None
    return hist,h,w,c

## Note sk is slower, but plays nice with colab
## CV2 works with collab with patched show (default print has wack channels)
## img = image[:, :, ::-1] # convert image from RGB (skimage) to BGR (opencv)
def url_to_image_cv2(url, readFlag=cv2.IMREAD_COLOR):
  # download the image, convert it to a NumPy array, and then read it into OpenCV format
  context = ssl._create_unverified_context()
  resp = urllib.request.urlopen(url, context=context)
  image = np.asarray(bytearray(resp.read()), dtype="uint8")
  # image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
  image = cv2.imdecode(image, readFlag)
  return image

def url_to_image_ski(url):
  image = skimage.io.imread( url )
  return image


## https://gist.github.com/jonathantneal/d3a259ebeb46de7ab0de
def nearest_normal_aspect_ratio(height, width, side=0, MAXW=16, MAXH=16):
    """
    NOT rigorously tested :)
    Finds the nearest normal aspect ratio based on the given width and height.

    Args:
        width: The width of the space.
        height: The height of the space.
        side (optional): The nearest ratio to side with.
            A positive value returns the nearest ratio equal to or higher than the actual ratio.
            A negative value returns the nearest ratio equal to or smaller than the actual ratio.
            Defaults to 0.
        MAXW (optional): The maximum width in the nearest normal aspect ratio.
            Defaults to 16.
        MAXH (optional): The maximum height in the nearest normal aspect ratio.
            Defaults to 16.

    Returns:
        str: The nearest aspect ratio in the format "width:height".
    """
    ratio = (width * 100) / (height * 100)
    ratios_w = [i + 1 for i in range(MAXW)]
    ratios_h = [i + 1 for i in range(MAXH)]
    ratios_t = {}
    ratios = {}
    match = None

    for ratio_w in ratios_w:
        for ratio_h in ratios_h:
            ratio_x = (ratio_w * 100) / (ratio_h * 100)

            if ratio_x not in ratios_t:
                ratios_t[ratio_x] = True
                ratios[f"{ratio_w}:{ratio_h}"] = ratio_x

    for key in ratios:
        if not match or (
            (not side) and abs(ratio - ratios[key]) < abs(ratio - ratios[match])
        ) or (
            (side < 0) and ratios[key] <= ratio and abs(ratio - ratios[key]) < abs(ratio - ratios[match])
        ) or (
            (side > 0) and ratios[key] >= ratio and abs(ratio - ratios[key]) < abs(ratio - ratios[match])
        ):
            match = key

    return match, {'w':match.split(':')[0], 'h':match.split(':')[1]}


def plot_mask(img, mask):
  black_pixels_mask = np.all(img == mask, axis=-1)
  non_black_pixels_mask = np.any(img != mask, axis=-1)
  # or non_black_pixels_mask = ~black_pixels_mask

  cpy = img.copy()
  cpy[black_pixels_mask] = [255, 255, 255]
  cpy[non_black_pixels_mask] = [0, 0, 0]

  plt.imshow(cpy)
  plt.show()

def to_hex(arr, format='RGB'):
  hex = None
  if format=='RGB':
    hex = f'#{arr[0]:02x}{arr[1]:02x}{arr[2]:02x}'
  elif format=='BGR':
    hex = f'#{arr[2]:02x}{arr[1]:02x}{arr[0]:02x}'
  else:
    logging.warn(f"Can't convert format {format} to hex")
  return hex

def hex_to_bgr(hex):
  hexnum = hex.lstrip('#')
  return tuple(int(hexnum[i:i+2], 16) for i in (4, 2, 0))

def compress_colors(img, color_radius=1):
  '''
  "Round" the colors by truncating each RGB value
  '''
  ## Round colors ensuring nothing over 255
  roundedcolors = np.clip( (img/color_radius).round()*color_radius , 0, 255).astype('uint8')

  ## Show both images
  # vertical= np.concatenate((img_cv2,roundedcolors), axis = 1)
  # cv2_imshow(vertical)

  return roundedcolors

def mode_colors(img):
  '''
  Return the top colors in array of hex codes.
  '''
  ## Convert each RGB to hex and put in a flat list
  arr = []
  h = len(img)
  w = len(img[0])
  for ri in range(h):
    row = img[ri]
    for ci in range(w):
      arr.append(to_hex(img[ri][ci], format='BGR'))
      # print('#%02x%02x%02x' % (0, 128, 64))
      # print(img[ri][ci])
  arr = np.array(arr)

  unique, counts = np.unique(arr, return_counts=True)
  ## More optimal would be a top 3 search of the lists
  hex_cts_df = pd.DataFrame( {"hex":unique, "counts":counts.astype(int)}).sort_values('counts', ascending=False)
  hex_cts_df['pct'] = hex_cts_df['counts']/(h*w)
  # print(f'{hex_cts_df}')
  sorted_hexes = hex_cts_df['hex'].tolist()
  return sorted_hexes


def diff_hash(img, hashSize=8):
  ## Extra column for horizontal gradient
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  resized = cv2.resize( img_gray, (hashSize,hashSize+1) )
  # cv2_imshow(resized)
  diff = resized[:, 1:] > resized[:, :-1]
  # cv2_imshow(diff*255)

  bool_array = diff.flatten()
  # print(bool_array)
  bin_array = bool_array.astype(int)
  # print(bin_array)
  bin_string = ''.join(str(x) for x in bin_array)
  # bin_string = np.array2string(bin_array,separator='').strip('[]')
  # print(bin_string)

  ## this conversion from the tutorial reads the binary backwards
  # decimalhash = sum([2 ** i for (i, v) in enumerate(bool_array) if v])
  # print(f'{int(bin_string,2) = }')
  return bin_string

def perceptual_hash(img):
  '''
  Use openCV pHash function to return 8-byte hash as hex string
  '''
  imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  h = cv2.img_hash.pHash(imgg) # 8-byte hash
  hex = '0x'+h.tobytes().hex()
  return hex

def bin_to_hex(bs, leading_zero_width=0, hex_prefix='0x'):
  return f'{hex_prefix}{int(bs,2):0{int(leading_zero_width)}x}'

def hex_to_bin(bs, bin_prefix=''):
  number, pad, rjust, size, kind = int(bs,16), '0', '>', len(bs)*4, 'b'
  return f'{number:{pad}{rjust}{size}{kind}}'


# Draw bounding boxes
def draw_boxes(img, bounds, label=None, color=[0,0,255], width=3):
  topleft = bounds[0]
  bottomright = bounds[2]
  logging.debug(f'{topleft} - {bottomright}')
  if label:
    cv2.putText(img, label,(topleft[0],topleft[1]+20),0,0.3, color)

  return cv2.rectangle(img,topleft,bottomright,(0,255,0),width)

# a = [0, 1, 0, 1, 0, 1, 1, 1, 0]
# s = ''.join(str(x) for x in a)
# s

def draw_all_contours(img, contours):
  contours = contours[0] if len(contours) == 2 else contours[1] ## no idea
  for c in contours:
    cv2.drawContours(img, [c], -1, (36,255,12), 2)

def clean_contours(contours, axis, round_up_length=None, max_length=None, min_length=None):
  '''
  Takes output of find contour and places max/min per axis in dict.
  If contour is at least round_up pct of height/width then extend to full length.
  '''

  if axis == 'vertical':
    index_dim = 0 ## x axis
    span_dim = 1
  else:
    index_dim = 1 ## y axis
    span_dim = 0
  logging.debug(f'{index_dim = } and {span_dim = }')

  contours = contours[0] if len(contours) == 2 else contours[1] ## no idea
  nicer_contours = []
  indexed = {}
  for c in contours:
    # cv2.drawContours(result, [c], -1, (36,12,255), 2)
    # print(f'{c = }')
    p1, p2 = c[0][0], c[1][0]
    # p2= c[1][0]
    # print(p1)
    nicer_contours.append( [p1,p2] )

    ## "connect" countours by taking min and maxes
    larger = max(p2[span_dim],p1[span_dim])
    smaller = min(p2[span_dim],p1[span_dim])
    key = p1[index_dim]
    if key not in indexed:
      indexed[key] = [smaller, larger]
    else:
      cursmall = indexed[key][0]
      curbig = indexed[key][1]
      indexed[key] = [ min(cursmall, smaller), max(curbig, larger) ]

    ## round up to full if length is sufficient
    if round_up_length and indexed[key][1]-indexed[key][0] > round_up_length:
      indexed[key] = [0,max_length]

  ## remove tiny lines
  if min_length:
    for key in list(indexed.keys()):
      if indexed[key][1]-indexed[key][0] < min_length:
        indexed.pop(key)
  return indexed

def draw_dict_countours(img, contours, axis):
  for key in contours:
    if axis == 'vertical':
      ## indexes are x
      p1 = (key,contours[key][0])
      p2 = (key,contours[key][1])
    else:
      ## indexes are y
      p1 = (contours[key][0], key)
      p2 = (contours[key][1], key)
    cv2.line(img,p1,p2,(255,0,0),2)


def detect_bars(img, hk=(33,1), vk=(1,33)):
  ## Detect horizontal lines
  horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, hk)
  detect_horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
  contours_horizontal = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  ## Detect vertical lines
  vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, vk)
  detect_vertical = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
  contours_vertical = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  return contours_horizontal, contours_vertical


def characterize_bars(vertical, horizontal, height, width, tolerance_factor=15, tolerance_px=5):
  snapbars = []
  SC_HEIGHT = 40
  ## Assumes cleaning leaves only relevant bars (no floaters, segments connect a chord)
  ## alternatively could use this to count regions https://www.geeksforgeeks.org/count-number-of-object-using-python-opencv/

  ##--------------------------------------------------------------------------------  remove double counts

  ## vertical bar segments
  vbars = list(vertical.keys())
  
  if len(vbars) == 0:
    logging.debug('No vertical bars')
  else:
    ## add fake bars at bounds
    vbars.append(0)
    vbars.append(width)
    vbars.sort()

    lens = np.diff(np.array(vbars))
    vsegments = np.where ( lens > width/tolerance_factor )[0]


  ## horizontal bars need an extra step for snapchat
  hbars = list(horizontal.keys())
  snapbars = []
  if len(hbars) == 0:
    logging.debug('No horizontal bars')
  else:
    ## add fake bars at bounds
    hbars.append(0)
    hbars.append(height)
    hbars.sort()

    lens = np.diff(np.array(hbars))
    ## remove snapchat text element
    for i in range(len(lens)):
      if abs(SC_HEIGHT - lens[i]) <= tolerance_px:
        logging.debug('Excluding SC bars in collage calculation')
        snapbars=hbars[i:i+1+1]
        del hbars[i:i+1+1] ## remove both snap bars
        break ## only expecting one element

    ## recalc sizes without snapchat
    lens = np.diff(np.array(hbars))
    hsegments = np.where ( lens > height/tolerance_factor )[0]

  ##-------------------------------------------------------------------------------- calculate number of image slices
  ## count if segments or chords
  hchords = 0
  hsegs = 0
  ## hgsegments holds "good" indices of diff array, offset by 1 to get hbars
  ## array. Remove last elt which was added manually
  if len(hbars) == 0:
    pass
  else:
    indices = np.array(hbars)[hsegments+1][:-1]
    for h in indices:
      length = horizontal[h][1]-horizontal[h][0]
      if abs(width - length) <= tolerance_px:
        hchords+=1
      else: hsegs+=1

  vchords = 0
  vsegs = 0
  if len(vbars) == 0:
    pass
  else:
    indices = np.array(vbars)[vsegments+1][:-1]
    for v in indices:
      length = vertical[v][1]-vertical[v][0]
      if abs(height - length) <= tolerance_px:
        vchords+=1
      else: vsegs+=1


  ## Assumes all segments intersect 1 chord
  slices = (1+hchords)*(1+vchords)+vsegs+hsegs

  return snapbars, slices


def has_gray_overlay(in_row, out_row, tolerance=0.15):
  '''
  Checks if in_row is essentially out_row with a gray overlay
  '''
  out_avg = np.average(out_row,axis=0)
  out_ratio = out_avg / np.sum(out_avg)
  in_avg = np.average(in_row,axis=0)
  in_ratio = in_avg / np.sum(in_avg)
  # ic.ic(out_ratio, in_ratio)
  ratio_diffs = abs(in_ratio-out_ratio)
  darker = np.average(out_avg - in_avg) > 30
  ## All color channels proportionally darker (within tolerance)
  proportional = ratio_diffs[0]<tolerance and ratio_diffs[1]<tolerance and ratio_diffs[2]<tolerance
  # ic.ic(ratio_diffs, darker, proportional)
  return darker and proportional

def has_snap_overlay(img, y_top, y_bot, in_offset=2, out_offset=5):
  '''
  Checks for gray overlay between y_top and y_bot.
  Offsets are how many pixels away from the bar to grab colors.
  '''
  top_out = img[y_top+out_offset][:]
  top_in = img[y_top+in_offset][:]
  bottom_in = img[y_bot+in_offset][:]
  bottom_out = img[y_bot+out_offset][:]
  top_graylay = has_gray_overlay(top_in, top_out)
  bottom_graylay = has_gray_overlay(bottom_in, bottom_out)
  return top_graylay and bottom_graylay

PYTHONENV = None
def set_image_env(env):
  '''
  Necessary for showing images
  '''
  assert env in ['Colab','ipynb','py'], 'Invalid python environment for my image library'
  global PYTHONENV 
  PYTHONENV = env
  logging.info(f'Set PYTHONENV to {env}')
  
  if PYTHONENV == 'Colab':
    from google.colab.patches import cv2_imshow 
    logging.info(f'Imported Colab imshow patch.')

def nbimshow(img):
  logging.debug('Running nbimshow')
  try:
    if PYTHONENV=='Colab':
      logging.debug('Showing with colab patch')
      cv2_imshow(img)
    elif PYTHONENV=='ipynb':
      logging.debug('Showing with matplotlib')
      plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      plt.show()
    elif PYTHONENV=='py':
      logging.warning('Showing with matplotlib, but could have use cv2.imshow')
      plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      plt.show()
    else:
      logging.warning('set_image_env must be run to use nbimshow')
  except Exception as e:
    logging.error("An error occurred:", str(e))
    
    
def eOCR(img, reader, shrinkfactor=.5):
  ## Preprocess image (BW, binarize, shrink)
  h,w,_ = img.shape
  image_cv_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  th, image_cv_th = cv2.threshold(image_cv_bw, 0, 255, cv2.THRESH_OTSU )
  image_cv_th = cv2.resize(image_cv_th, (0,0), fx = shrinkfactor, fy = shrinkfactor)
  
  logging.debug(f'OCRing image with shape: {image_cv_th.shape}')
  paragraphs = reader.readtext(image_cv_th, paragraph=True)
  
  ## Return bounding box in original size
  # ic.ic(paragraphs)
  for paragraph in paragraphs:
    bounding_box, _ = paragraph
    rescaled_bounding_box = [[int(value/shrinkfactor) for value in pt] for pt in bounding_box]
    ## Ensure rescaled bouding box is within image bounds
    rescaled_bounding_box = [[max(0,min(w-1,pt[0])), max(0,min(h-1,pt[1]))] for pt in rescaled_bounding_box]
    paragraph[0] = rescaled_bounding_box

  logging.debug(f'OCR complete.')
  return paragraphs


def remove_punctuation(text):
  """
  Removes punctuation characters from the beginning and end of a string.

  Parameters:
  text (str): The input text.

  Returns:
  str: The text with punctuation characters removed from the beginning and end.
  """
  # Define punctuation characters
  punctuation = string.punctuation+"e" ## OCR commonly views ES upsidedown ? as e

  # Remove punctuation from the beginning of the text
  start_index = 0
  for char in text:
    if char in punctuation:
      start_index += 1
    else:
      break

  # Remove punctuation from the end of the text
  end_index = len(text)
  for char in text[::-1]:
    if char in punctuation:
      end_index -= 1
    else:
      break

  return text[start_index:end_index]

def detect_language(text, language_dicts):
  """
  Detects the most likely intended language for the given text.

  Parameters:
  text (str): The input text.
  language_dicts (dict): A dictionary containing language codes as keys
                          and SpellChecker instances as values.

  Returns:
  tuple: A tuple containing the most likely language code and its accuracy. Accuracy will be 2 if a perfect match.
  """
  if text is None or len(text)==0:
    logging.debug("No text to detect language")
    return None, None, None
  # Initialize variables to store counts for each language
  language_counts = {}

  # Count the number of correctly spelled words for each language
  for lang_code, spell_checker in language_dicts.items():
    num_matching = sum(1 for word in text.split() if word in spell_checker)
    corrected = [spell_checker.correction(word) for word in text.split()]
    ## Remove numbers as they are a common OCR misread. Clean none so we can return as string.
    corrected = ['None' if (elem is None or re.match(r'^\d+$', elem) ) else elem for elem in corrected]
    num_in_dist = len([elem for elem in corrected if elem != 'None'])
    
    language_counts[lang_code] = []
    ## Perfect matches will show up in both lists, therefore worth double partial matches
    language_counts[lang_code].append(num_matching + num_in_dist) ## accuracy
    language_counts[lang_code].append(corrected) ## corrected text
    
    if num_matching == 1:
      logging.debug('Ending lang check early.') 
      break
    # ic.ic(language_counts[lang_code] / len(text.split()))

  # Determine the language with the highest count
  most_likely_language = max(language_counts, key=language_counts.get)
  word_score = language_counts[most_likely_language][0]
  num_words = len(text.split())
  accuracy =  num_words and word_score / num_words or 1 ## div1 (if numwords is 0 returns 1)
  corrected_text = language_counts[most_likely_language][1]

  return most_likely_language, ' '.join(corrected_text), accuracy

def main():
  logging.debug(f'RUNNING MAIN {__file__}')
  url = 'https://scontent-sjc3-1.xx.fbcdn.net/v/t39.35426-6/434676141_1594946954586319_7322015044542239282_n.jpg?stp=dst-jpg_s600x600&_nc_cat=100&ccb=1-7&_nc_sid=c53f8f&_nc_ohc=i7RJa6w96MEAX_vWKW2&_nc_ht=scontent-sjc3-1.xx&oh=00_AfDKgfghAGZDNgM4HGoGac0ezIVt54jGxGUERRh_j_k_Bw&oe=66114B24'
  image_cv = url_to_image_cv2( url )
  set_image_env('py')
  nbimshow(image_cv)

if __name__=='__main__':
  logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
  main()