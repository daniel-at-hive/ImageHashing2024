from datetime import datetime
import warnings
from ImageLib import *
import pandas as pd

## Globals + Constants
OUTPUT_COLS = [ ### Currently unused
    'img_url', 'upload_date',  # from input
    'height', 'width', 'aspect_ratio',  # from get_general_info
    'language', 'language_accuracy', 'ocr_text', 'corrected_text', 'is_snap_ocr',  # from perform_ocr_and_detect_language
    'num_images', 'h_bars', 'v_bars', 'is_snap_lines',  # from process_collage_and_snapchat
    'dhash_8', 'dhash_8_bin', 'dhash_16', 'phash', 'phash_bin',  # from calculate_image_hashes
    'avg_color', 'palette',  # from analyze_colors
    'failed' # from main, if process_image fails
]
FORMATTED_NOW = datetime.now().strftime('%y%m%d_%H%M')
BATCH_SIZE = 10 ## number of images before writing to file
FILE_SIZE = 100_000 ## number of images before new file
INPUT_FILE = ''
OUTPUT_FILE = f'OUTPUT-{FORMATTED_NOW}.csv'
LOG_FILE = f'LOG-{FORMATTED_NOW}.txt'
URL_RETRIES = 5 
OCR_SCALE_FACTOR = 0.6 ## First try OCR downscaled by this factor
SC_BARS_HEIGHT = 40 
SC_TOLERANCE = 8
COLOR_BIN_SIZE = 5 ## Signficance to use when calculating mode colors
BAR_ROUNDUP = 0.55 ## Percentage threshold for rounding up contours to match the complete height/width of the image 
BAR_ROUNDDOWN = 0.2 ## Percentage threshold for removing contours 


def log_time(func):
    """Decorator to log the time taken by a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.2f} seconds.")
        return result
    return wrapper

@log_time
def read_image_from_url(url):
    """Read image from URL. Will attemp URL_RETRIES times on failure."""
    return retry(url_to_image_cv2, {'url': url}, retries=URL_RETRIES)

@log_time
def get_general_info(image):
    """Get general info about the image."""
    _, h, w, _ = image_info(image)
    ratio_string, _ = nearest_normal_aspect_ratio(h, w)
    return h, w, ratio_string

def ocr_and_detect_language_helper(image, scale_factor=OCR_SCALE_FACTOR):
    """Perform OCR and detect language."""
    ## OCR as separate paragraphs for bounding box info, then recombine text and read
    ocr = eOCR(image, reader, shrinkfactor=scale_factor)
    texts = [text for _, text in ocr]
    bounds = [bounds for bounds, _ in ocr]
    combined_text = ' '.join([remove_punctuation(text) for text in texts])
    most_likely_language, corrected, accuracy = detect_language(remove_punctuation(combined_text), language_dicts)
    return texts, bounds, combined_text, most_likely_language, corrected, accuracy

@log_time
def perform_ocr_and_detect_language(image):
    '''Perform OCR and repeat without scaling if unconfident in the output. Checks for snapchat textbox.'''
    ## Run OCR and retry if necessary
    texts, bounds, combined_text, most_likely_language, corrected, accuracy = ocr_and_detect_language_helper(image)
    if most_likely_language is None: 
        pass
    elif accuracy<1.3:
        logging.debug(f'Poor accuracy, retrying at full scale')
        texts, bounds, combined_text, most_likely_language, corrected, accuracy = ocr_and_detect_language_helper(image, scale_factor=1)
    elif accuracy<0.8:
        logging.info(f'Poor accuracy even at full scale.')
        
    ## Check for snapchat textbox
    ocr_snap_test = False
    if most_likely_language is None: pass
    else:
        bounds_height = [abs(bound[0][1]-bound[2][1]) for bound in bounds]
        for bh in bounds_height:
            if abs(SC_BARS_HEIGHT - bh) < SC_TOLERANCE: 
                ocr_snap_test = True
                break
    return texts, bounds, combined_text, most_likely_language, corrected, accuracy, ocr_snap_test
        
@log_time
def process_collage_and_snapchat(image, h, w):
    """Process collage and snapchat detection."""
    canny1 = cv2.Canny(image=image, threshold1=0, threshold2=255)
    contours_horizontal, contours_vertical = detect_bars(canny1)
    hori_clean = clean_contours(contours_horizontal, 'horizontal', w*BAR_ROUNDUP, w-1, w*BAR_ROUNDDOWN)
    verti_clean = clean_contours(contours_vertical, 'vertical', h*BAR_ROUNDUP, h-1, h*BAR_ROUNDDOWN)
    snapheights, numimages = characterize_bars(verti_clean, hori_clean, h, w)
    contour_snap_test = None
    if snapheights:
        snap_top = min(snapheights[0], snapheights[1])
        snap_bot = max(snapheights[0], snapheights[1])
        contour_snap_test = has_snap_overlay(image, snap_top, snap_bot)
    return numimages, hori_clean, verti_clean, contour_snap_test

@log_time
def calculate_image_hashes(image):
    """Calculate image hashes."""
    dhash_8_bin = diff_hash(image, 8)
    dhash_8 = bin_to_hex(dhash_8_bin, leading_zero_width=8 * 8 / 4)
    dhash_16_bin = diff_hash(image, 16)
    dhash_16 = bin_to_hex(dhash_16_bin, leading_zero_width=16 * 16 / 4)
    phash = perceptual_hash(image)
    phash_bin = hex_to_bin(phash)
    return dhash_8, dhash_8_bin, dhash_16, phash, phash_bin

@log_time
def analyze_colors(image):
    """Analyze colors in the image."""
    onebyone = cv2.resize(image, (1, 1), interpolation=cv2.INTER_LINEAR)
    avg_color = to_hex(onebyone[0][0], format='BGR')
    image_color_compressed = compress_colors(image, color_radius=COLOR_BIN_SIZE)
    mode_colors_ = mode_colors(image_color_compressed)[0:2]
    return avg_color, mode_colors_

@log_time
def process_image(url):
    image_data = {}
    image = read_image_from_url(url)
    if image is None:
        return {}
    
    h, w, ratio_string = get_general_info(image)
    image_data.update({'height': h, 'width': w, 'aspect_ratio': ratio_string})
    
    # _, _, text, lang, corrected, accuracy, snap = perform_ocr_and_detect_language(image)
    # image_data.update({'language': lang, 'language_accuracy': accuracy, 'ocr_text': text, 'corrected_text': corrected, 'is_snap_ocr': snap})
    
    # numimages, hori_clean, verti_clean, contour_snap_test = process_collage_and_snapchat(image, h, w)
    # image_data.update({'num_images': numimages, 'h_bars': hori_clean, 'v_bars': verti_clean, 'is_snap_line': contour_snap_test})
    
    dhash_8, dhash_8_bin, dhash_16, phash, phash_bin = calculate_image_hashes(image)
    image_data.update({'dhash_8': dhash_8, 'dhash_8_bin': dhash_8_bin, 'dhash_16': dhash_16, 'phash': phash, 'phash_bin': phash_bin})
    
    # avg_color, mode_colors_ = analyze_colors(image)
    # image_data.update({'avg_color': avg_color, 'palette': mode_colors_})
    
    image_data.update({'failed': False})
    
    return image_data



def main():
    global OUTPUT_FILE ## sloppily redefining this occassionally
    df = pd.DataFrame()
    for i in range(len(img_urls)):
        logging.info(f"Processing image {i:,}")
        
        ## Process image and append
        new_row = {}
        try:
            new_row = process_image( img_urls[i] )
        except Exception as e:
            logging.error(f'FAILED to process image: {img_urls[i]}\n{e}')
            print(f'FAILED to process image: {img_urls[i]}\n{e}')
            new_row['failed'] = True
            
        if new_row:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_row['img_url'] = img_urls[i]
                new_row['upload_date'] = upload_dates[i]
                new_row['img_names'] = img_names[i]
                new_row_df = pd.DataFrame([new_row])
                df = pd.concat([df, new_row_df], ignore_index=True)
        else: 
            continue
        
        if i == 0:
            df.to_csv(OUTPUT_FILE, mode='w', header=True, index=False)  
            continue
        ## Dump DF to file every BATCH_SIZE images
        elif i%BATCH_SIZE == 0:
            logging.info(f"Appending {len(df)} rows to {OUTPUT_FILE}")
            print(f"Processing image {i:,}")
            df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            df = df.head(0)  
        ## Dump DF to NEW file every FILE_SIZE images
        if i%FILE_SIZE == 0:
            ## Dump
            logging.info(f"Appending {len(df)} rows to {OUTPUT_FILE}")
            df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            df = df.head(0)  
            
            ## Start new file
            OUTPUT_FILE = f"{i/FILE_SIZE}"+OUTPUT_FILE
            df.to_csv(OUTPUT_FILE, mode='w', header=True, index=False)  

    ## Final clean up
    df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)  
    logging.info('Finished final append.')
    return


def input_setup(file=None, start_index=0,end_index=None):
    S3_ROOT = 'https://s3.amazonaws.com/hivemedia-images/creatives/'
    img_urls, upload_dates, img_names = [], [], []
    if file is None:
        img_names = [
        '2a293f7b3d88c205b239e6c880e448b3.jpg', '46ff5ae7be1759292d6a259da42ae3b6.jpg', '884cdda9ceb754dc905d5be768700463.jpg', 'd98389b060ded14206c65abce3ca13d2.jpg', 'e118b149d753a482221ac03bf8475ce7.jpg', 'f02abc4967740835b4bf9a67ae019772.jpg', 'd0ff493397a33fa6cc1c1d633cf064d2.jpg', '5d016b9867a35676d35c186f8716e6c0.jpg', '096691d558c6205f8edae90931294547.jpg', 'c5e9c81fb81ff935d3a81a1f8782a845.jpg', 'b51a7ddc20a5c5b683832474b56a0e0d.jpg', '24b2c40e6859bd9c1be75db5664686e9.jpg', '8d9c6e76c0376b07f76d350d02a50b10.jpg', 'b84b33161c94965cf5d31227db9f77a1.jpg', '57d721c0bbf34f4fa092d0af67774295.jpg', 'a7bb1c90ea65c2185b0cdbb7aec43059.jpg', '326032daacfa3a3f2e297dca21c48fd7.jpg', 'acf2401b9ef648d5c1e6f154d40378d4.jpg', '45664f2a1c56ff91335b42f8a72cdb09.jpg', '9a90b39ed99dbee80213b84547f97daa.jpg', '80dc90e1562f27f38396f58fe6817081.jpg', 'a661a7d08b4e5bda9c41a178be028804.jpg', 'f123fb3758a4bd8287c2cab418cdae38.jpg', 'a7259691cf0d2d9b3ff445b79637f787.jpg', 'c3dd037a71d39af3a7301e49957c99ed.jpg', 'd019947de69a65fd895f0beee649cafa.jpg', 'ee45cdf05d90ffa4e784365cbfc56b43.jpg', '1c5a9f871550808fdccc8fc88d53f192.jpg', '753213c68372ae2006f7a83a9b662061.jpg'
        ]
        img_urls = [f'{S3_ROOT}{i}' for i in img_names]
        img_urls.append('https://scontent-sjc3-1.xx.fbcdn.net/v/t39.35426-6/434676141_1594946954586319_7322015044542239282_n.jpg?stp=dst-jpg_s600x600&_nc_cat=100&ccb=1-7&_nc_sid=c53f8f&_nc_ohc=i7RJa6w96MEAX_vWKW2&_nc_ht=scontent-sjc3-1.xx&oh=00_AfDKgfghAGZDNgM4HGoGac0ezIVt54jGxGUERRh_j_k_Bw&oe=66114B24')
    else:
        df = pd.read_csv(file)
        img_urls=[]
        ## If file has s3 in the name
        if 's3' in file:
            df = df[df['bucket'] == 'creatives']
            df = df[df['file_name'] != '']
            df['upload_date'] = pd.to_datetime(df['upload_date'], format='%m/%d/%Y %H:%M')
        elif 'launches' in file:
            ## rename columns LAUNCH_DATE->upload_date, IMAGE_URL_LAUNCHER_RAW->img_urls
            df = df.rename(columns={'LAUNCH_DATE':'upload_date', 'IMAGE_URL_LAUNCHER_RAW':'img_urls'})
            df['upload_date'] = pd.to_datetime(df['upload_date']) 
        elif 'FB' in file:
            ## rename columns LAUNCH_DATE->upload_date, IMAGE_URL_LAUNCHER_RAW->img_urls
            df = df.rename(columns={'UPLOAD_DATE':'upload_date', 'IMG_URL':'img_urls','IMG_NAME':'img_names'})
            df['upload_date'] = pd.to_datetime(df['upload_date']) 
        
        df = df.sort_values(by='upload_date', ascending=False) 
        df = df.iloc[start_index:end_index]
        
        
        if 's3' in file: img_urls = [f'{S3_ROOT}{i}' for i in img_names]
        elif 'launches' in file: img_urls = df['img_urls'].tolist()
        if 'FB' in file: 
            img_urls = df['img_urls'].tolist()
            img_names = df['img_names'].tolist()
        
        upload_dates = df['upload_date'].tolist()

    return img_urls, upload_dates, img_names


# Initialize logging '%y%m%d_%H%M')}
# logging.basicConfig(level=logging.INFO, format='%(levelname)s|%(funcName)s\t(%(asctime)s) %(message)s', datefmt='%H:%M:%S')
logging.basicConfig(level=logging.INFO, format='%(levelname)s|%(funcName)s\t(%(asctime)s) %(message)s', datefmt='%H:%M:%S', filename=LOG_FILE)
if __name__=='__main__':
    # img_urls, upload_dates = input_setup('launches_no_hash.csv', start_index=0)
    img_urls, upload_dates, img_names = input_setup('FB0410-bad.csv', start_index=126230)
    

    reader = eo.Reader(['es', 'en', 'pt'])
    spell_checker_en = SpellChecker(language='en',distance=1)
    spell_checker_es = SpellChecker(language='es',distance=1)
    spell_checker_pt = SpellChecker(language='pt',distance=1)
    language_dicts = {"EN": spell_checker_en, "ES": spell_checker_es, "PT": spell_checker_pt}
    main()