import os
import cv2
import uuid
import json
import shutil
import moviepy
import subprocess
import numpy as np
# import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
import keras.backend as K
import moviepy.editor as mp

from glob import glob
from PIL import Image, ImageFilter
from distutils.command.upload import upload
from tensorflow.keras.utils import CustomObjectScope

from logging import lastResort
from django.conf import settings
from django.contrib import messages
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from django.contrib.sites.shortcuts import get_current_site

from .forms import *
from .models import *
from .metrics import dice_loss, dice_coef, iou

from turtle import back
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Create your views here.


""" Global parameters """
H = 512
W = 512



def uploadTheVideo(request):
    
    if request.method == 'POST':
        video = request.FILES['upload']
       
        Upload.objects.create(upload=video)

        return render(request, 'videos.html')
    return render(request, 'videos.html')


import tensorflow as tf
from tensorflow.keras.layers import Layer
import keras

class TensorFlowOpLayer(Layer):
    def __init__(self, node_def, constants=None, **kwargs):
        super(TensorFlowOpLayer, self).__init__(**kwargs)
        self.node_def = node_def
        self.constants = constants or {}

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return tf.multiply(inputs[0], inputs[1])
        return inputs

    def get_config(self):
        config = super(TensorFlowOpLayer, self).get_config()
        config.update({
            'node_def': self.node_def,
            'constants': self.constants,
        })
        return config

def load_legacy_model():
    """Load model with proper custom objects"""
    model_path = "model_v3/model.h5"
    
    # Enable eager execution
    if not tf.executing_eagerly():
        tf.compat.v1.enable_eager_execution()
    
    # Define all possible custom objects
    custom_objects = {
        'iou': iou,
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'TensorFlowOpLayer': TensorFlowOpLayer,
        'Mul': TensorFlowOpLayer,
    }
    
    try:
        # Register custom objects globally
        tf.keras.utils.get_custom_objects().update(custom_objects)
        
        # Load the model with custom objects scope
        with tf.keras.utils.CustomObjectScope(custom_objects):
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss=dice_loss,
                metrics=[iou, dice_coef]
            )
            
            return model
            
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        try:
            # Alternative approach: Try loading saved model format
            model = tf.saved_model.load(model_path)
            return model
        except Exception as e:
            print(f"Failed to load saved model: {str(e)}")
            try:
                # Last resort: Try loading with minimal custom objects
                return tf.keras.models.load_model(
                    model_path,
                    custom_objects={
                        'iou': iou,
                        'dice_coef': dice_coef,
                        'dice_loss': dice_loss
                    },
                    compile=False
                )
            except Exception as e:
                print(f"All loading attempts failed: {str(e)}")
                return None


def createFrames(request):

    if request.method == 'POST':
        form = Uploadfileform(request.POST, request.FILES)
        
        print(form.is_valid())
        if form.is_valid():
            newvid = Upload(upload=request.FILES['upload'])
            vid_name = str(request.FILES['upload']).replace(' ', '_')
            if is_video_file(vid_name) == False:
                print('hola')
                data = {'tag': 'warning',
                            'message': 'Please Upload a Video File!', 'bool': True}
                return render(request, 'videos.html',data )
            newvid.save()

        new_video_name = str(uuid.uuid4())[:12]+"."+vid_name.split('.')[-1]

        old_name = f"media/videos/"+vid_name
        new_name = f"media/videos/"+new_video_name

        # enclosing inside try-except
        
        try:
            os.rename(old_name, new_name)
        except:
            # messages.warning(request, 'Incorrect Email and Password.')
            print("File already Exists")
            print("Removing existing file")
            # skip the below code
            # if you don't' want to forcefully rename
            # os.remove(new_name)
            # # rename it
            # os.rename(old_name, new_name)
            # print('Done renaming a file')

        """ Seeding """
        np.random.seed(42)
        tf.random.set_seed(42)

        """ Directory for storing files """
        create_dir("frames")
        create_dir("processed_videos")
        create_dir("audio")
        create_dir("overlay")

        # """ Loading model: DeepLabV3+ """
        # with CustomObjectScope({
        #     'iou': iou,
        #     'dice_coef': dice_coef,
        #     'dice_loss': dice_loss
        # }):
        #     model = tf.keras.models.load_model("model_v3/model.h5")
        
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
    
        # Try loading the model
        model = load_legacy_model()
        if model is None:
            raise Exception("Failed to load model")

        # Create prediction function
        @tf.function
        def predict_mask(x):
            return model(x, training=False)
        
        
        # video = Upload.objects.get().last()
        """ Video Path """
        video_path = new_name


        """ Extracting the audio from video """
        # Replace the parameter with the location of the video
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        # Replace the parameter with the location along with filename
        audio.write_audiofile("audio/sample.mp3")

        """ Reading frames """
        vs = cv2.VideoCapture(video_path)
        _, frame = vs.read()
        h, w, _ = frame.shape

        """ Resize the white transparent image """
        image = Image.open('processed_videos/2000x2000.png')
        new_image = image.resize((w, h))
        new_image.save('processed_videos/transparent.png')
        vs.release()

        cap = cv2.VideoCapture(video_path)
        idx = 0

        cap = cv2.VideoCapture(new_name)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if ret == False:
                cap.release()
                break

            h, w, _ = frame.shape
            ori_frame = frame
            frame = cv2.resize(frame, (W, H))
            frame = np.expand_dims(frame, axis=0)
            frame = frame / 255.0

            """ Predict the object and removing background from frame """
            mask = model.predict(frame)[0]
            mask = cv2.resize(mask, (w, h))
            mask = mask > 0.5
            mask = mask.astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

            photo_mask = mask
            background_mask = np.abs(1-mask)
            masked_frame = ori_frame * photo_mask

            """ Add Alpha channel for transparancy """
            hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
            RGBA = np.dstack(
                (masked_frame, np.zeros((h, w), dtype=np.uint8)+255))
            mBlack = (RGBA[:, :, 0:3] == [0, 0, 0]).all(2)
            RGBA[mBlack] = (0, 0, 0, 0)
            final_frame1 = RGBA.astype(np.uint8)
            cv2.imwrite(f"overlay/{idx}.png", final_frame1)

            img = cv2.imread(f"overlay/{idx}.png", cv2.IMREAD_UNCHANGED)
            img_alpha = img[:, :, 3]
            thresh = 127
            im_bw = cv2.threshold(img_alpha, thresh, 255, cv2.THRESH_BINARY)[1]
            # cv2.imwrite(f"mask/{idx}.png", im_bw)
            img = np.array(im_bw)
            mean = 0
            # var = 0.1
            # sigma = var**0.5
            gauss = np.random.normal(mean, 1, img.shape)

            # normalize image to range [0,255]
            noisy = img + gauss
            minv = np.amin(noisy)
            maxv = np.amax(noisy)
            noisy = (255 * (noisy - minv) / (maxv - minv)).astype(np.uint8)
           
            """ Conver numpy array  """
            mask = Image.fromarray(noisy)
            with Image.open(f"overlay/{idx}.png") as img:
                img.load()
            mask = mask.convert("L")
           
            """ Bluring the edges """
            mask = mask.filter(ImageFilter.BoxBlur(2))
            blank = img.point(lambda _: 0)
            segmented = Image.composite(img, blank, mask)
            segmented.save(f'frames/{idx}.png')

            idx += 1

            """ Websocket to send percentage of frames """
            per = int((idx/length)*100)
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                'notification_broadcast',
                {
                    'type': 'send_notification',
                    'message': json.dumps(per)
                }
            )
        response_data = {"message": "removed bg from video"}
        return HttpResponse(json.dumps(response_data), content_type="application/json")
        
    error_data = {"message": "FAILED"}
    return HttpResponse(json.dumps(error_data), content_type="application/json")


def mergeFrames(request):
    os.chdir('frames')
    output = str(uuid.uuid4())[:12]

    """ merging the whole frames and create the video """
    frameCommand = [
        " -framerate",
        " 30",
        " -i",
        " %d.png",
        " -vcodec",
        " png",
        f" ../processed_videos/{output}.mp4"
    ]
    
    finalCommand = "ffmpeg" + " ".join(frameCommand)

    os.system(finalCommand)
    os.chdir('../processed_videos/')

    """ set the white background in bgremoved video """
    bg_removed_vid_name = str(uuid.uuid4())[:12]
    # print("Current working directory: {0}".format(os.getcwd()))
    subprocess.call([
        'ffmpeg',
        '-loop',
        '1',
        '-i',
        'transparent.png',
        '-i',
        f'{output}.mp4',
        '-filter_complex',
        'overlay=(W-w)/2:shortest=1',
        f'{bg_removed_vid_name}.mp4'
    ])

    """ add audio """
    videoclip = moviepy.editor.VideoFileClip(f"{bg_removed_vid_name}.mp4")
    audioclip = moviepy.editor.AudioFileClip("../audio/sample.mp3")
    new_audioclip = moviepy.editor.CompositeAudioClip([audioclip])
    videoclip.audio = new_audioclip
    bg_vid_name = str(uuid.uuid4())[:12]
    videoclip.write_videofile(f"../media/videos/{bg_vid_name}.mp4")

    """ remove frames folder """
    os.chdir('../')
    shutil.rmtree('frames')
    shutil.rmtree('overlay')
    if os.path.exists("processed_videos/transparent.png"):
        os.remove("processed_videos/transparent.png")

    current_site = get_current_site(request=request).domain    
    video_file = "http://" + current_site + settings.MEDIA_URL + f"videos/{bg_vid_name}.mp4"
    
    return JsonResponse({'data':video_file})


def home(request):
    if request.method == 'POST':
        video = request.FILES['upload']
        Upload.objects.create(upload=video)

        return render(request, 'videos.html', {"message": "removed bg from video"})
    return render(request, 'videos.html')
    # return render(request, 'videos.html')


""" Creating a directory function """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def is_video_file(filename):
    video_file_extensions = (
'.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2', '.60d', '.787', '.89', '.aaf', '.aec', '.aep', '.aepx',
'.aet', '.aetx', '.ajp', '.ale', '.am', '.amc', '.amv', '.amx', '.anim', '.aqt', '.arcut', '.arf', '.asf', '.asx', '.avb',
'.avc', '.avd', '.avi', '.avp', '.avs', '.avs', '.avv', '.axm', '.bdm', '.bdmv', '.bdt2', '.bdt3', '.bik', '.bin', '.bix',
'.bmk', '.bnp', '.box', '.bs4', '.bsf', '.bvr', '.byu', '.camproj', '.camrec', '.camv', '.ced', '.cel', '.cine', '.cip',
'.clpi', '.cmmp', '.cmmtpl', '.cmproj', '.cmrec', '.cpi', '.cst', '.cvc', '.cx3', '.d2v', '.d3v', '.dat', '.dav', '.dce',
'.dck', '.dcr', '.dcr', '.ddat', '.dif', '.dir', '.divx', '.dlx', '.dmb', '.dmsd', '.dmsd3d', '.dmsm', '.dmsm3d', '.dmss',
'.dmx', '.dnc', '.dpa', '.dpg', '.dream', '.dsy', '.dv', '.dv-avi', '.dv4', '.dvdmedia', '.dvr', '.dvr-ms', '.dvx', '.dxr',
'.dzm', '.dzp', '.dzt', '.edl', '.evo', '.eye', '.ezt', '.f4p', '.f4v', '.fbr', '.fbr', '.fbz', '.fcp', '.fcproject',
'.ffd', '.flc', '.flh', '.fli', '.flv', '.flx', '.gfp', '.gl', '.gom', '.grasp', '.gts', '.gvi', '.gvp', '.h264', '.hdmov',
'.hkm', '.ifo', '.imovieproj', '.imovieproject', '.ircp', '.irf', '.ism', '.ismc', '.ismv', '.iva', '.ivf', '.ivr', '.ivs',
'.izz', '.izzy', '.jss', '.jts', '.jtv', '.k3g', '.kmv', '.ktn', '.lrec', '.lsf', '.lsx', '.m15', '.m1pg', '.m1v', '.m21',
'.m21', '.m2a', '.m2p', '.m2t', '.m2ts', '.m2v', '.m4e', '.m4u', '.m4v', '.m75', '.mani', '.meta', '.mgv', '.mj2', '.mjp',
'.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd', '.moff', '.moi', '.moov', '.mov', '.movie', '.mp21',
'.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1', '.mpeg4', '.mpf', '.mpg', '.mpg2', '.mpgindex', '.mpl',
'.mpl', '.mpls', '.mpsub', '.mpv', '.mpv2', '.mqv', '.msdvd', '.mse', '.msh', '.mswmm', '.mts', '.mtv', '.mvb', '.mvc',
'.mvd', '.mve', '.mvex', '.mvp', '.mvp', '.mvy', '.mxf', '.mxv', '.mys', '.ncor', '.nsv', '.nut', '.nuv', '.nvc', '.ogm',
'.ogv', '.ogx', '.osp', '.otrkey', '.pac', '.par', '.pds', '.pgi', '.photoshow', '.piv', '.pjs', '.playlist', '.plproj',
'.pmf', '.pmv', '.pns', '.ppj', '.prel', '.pro', '.prproj', '.prtl', '.psb', '.psh', '.pssd', '.pva', '.pvr', '.pxv',
'.qt', '.qtch', '.qtindex', '.qtl', '.qtm', '.qtz', '.r3d', '.rcd', '.rcproject', '.rdb', '.rec', '.rm', '.rmd', '.rmd',
'.rmp', '.rms', '.rmv', '.rmvb', '.roq', '.rp', '.rsx', '.rts', '.rts', '.rum', '.rv', '.rvid', '.rvl', '.sbk', '.sbt',
'.scc', '.scm', '.scm', '.scn', '.screenflow', '.sec', '.sedprj', '.seq', '.sfd', '.sfvidcap', '.siv', '.smi', '.smi',
'.smil', '.smk', '.sml', '.smv', '.spl', '.sqz', '.srt', '.ssf', '.ssm', '.stl', '.str', '.stx', '.svi', '.swf', '.swi',
'.swt', '.tda3mt', '.tdx', '.thp', '.tivo', '.tix', '.tod', '.tp', '.tp0', '.tpd', '.tpr', '.trp', '.ts', '.tsp', '.ttxt',
'.tvs', '.usf', '.usm', '.vc1', '.vcpf', '.vcr', '.vcv', '.vdo', '.vdr', '.vdx', '.veg','.vem', '.vep', '.vf', '.vft',
'.vfw', '.vfz', '.vgz', '.vid', '.video', '.viewlet', '.viv', '.vivo', '.vlab', '.vob', '.vp3', '.vp6', '.vp7', '.vpj',
'.vro', '.vs4', '.vse', '.vsp', '.w32', '.wcp', '.webm', '.wlmp', '.wm', '.wmd', '.wmmp', '.wmv', '.wmx', '.wot', '.wp3',
'.wpl', '.wtv', '.wve', '.wvx', '.xej', '.xel', '.xesc', '.xfl', '.xlmv', '.xmv', '.xvid', '.y4m', '.yog', '.yuv', '.zeg',
'.zm1', '.zm2', '.zm3', '.zmv'  )
    print(filename)
    if filename.endswith((video_file_extensions)):
        return True
    else:
        return False
