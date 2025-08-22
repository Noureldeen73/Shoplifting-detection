from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
import os
import json
from .utils import process_video_for_inference, get_model_info
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Create your views here.

# ...existing code...
def index(request):
    return render(request, 'polls/index.html')

@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join('media', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the video file
        save_path = os.path.join(upload_dir, video_file.name)
        with open(save_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)
        
        try:
            # Process video using our utility function with 3D Convolution model
            logger.info(f"Processing video with 3D Convolution model: {save_path}")
            
            # Use the 3D Convolution model
            prediction_result = process_video_for_inference(save_path)
            
            # Check if there was an error during processing
            if 'error' in prediction_result:
                result = {
                    'success': False,
                    'error': prediction_result['error'],
                    'filename': video_file.name,
                    'file_path': save_path
                }
            else:
                # Successful prediction
                result = {
                    'success': True,
                    'message': 'Video analyzed successfully with 3D Convolution model',
                    'filename': video_file.name,
                    'file_path': save_path,
                    'theft_detected': prediction_result['theft_detected'],
                    'confidence': prediction_result['confidence'],
                    'prediction_probability': prediction_result['prediction_probability'],
                    'frames_processed': prediction_result.get('frames_processed', 0),
                    'model_used': prediction_result.get('model_used', '3D Convolution')
                }
                
            logger.info(f"3D Convolution analysis complete: {result}")
            
        except Exception as e:
            logger.error(f"Error during 3D Convolution video analysis: {str(e)}")
            result = {
                'success': False,
                'error': f'Analysis failed: {str(e)}',
                'filename': video_file.name,
                'file_path': save_path,
                'theft_detected': False,
                'confidence': 0.0
            }
        
        return JsonResponse(result)
    
    elif request.method == 'GET':
        # Return the HTML template for video upload
        return render(request, 'polls/upload_video.html')
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)


def model_info(request):
    """
    View to display information about the LSTM model
    """
    try:
        model_info_data = get_model_info()
        
        return JsonResponse({
            'model_info': model_info_data,
            'status': 'LSTM model configured for theft detection'
        })
        
    except Exception as e:
        return JsonResponse({'error': f'Failed to get model info: {str(e)}'}, status=500)
