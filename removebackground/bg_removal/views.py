from django.shortcuts import render
from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from .models import UploadedImage
from rembg import remove
from PIL import Image
import io

# Create your views here.
def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()

            # Process the image
            with open(uploaded_image.original_image.path, 'rb') as input_file:
                input_data = input_file.read()
                output_data = remove(input_data)
            
            # Save the processed image
            processed_image_path = uploaded_image.original_image.path.replace('uploads', 'processed')
            with open(processed_image_path, 'wb') as output_file:
                output_file.write(output_data)

            uploaded_image.processed_image = processed_image_path.replace('media/', '')
            uploaded_image.save()

            return redirect('image_result', pk=uploaded_image.pk)
    else:
        form = ImageUploadForm()
    return render(request, 'upload_image.html', {'form': form})
        
    


def image_result(request, pk):
    uploaded_image = UploadedImage.objects.get(pk=pk)
    return render(request, 'image_result.html', {'image': uploaded_image})

