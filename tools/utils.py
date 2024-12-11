import os
import shutil
from google.colab import drive

def save_model_to_gdrive(model_id="genmo/mochi-1-preview"):
  """
  Saves the downloaded model to Google Drive.

  Args:
    model_id: The ID of the model to save.
  """

  # Check if running in Google Colab
  if 'google.colab' in str(get_ipython()):
    print("Running on Google Colab. Mounting Google Drive.")
    drive.mount('/content/drive')

    # Construct the correct cache directory path
    cache_dir = f"/root/.cache/huggingface/hub/models--{model_id.replace('/', '--')}"

    # Destination path in Google Drive
    dest_path = f"/content/drive/MyDrive/models/{model_id.replace('/', '_')}"

    # Copy the model to Google Drive
    try:
      shutil.copytree(cache_dir, dest_path)
      print(f"Model successfully copied to {dest_path}")
    except FileExistsError:
      print(f"Model already exists in {dest_path}")
    except Exception as e:
      print(f"An error occurred: {e}")
  else:
    print("Not running on Google Colab. Model saving skipped.")

# Call the function after downloading the model (make sure pipe is loaded first)
save_model_to_gdrive(MODEL_PRE_TRAINED_ID)