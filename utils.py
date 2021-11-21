# System-level utilities
import os
# For downloading pre-trained models
import urllib.request
from urllib.error import HTTPError


# Small function for printing the test scores
def print_results(result_dict):
    if "train" in result_dict:
        print("Train accuracy: %4.2f%%" % (100.0 * result_dict["train"]))
    if "val" in result_dict:
        print("Val accuracy:   %4.2f%%" % (100.0 * result_dict["val"]))
    print("Test accuracy:  %4.2f%%" % (100.0 * result_dict["test"]))


def download_pretrained_weights(exist_ok, checkpoint_path, base_url, pretrained_files):
    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = os.path.join(checkpoint_path, file_name)
        if "/" in file_name:
            os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=exist_ok)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print("Downloading %s..." % file_url)
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print(
                    "Something went wrong. Please try to download the file from the GDrive folder,"
                    " or contact the author with the full output including the following error:\n",
                    e,
                )
