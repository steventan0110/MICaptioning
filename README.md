# MICaptioning

## Project Set-up:
- first download all required packages
    - create python virtualenv for the project first
    - then pip install the requirement.txt
- Once packages are setup, run `data_download.py`, the evaluation code in it is not important, just run download lines. It might be helpful to download that embedding file as well.
    ```
    from bioCaption.data.downloads import DownloadData
    downloads = DownloadData()
    downloads.download_iu_xray()
    ```
- Once data are donwloaded, run `build_dataset_folder.py` to automatically create the train/valid/test folders and populate with images and captions. This is required for dataloader to work properly.