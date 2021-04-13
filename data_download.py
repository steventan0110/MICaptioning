# from bioCaption.data.downloads import DownloadData
from bioCaption.models.captionModels.baselines import Baselines
from bioCaption.models.captionModels.caption_models_evaluation import CaptionsEvaluation

# downloads = DownloadData()
# # download the iu_xray dataset in the current directory
# downloads.download_iu_xray()

baselines = Baselines('iu_xray/train_images.tsv','iu_xray/test_images.tsv','iu_xray/iu_xray_images/','results')
baselines.most_frequent_word_in_captions()

evaluation = CaptionsEvaluation('iu_xray/test_images.tsv', 'results/most_frequent_word_results.json')

# if the directory "embeddings" does not exits, it will be created
# and the embeddings will be downloaded there.
evaluation.compute_WMD('embeddings/', embeddings_file="pubmed2018_w2v_200D.bin")
evaluation.compute_ms_coco()