from torch.utils.data import DataLoader

from dataloaders.json_dataset import JsonDataset
from dataset.PairKeypointDataset import PairKeypointDataset
from models.LoFTR.LoFTR import LoFTRRunner
from models.SuperPointSuperGlue.SuperPointSuperGlueMatcher import SuperPointSuperGlueMatcher
from dataset.DatasetGenerator import DatasetGenerator
from dataloaders.glue import GlueDataset
from models.ASpanFormer.aspanformer import ASpanFormerModel
from evaluator import run_tests
from pathlib import Path

datasetGenerator = DatasetGenerator(
    [
        'dataset/Note/note_000.png',
        'dataset/Lepidla/lepidla1/lepidla1.jpg',
        'dataset/Lepidla/lepidla2/lepidla2.jpg',
        'dataset/Lepidla/lepidla3/lepidla3.jpg',
        'dataset/Lepidla/lepidla4/lepidla4.jpg',
        'dataset/Lepidla/lepidla5/lepidla5.jpg'
    ],
    [
        'dataset/Note/note_000.json',
        'dataset/Lepidla/lepidla1/lepidla1.json',
        'dataset/Lepidla/lepidla2/lepidla2.json',
        'dataset/Lepidla/lepidla3/lepidla3.json',
        'dataset/Lepidla/lepidla4/lepidla4.json',
        'dataset/Lepidla/lepidla5/lepidla5.json'
    ],
    [
        'dataset/Note/images',
        'dataset/Lepidla/lepidla1/images',
        'dataset/Lepidla/lepidla2/images',
        'dataset/Lepidla/lepidla3/images',
        'dataset/Lepidla/lepidla4/images',
        'dataset/Lepidla/lepidla5/images'
    ],
    [
        'dataset/Note/keypoints/note_keypoints.json',
        'dataset/Lepidla/lepidla1/keypoints/lepidla1.json',
        'dataset/Lepidla/lepidla2/keypoints/lepidla2.json',
        'dataset/Lepidla/lepidla3/keypoints/lepidla3.json',
        'dataset/Lepidla/lepidla4/keypoints/lepidla4.json',
        'dataset/Lepidla/lepidla5/keypoints/lepidla5.json'
    ],
    add_noise=True)

datasetGenerator.generate(1000)

glue_dataset = GlueDataset(
    long_dim=1024,
    dataset_size=1000,
)
note_dataset = JsonDataset(
    json_path="dataset/Note/keypoints/note_keypoints.json",
    image_dir="dataset/Note/images",
    long_dim=200,
    dataset_size=1000,
)

lepidla_dataset = JsonDataset(
    json_path=[f"dataset/Lepidla/lepidla{x}/keypoints/lepidla{x}.json" for x in range(1, 6)],
    image_dir=[f"dataset/Lepidla/lepidla{x}/images" for x in range(1, 6)],
    long_dim=1024,
    dataset_size=5000,
)


Path("histograms").mkdir(parents=True, exist_ok=True)

def run_tests_all_datasets(name, model):
    print(f"{name} Note test")
    result = run_tests(model, note_dataset, print_output=True)
    result.save(f"histograms/{name}_Note.npz")
    print(result)

    print(f"{name} Glue test")
    result = run_tests(model, glue_dataset, print_output=True)
    result.save(f"histograms/{name}_Glue.npz")
    print(result)

    print(f"{name} Lepidla test")
    result = run_tests(model, lepidla_dataset, print_output=True)
    result.save(f"histograms/{name}_Lepidla.npz")
    print(result)



#---------------------------------------------------------------------------------------
#SuperPoint testy START
#---------------------------------------------------------------------------------------
if  1 == 2:
    run_tests_all_datasets("SuperPoint",
                           SuperPointSuperGlueMatcher(
                               'dataset/Lepidla/lepidla1/keypoints/lepidla1.json',
                               'dataset/Lepidla/lepidla1/lepidla1.jpg'))
#---------------------------------------------------------------------------------------
#SuperPoint testy END
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
#LoFTR testy START
#---------------------------------------------------------------------------------------
run_tests_all_datasets('LoFTR_indoor', LoFTRRunner(pretrained='indoor'))
run_tests_all_datasets('LoFTR_outdoor', LoFTRRunner(pretrained='outdoor'))
#---------------------------------------------------------------------------------------
#LoFTR testy END
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
#AspanFormer testy START
#---------------------------------------------------------------------------------------
run_tests_all_datasets('ASpanFormer_outdoor', ASpanFormerModel('outdoor'))
run_tests_all_datasets('ASpanFormer_indoor', ASpanFormerModel('indoor'))
#---------------------------------------------------------------------------------------
#AspanFormer testy END
#---------------------------------------------------------------------------------------
