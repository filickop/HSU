from torch.utils.data import DataLoader

from dataloaders.json_dataset import JsonDataset
from dataset.PairKeypointDataset import PairKeypointDataset
from models.LoFTR.LoFTR import LoFTRRunner
from models.SuperPointSuperGlue.SuperPointSuperGlueMatcher import SuperPointSuperGlueMatcher
from dataset.DatasetGenerator import DatasetGenerator
from dataloaders.glue import GlueDataset
from models.ASpanFormer.aspanformer import ASpanFormerModel
from evaluator import run_tests

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

glue_dataset = GlueDataset()
note_dataset = JsonDataset(
        json_path="dataset/Note/note_000.json",
        image_dir="dataset/Note/images",
        long_dim=200,
    )


lepidla1_dataset = JsonDataset(
        json_path="dataset/Lepidla/lepidla1/keypoints/lepidla1.json",
        image_dir="dataset/Lepidla/lepidla1/images",
        long_dim=1024,
    )

lepidla2_dataset = JsonDataset(
        json_path="dataset/Lepidla/lepidla2/keypoints/lepidla2.json",
        image_dir="dataset/Lepidla/lepidla2/images",
        long_dim=1024,
    )

lepidla3_dataset = JsonDataset(
        json_path="dataset/Lepidla/lepidla3/keypoints/lepidla3.json",
        image_dir="dataset/Lepidla/lepidla3/images",
        long_dim=1024,
    )

lepidla4_dataset = JsonDataset(
        json_path="dataset/Lepidla/lepidla4/keypoints/lepidla4.json",
        image_dir="dataset/Lepidla/lepidla4/images",
        long_dim=1024,
    )

lepidla5_dataset = JsonDataset(
        json_path="dataset/Lepidla/lepidla5/keypoints/lepidla5.json",
        image_dir="dataset/Lepidla/lepidla5/images",
        long_dim=1024,
    )


#MZ dorob ako sa ma toto pouzivat vsetky pripady ake chces
superpoint_model = SuperPointSuperGlueMatcher(
    'dataset/Lepidla/lepidla1/keypoints/lepidla1.json',
    'dataset/Lepidla/lepidla1/lepidla1.jpg',)


loftr_indoor_model = LoFTRRunner(pretrained='indoor')
loftr_outdoor_model = LoFTRRunner(pretrained='outdoor')

aspanformer_model = ASpanFormerModel()

#---------------------------------------------------------------------------------------
#SuperPoint testy START
#---------------------------------------------------------------------------------------
if  1 == 2:
    print('SuperPoint_Note_test')
    result = run_tests(superpoint_model, note_dataset, print_output=False)
    print(result)

    print('SuperPoint_Glue_test')
    result = run_tests(superpoint_model, glue_dataset, print_output=False)
    print(result)

    print('SuperPoint_Lepidla1_test')
    result = run_tests(superpoint_model, lepidla1_dataset, print_output=False)
    print(result)

    print('SuperPoint_Lepidla2_test')
    result = run_tests(superpoint_model, lepidla2_dataset, print_output=False)
    print(result)

    print('SuperPoint_Lepidla3_test')
    result = run_tests(superpoint_model, lepidla3_dataset, print_output=False)
    print(result)

    print('SuperPoint_Lepidla4_test')
    result = run_tests(superpoint_model, lepidla4_dataset, print_output=False)
    print(result)

    print('SuperPoint_Lepidla5_test')
    result = run_tests(superpoint_model, lepidla5_dataset, print_output=False)
    print(result)

#---------------------------------------------------------------------------------------
#SuperPoint testy END
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
#LoFTR testy START
#---------------------------------------------------------------------------------------
print('LoFTR_indoor_Note_test')
result = run_tests(loftr_indoor_model, note_dataset, print_output=False)
print(result)

print('LoFTR_indoor_Glue_test')
result = run_tests(loftr_indoor_model, glue_dataset, print_output=False)
print(result)

print('LoFTR_indoor_Lepidla1_test')
result = run_tests(loftr_indoor_model, lepidla1_dataset, print_output=False)
print(result)

print('LoFTR_indoor_Lepidla2_test')
result = run_tests(loftr_indoor_model, lepidla2_dataset, print_output=False)
print(result)

print('LoFTR_indoor_Lepidla3_test')
result = run_tests(loftr_indoor_model, lepidla3_dataset, print_output=False)
print(result)

print('LoFTR_indoor_Lepidla4_test')
result = run_tests(loftr_indoor_model, lepidla4_dataset, print_output=False)
print(result)

print('LoFTR_indoor_Lepidla5_test')
result = run_tests(loftr_indoor_model, lepidla5_dataset, print_output=False)
print(result)

print('LoFTR_outdoor_Note_test')
result = run_tests(loftr_outdoor_model, note_dataset, print_output=False)
print(result)

print('LoFTR_outdoor_Glue_test')
result = run_tests(loftr_outdoor_model, glue_dataset, print_output=False)
print(result)

print('LoFTR_outdoor_Lepidla1_test')
result = run_tests(loftr_outdoor_model, lepidla1_dataset, print_output=False)
print(result)

print('LoFTR_outdoor_Lepidla2_test')
result = run_tests(loftr_outdoor_model, lepidla2_dataset, print_output=False)
print(result)

print('LoFTR_outdoor_Lepidla3_test')
result = run_tests(loftr_outdoor_model, lepidla3_dataset, print_output=False)
print(result)

print('LoFTR_outdoor_Lepidla4_test')
result = run_tests(loftr_outdoor_model, lepidla4_dataset, print_output=False)
print(result)

print('LoFTR_outdoor_Lepidla5_test')
result = run_tests(loftr_outdoor_model, lepidla5_dataset, print_output=False)
print(result)

#---------------------------------------------------------------------------------------
#LoFTR testy END
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
#AspanFormer testy START
#---------------------------------------------------------------------------------------

print('AspanFormer_Note_test')
result = run_tests(aspanformer_model, note_dataset, print_output=False)
print(result)

print('AspanFormer_Glue_test')
result = run_tests(aspanformer_model, glue_dataset, print_output=False)
print(result)

print('AspanFormer_Lepidla1_test')
result = run_tests(aspanformer_model, lepidla1_dataset, print_output=False)
print(result)

print('AspanFormer_Lepidla2_test')
result = run_tests(aspanformer_model, lepidla2_dataset, print_output=False)
print(result)

print('AspanFormer_Lepidla3_test')
result = run_tests(aspanformer_model, lepidla3_dataset, print_output=False)
print(result)

print('AspanFormer_Lepidla4_test')
result = run_tests(aspanformer_model, lepidla4_dataset, print_output=False)
print(result)

print('AspanFormer_Lepidla5_test')
result = run_tests(aspanformer_model, lepidla5_dataset, print_output=False)
print(result)

#---------------------------------------------------------------------------------------
#AspanFormer testy END
#---------------------------------------------------------------------------------------



