from torch.utils.data import DataLoader

from dataset.Note.NotePairKeypointDataset import NotePairKeypointDataset
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
    ])

datasetGenerator.generate(100)

developer = ""
if developer == "MZ":
    matcher = SuperPointSuperGlueMatcher(
        'dataset/Lepidla/lepidla1/lepidla1.json',
        'dataset/Lepidla/lepidla1/lepidla1.jpg',)

    img2, mkpt0, mkpt1 = matcher.match(4)
    mkpts0_0, mkpts0_1 = matcher.select_nearest_keypoint(mkpt0, mkpt1, (600, 595))
    matcher.visualize(img2, mkpts0_0, mkpts0_1)

elif developer == "PF":
    dataset = NotePairKeypointDataset(
        json_path="dataset/Note/keypoints/note_keypoints.json",
        image_dir="dataset/Note/images"
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    runner = LoFTRRunner()

    for i, batch in enumerate(dataloader):
        runner.run_batch(batch, output_path=f"foundKeypoints/matched_{i:03}.json")

elif developer == "RZ":
    model = ASpanFormerModel()
    dataset = GlueDataset()
    result = run_tests(model, dataset)
    print(result)
