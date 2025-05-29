from torch.utils.data import DataLoader

from dataset.PairKeypointDataset import PairKeypointDataset
from models.LoFTR.LoFTR import LoFTRRunner
from models.SuperPointSuperGlue.SuperPointSuperGlueMatcher import SuperPointSuperGlueMatcher
from dataset.DatasetGenerator import DatasetGenerator
from dataloaders.glue import GlueDataset
from dataloaders.json_dataset import JsonDataset
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

datasetGenerator.generate(100)
developer = ""
if developer == "MZ":
    model = SuperPointSuperGlueMatcher(
        'dataset/Lepidla/lepidla1/keypoints/lepidla1.json',
        'dataset/Lepidla/lepidla1/lepidla1.jpg',)

    # img2, mkpt0, mkpt1 = matcher.match(4)
    # mkpts0_0, mkpts0_1 = matcher.select_nearest_keypoint(mkpt0, mkpt1, (600, 595))
    # matcher.visualize(img2, mkpts0_0, mkpts0_1)
    dataset = GlueDataset()
    result = run_tests(model, dataset)
    print(result)

elif developer == "PF":
    dataset = PairKeypointDataset(
        ref_img="dataset/Note",
        ref_json="dataset/Note/note_000.json",
        json_path="dataset/Note/keypoints/note_keypoints.json",
        image_dir="dataset/Note/images"
    )

    # dataset = PairKeypointDataset(
    #     ref_img='dataset/Lepidla/lepidla1',
    #     ref_json='dataset/Lepidla/lepidla1/lepidla1.json',
    #     json_path="dataset/Lepidla/lepidla1/keypoints/lepidla1.json",
    #     image_dir="dataset/Lepidla/lepidla1/images"
    # )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    runner = LoFTRRunner(pretrained='indoor')

    # dataset2 = GlueDataset()
    #
    # result = run_tests(runner, dataset2)
    # print(result)

    for i, batch in enumerate(dataloader):
        runner.run_batch(batch)
elif developer == "RZ":
    # model = SuperPointSuperGlueMatcher(
    #     'dataset/Lepidla/lepidla1/keypoints/lepidla1.json',
    #     'dataset/Lepidla/lepidla1/lepidla1.jpg',)
    # model = LoFTRRunner()
    model = ASpanFormerModel("outdoor")

    # dataset = GlueDataset(long_dim=1024, dataset_size=200)
    # dataset = JsonDataset(
    #     json_path="dataset/Note/keypoints/note_keypoints.json",
    #     image_dir="dataset/Note/images",
    #     long_dim=256,
    #     dataset_size=100,
    # )
    dataset = JsonDataset(
        json_path=[f"dataset/Lepidla/lepidla{x}/keypoints/lepidla{x}.json" for x in range(1, 6)],
        image_dir=[f"dataset/Lepidla/lepidla{x}/images" for x in range(1, 6)],
        long_dim=1024,
        dataset_size=500,
    )

    result = run_tests(model, dataset)
    print(result)
