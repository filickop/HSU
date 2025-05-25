from models.SuperPointSuperGlue.SuperPointSuperGlueMatcher import SuperPointSuperGlueMatcher
from dataset.DatasetGenerator import DatasetGenerator

datasetGenerator = DatasetGenerator(
    'dataset/Note/note_000.png',
    'dataset/Note/note_000.json',
    'dataset/Note/images',
    'dataset/Note/keypoints/note_keypoints.json')

datasetGenerator.generate(100)

developer = ""
if developer == "MZ":
    matcher = SuperPointSuperGlueMatcher(
        'dataset/Note/keypoints/note_keypoints.json',
        'dataset/Note/note_000.png')

    img2, mkpt0, mkpt1 = matcher.match(2)
    mkpts0_0, mkpts0_1 = matcher.select_nearest_keypoint(mkpt0, mkpt1, (90, 70))
    matcher.visualize(img2, mkpts0_0, mkpts0_1)