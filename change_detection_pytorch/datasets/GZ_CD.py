from .custom import CustomDataset


class GZ_CD_Dataset(CustomDataset):
    """GZ-CD (Guangzhou Change Detection) dataset"""

    def __init__(self, img_dir, sub_dir_1='A', sub_dir_2='B', ann_dir=None, img_suffix='.png', seg_map_suffix='.png',
                 transform=None, split=None, data_root=None, test_mode=False, size=256, debug=False):
        super().__init__(img_dir, sub_dir_1, sub_dir_2, ann_dir, img_suffix, seg_map_suffix, transform, split,
                         data_root, test_mode, size, debug)

    # Inherited from CustomDataset.get_default_transform()
    # Inherited from CustomDataset.get_test_transform()

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if not self.ann_dir:
            ann = None
            img1, img2, filename = self.prepare_img(idx)
            transformed_data = self.transform(image=img1, image_2=img2)
            img1, img2 = transformed_data['image'], transformed_data['image_2']
            return img1, img2, filename
        else:
            img1, img2, ann, filename = self.prepare_img_ann(idx)
            transformed_data = self.transform(image=img1, image_2=img2, mask=ann)
            img1, img2, ann = transformed_data['image'], transformed_data['image_2'], transformed_data['mask']
            return img1, img2, ann, filename


if __name__ == "__main__":
    GZ_CD_Dataset('dir')
