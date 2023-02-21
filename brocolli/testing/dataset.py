import os
from torchvision import datasets
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from urllib.error import URLError


class ImageNetDatasetValCHINA(datasets.ImageFolder):
    mirrors = ["http://120.224.26.73:15030/aifarm/imagenet/"]

    resources = [("imagenet-mini-val.zip", "9634a5c9a077baf161dd436bbe98f9fe")]

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    def __init__(self, root: str, transform=None, download=False):
        self.root = root
        self.transform = transform

        if not self._check_exist() and download:
            self.download()

        if not self._check_exist():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )
        super(ImageNetDatasetValCHINA, self).__init__(self.processed_folder, transform)

    def _check_exist(self):
        return os.path.exists(self.processed_folder)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "val")

    def _check_exists(self) -> bool:
        return all(
            check_integrity(
                os.path.join(
                    self.raw_folder, os.path.splitext(os.path.basename(url))[0]
                )
            )
            for url, _ in self.resources
        )

    def download(self) -> None:
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_and_extract_archive(
                        url, download_root=self.raw_folder, filename=filename, md5=md5
                    )
                except URLError as error:
                    print("Failed to download (trying next):\n{}".format(error))
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))
