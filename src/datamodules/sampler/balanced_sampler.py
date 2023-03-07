from typing import Iterator

import numpy as np
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(
        self,
        labels: list[int],
        label_set: list[int],
        label_idxs: list[list[int]],
        n_classes: int,
        n_samples: int,
        episode_per_epoch: int,
    ) -> None:
        self.labels = labels
        # self.labels_set = list(set(self.labels.numpy()))
        self.labels_set = label_set
        # self.label_to_indices = {
        #     label: np.where(self.labels.numpy() == label)[0]
        #     for label in self.labels_set
        # }
        self.label_to_indices = {
            label: np.array(label_idxs[i]) for i, label in enumerate(self.labels_set)
        }
        for li in self.labels_set:
            np.random.shuffle(self.label_to_indices[li])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes
        self.episode_per_epoch = episode_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        # self.count = 0
        # print(self.count, self.batch_size, self.n_dataset)
        # print(self.count + self.batch_size < self.n_dataset)
        # while self.count + self.batch_size < self.n_dataset:
        for _ in range(self.episode_per_epoch):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            for class_ in classes:
                np.random.shuffle(self.label_to_indices[class_])
            indices: list[int] = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                        self.used_label_indices_count[
                            class_
                        ] : self.used_label_indices_count[class_]
                        + self.n_samples
                    ]
                )
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(
                    self.label_to_indices[class_]
                ):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            # self.count += self.n_classes * self.n_samples

    def __len__(self) -> int:
        # return self.n_dataset // self.batch_size
        return self.episode_per_epoch
