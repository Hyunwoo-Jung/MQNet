from .almethod import ALMethod
import numpy as np
import torch

class Random(ALMethod):
    """
    Random querying for Active Learning.
    - U_index에서 args.subset 만큼 부분집합을 만든 뒤,
      그 안에서 args.n_query 개를 무작위로 샘플링하여 반환.
    """
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

        # 재현성: seed가 있으면 cycle 등과 조합해서 고정하고 싶다면 kwargs에서 받아 써도 됨
        self.seed = getattr(args, "seed", 115)
        self.rng = np.random.default_rng(self.seed)

        # subset selection (OOM 방지 및 diversity를 위해 LL과 동일한 방식)
        subset_size = min(getattr(self.args, "subset", len(self.U_index)), len(self.U_index))
        subset_idx = self.rng.choice(len(self.U_index), size=subset_size, replace=False)
        self.U_index_sub = np.array(self.U_index)[subset_idx]

    def run(self):
        if len(self.U_index_sub) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        k = min(getattr(self.args, "n_query", 1), len(self.U_index_sub))
        # 부분집합 인덱스 공간(0..len(U_index_sub)-1)에서 무작위 선택
        selected_idx_sub = self.rng.choice(len(self.U_index_sub), size=k, replace=False)

        # 점수는 의미 없음: 0으로 채움 (후처리 호환용)
        scores = np.zeros(k, dtype=float)
        return selected_idx_sub.astype(int), scores

    def select(self, **kwargs):
        selected_idx_sub, scores = self.run()
        # 부분집합 로컬 인덱스를 글로벌 U_index의 실제 인덱스로 변환
        Q_index = self.U_index_sub[selected_idx_sub]
        return Q_index.tolist(), scores
