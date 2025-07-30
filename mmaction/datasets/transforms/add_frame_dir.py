# mmaction/datasets/transforms/add_frame_dir.py

from mmcv.transforms import BaseTransform
from mmaction.registry import TRANSFORMS

@TRANSFORMS.register_module()
class AddFrameDirToMeta(BaseTransform):
    """frame_dir을 DataSample.metainfo에 추가하는 transform.
    
    ST-GCN 기반 임베딩 추출이나 로그 추적 시 sample 식별에 유용함.
    """

    def transform(self, results: dict) -> dict:
        if 'frame_dir' in results:
            if 'metainfo' not in results:
                results['metainfo'] = {}
            results['metainfo']['frame_dir'] = results['frame_dir']
        return results
