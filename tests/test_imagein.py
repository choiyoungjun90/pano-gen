import numpy as np

from panogen.imagein import is_panorama, photo_to_equirect, prepare


def test_panorama_detection():
    assert is_panorama(2048, 1024) and is_panorama(4000, 1980)
    assert not is_panorama(4032, 3024) and not is_panorama(1080, 1920)


def test_photo_projects_to_front_center():
    photo = np.zeros((300, 400, 3), np.uint8)
    photo[140:160, 190:210] = 255                       # 사진 가운데 흰 점
    erp, mask = photo_to_equirect(photo, 512, 256, hfov=90)
    assert erp[128, 256].min() > 200                     # equirect 정면 가운데로
    assert mask[128, 256] == 0 and mask[128, 0] == 255   # 정면은 보존, 뒤쪽은 생성
    cols = np.nonzero(mask[128] == 0)[0]                 # 화각 90° → 경도 ±45° ≈ 가로 1/4
    assert abs(len(cols) / 512 - 0.25) < 0.03


def test_prepare_modes():
    pano = prepare(np.zeros((500, 1000, 3), np.uint8), 256, 128)
    assert pano.mode == "img2img" and pano.image.shape == (128, 256, 3) and pano.strength == 0.6
    photo = prepare(np.zeros((300, 400, 3), np.uint8), 256, 128, strength=0.0, fov=70)
    assert photo.mode == "outpaint" and photo.mask.shape == (128, 256) and photo.fov == 70 and photo.strength == 0.0
