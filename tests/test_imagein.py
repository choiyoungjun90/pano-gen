import numpy as np

from panogen.imagein import flat_size, is_panorama, photo_to_equirect, prepare, prepare_flat, yaws_for


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


def test_photo_yaw_moves_photo():
    photo = np.full((100, 100, 3), 255, np.uint8)
    _, mask = photo_to_equirect(photo, 360, 180, hfov=60, yaw=90)
    cols = np.nonzero(mask[90] == 0)[0]
    assert abs(cols.mean() - 270) < 3                    # 정면(180) 에서 오른쪽 90° → 270


def test_prepare_multi_photos_and_panorama_priority():
    photos = [np.full((60, 80, 3), v, np.uint8) for v in (50, 150, 250)]
    init = prepare(photos, 360, 180, fov=100)
    assert init.mode == "outpaint" and init.yaws == yaws_for(3) == [0, 120, 240]
    assert init.fov <= 116 and len(init.sources) == 3
    for yaw, v in zip(init.yaws, (50, 150, 250)):          # 각 방향 가운데에 그 사진 색
        x = int((180 + yaw) % 360)
        assert init.mask[90, x] == 0 and abs(int(init.image[90, x, 0]) - v) <= 2
    mixed = prepare([photos[0], np.zeros((100, 200, 3), np.uint8)], 256, 128)
    assert mixed.mode == "img2img" and mixed.unused == 1   # 파노라마가 있으면 그걸 고치고 나머지는 klein 참조로만


def test_flat_size_and_prepare_flat():
    assert flat_size("1:1") == (1024, 1024) and flat_size("16:9") == (1344, 768)
    assert flat_size("auto", np.zeros((1920, 1080, 3), np.uint8)) == (768, 1344)
    assert flat_size("auto") == (1024, 1024) and flat_size("4:3", base=128) == (144, 112)
    init = prepare_flat([np.zeros((300, 600, 3), np.uint8), np.zeros((10, 10, 3), np.uint8)], 128, 128)
    assert init.image.shape == (128, 128, 3) and init.strength == 0.6 and init.unused == 1
