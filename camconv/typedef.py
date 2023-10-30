from typing import Dict, Tuple, Final
from enum import Enum

class CubeFace(Enum):
    FRONT = 0
    RIGHT = 1
    BACK = 2
    LEFT = 3
    UP = 4
    DOWN = 5


k_opposite_face_table: Dict[CubeFace, CubeFace] = {
    CubeFace.FRONT: CubeFace.BACK,
    CubeFace.RIGHT: CubeFace.LEFT,
    CubeFace.BACK:  CubeFace.FRONT,
    CubeFace.LEFT:  CubeFace.RIGHT,
    CubeFace.UP:    CubeFace.DOWN,
    CubeFace.DOWN:  CubeFace.UP
}


class CoordinateSystem(Enum):
    REFERENCE = 0
    PYTORCH3D = 1
    OPENCV = 2
    COLMAP = 3
    OPENGL = 4
    NGP = 5

def split_axes_convention(convention: str) -> Tuple[CubeFace, CubeFace, CubeFace]:
    '''
    With a convention specified as a string ('LEFT_UP_FRONT' -> 'x_y_z'),
    get the corresponding CubeFace enums.
    '''
    splits = convention.upper().split('_')
    assert len(splits) == 3
    for split in splits:
        assert split in ['FRONT', 'BACK', 'LEFT', 'RIGHT', 'UP', 'DOWN']
    return CubeFace(splits[0]), CubeFace(splits[1]), CubeFace(splits[2])


def get_reference_axes(ref_convention: str) -> Dict[CubeFace, Tuple[float, float, float]]:
    '''
    Get the reference coordinates of the cube directions
    '''
    x, y, z = split_axes_convention(ref_convention)
    return {
        x: (1.0, 0.0, 0.0),
        y: (0.0, 1.0, 0.0),
        z: (0.0, 0.0, 1.0),
        k_opposite_face_table[x]: (-1.0, 0.0, 0.0),
        k_opposite_face_table[y]: (0.0, -1.0, 0.0),
        k_opposite_face_table[z]: (0.0, 0.0, -1.0)
    }

