from typing import Final, Dict
from camconv.typedef import *
import numpy as np

k_ref_convention: Final[str] = "LEFT_UP_FRONT"
k_ref_axes: Final[Dict[CubeFace, Tuple[float, float, float]]] = get_reference_axes(k_ref_convention)


def get_transform_to_ref(convention: str) -> np.ndarray:
    '''
    Get the 3x3 matrix mapping from the input convention to the 
    reference one.
    '''
    x, y, z = split_axes_convention(convention)
    return np.column_stack((k_ref_axes[x], k_ref_axes[y], k_ref_axes[z]))


k_camera_to_ref: Final[Dict[CoordinateSystem, np.ndarray]] = {
    CoordinateSystem.REFERENCE: get_transform_to_ref(k_ref_convention),
    CoordinateSystem.PYTORCH3D: get_transform_to_ref("LEFT_UP_FRONT"),
    CoordinateSystem.COLMAP:    get_transform_to_ref("RIGHT_DOWN_FRONT"),
    CoordinateSystem.OPENCV:    get_transform_to_ref("RIGHT_DOWN_FRONT"),
    CoordinateSystem.OPENGL:    get_transform_to_ref("RIGHT_UP_BACK"),
    CoordinateSystem.NGP:       get_transform_to_ref("RIGHT_UP_BACK")
}

k_world_to_ref: Final[Dict[CoordinateSystem, np.ndarray]] = {
    CoordinateSystem.NGP: get_transform_to_ref("FRONT_LEFT_UP")
}


def convert_vertices(verts_in: np.ndarray, system_in: CoordinateSystem, system_out: CoordinateSystem) -> np.ndarray:
    '''
    Transform vertices (N x 3) between different coordinate systems
    '''
    t_ref_win = k_world_to_ref.get(system_in, k_camera_to_ref[system_in])
    t_ref_wout = k_world_to_ref.get(system_out, k_camera_to_ref[system_out])
    t_wout_ref = np.linalg.inv(t_ref_wout)

    verts_out = np.linalg.multi_dot((t_wout_ref, t_ref_win, verts_in.T)).T
    return verts_out

def convert_pose(r_in: np.ndarray, t_in: np.ndarray, system_in: CoordinateSystem, system_out: CoordinateSystem) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Convert pose from one coordinate system to another.
    '''
    t_ref_cin = k_camera_to_ref[system_in]
    t_ref_win = k_world_to_ref.get(system_in, t_ref_cin)
    t_cin_ref = np.linalg.inv(t_ref_cin)

    t_ref_cout = k_camera_to_ref[system_out]
    t_ref_wout = k_world_to_ref.get(system_out, t_ref_cout)
    t_wout_ref = np.linalg.inv(t_ref_wout)

    r_out = np.linalg.multi_dot((t_wout_ref, t_ref_win, r_in, t_cin_ref, t_ref_cout))
    t_out = np.linalg.multi_dot((t_wout_ref, t_ref_win, t_in))
    return r_out, t_out