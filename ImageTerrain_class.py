# Imports
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional, Callable
from dm_control import mjcf
from PIL import Image
from skimage import io
from flygym.mujoco.arena.base import BaseArena

# Imports
class ImageTerrain(BaseArena):
    """Terrain with an image displayed.

    Attributes
    ----------
    root_element : mjcf.RootElement
        The root MJCF element of the arena.
    friction : Tuple[float, float, float]
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001).
    size : Tuple[float, float]
        The size in mm of the arena, by default (250, 250)
    num_sensors : int, optional
        The number of odor sensors, by default 4: 2 antennae + 2 maxillary
        palps.
    odor_source : np.ndarray, optional
        The position of the odor source in (x, y, z) coordinates. The shape
        of the array is (n_sources, 3).
    peak_intensity : np.ndarray, optional
        The peak intensity of the odor source. The shape of the array is
        (n_sources, n_dimensions). Note that the odor intensity can be
        multidimensional.
    diffuse_func : Callable, optional
        The function that, given a distance from the odor source, returns
        the relative intensity of the odor. By default, this is a inverse
        square relationship.
    marker_colors : List[Tuple[float, float, float, float]], optional
        A list of n_sources RGBA values (each as a tuple) indicating the
        colors of the markers indicating the positions of the odor sources.
        The RGBA values should be given in the range [0, 1]. By default,
        the matplotlib color cycle is used.
    marker_size : float, optional
        The size of the odor source markers, by default 0.25.
    ground_alpha : float, optional
        Opacity of the ground, by default 1 (fully opaque).
    scale_bar_pos : Tuple[float, float, float], optional
        If supplied, a 1 mm scale bar will be placed at this location.
    image_path : str, optional
        Path of the image to display on the map
    image_size : Tuple[float, float], optional
        Size of the image in pixels, by default (1000, 1000)
    img_centers : np.ndarray, optional
        Attribute to save the centers of encountered sucrose zones
    

    Parameters
    ----------
    size : Tuple[float, float], optional
        The size of the arena in mm, by default (300, 300).
    friction : Tuple[float, float, float], optional
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001).
    num_sensors : int, optional
        The number of odor sensors, by default 4: 2 antennae + 2 maxillary
        palps.
    odor_source : np.ndarray, optional
        The position of the odor source in (x, y, z) coordinates. The shape
        of the array is (n_sources, 3).
    peak_intensity : np.ndarray, optional
        The peak intensity of the odor source. The shape of the array is
        (n_sources, n_dimensions). Note that the odor intensity can be
        multidimensional.
    diffuse_func : Callable, optional
        The function that, given a distance from the odor source, returns
        the relative intensity of the odor. By default, this is a inverse
        square relationship.
    marker_colors : List[Tuple[float, float, float, float]], optional
        A list of n_sources RGBA values (each as a tuple) indicating the
        colors of the markers indicating the positions of the odor sources.
        The RGBA values should be given in the range [0, 1]. By default,
        the matplotlib color cycle is used.
    marker_size : float, optional
        The size of the odor source markers, by default 0.25.
    ground_alpha : float
        Opacity of the ground, by default 1 (fully opaque).
    scale_bar_pos : Tuple[float, float, float], optional
        If supplied, a 1 mm scale bar will be placed at this location.
    image_path : str, optional
        Path of the image to display on the map
    image_size : Tuple[float, float], optional
        Size of the image in pixels, by default (1000, 1000)
    img_centers : np.ndarray, optional
        Attribute to save the centers of encountered sucrose zones
    """

    def __init__(
        self,
        size: Tuple[float, float] = (250, 250),
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
        #
        num_sensors: int = 4,
        odor_source: np.ndarray = np.array([[10, 0, 0]]),
        peak_intensity: np.ndarray = np.array([[1]]),
        diffuse_func: Callable = lambda x: x**-2,
        marker_colors: Optional[List[Tuple[float, float, float, float]]] = None,
        marker_size: float = 0.25,
        #
        ground_alpha: float = 1.0,
        scale_bar_pos: Optional[Tuple[float, float, float]] = None,
        image_path: str = r"C:\Users\prona\Documents\Cobar\map_created.png",
        image_size: Tuple[float, float] = (1000, 1000),
        img_centers = np.zeros([]),
    ):
        self.root_element = mjcf.RootElement()
        ground_size = [*size, 1]
        imaged = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="none",
            file = image_path,
            width=800, #800
            height=300, #300
            rgb1=(0.3, 0.3, 0.3),
            rgb2=(0.4, 0.4, 0.4),
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=imaged,
            texrepeat=(1, 1),
            reflectance=0.1,
            rgba=(1.0, 1.0, 1.0, ground_alpha),
        )
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground",
            material=grid,
            size=ground_size,
            friction=friction,
        )
        self.friction = friction
        #
        self.num_sensors = num_sensors
        self.odor_source = np.array(odor_source)
        self.peak_odor_intensity = np.array(peak_intensity)
        self.num_odor_sources = self.odor_source.shape[0]
        self.image_size = image_size
        self.image_path = image_path
        self.size = size
        self.img_tab = io.imread(image_path, as_gray=False).astype('uint8')
        self.img_centers = img_centers

        if self.odor_source.shape[0] != self.peak_odor_intensity.shape[0]:
            raise ValueError(
                "Number of odor source locations and peak intensities must match."
            )
        self.diffuse_func = diffuse_func
        #
        if scale_bar_pos:
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                size=(0.05, 0.5),
                pos=scale_bar_pos,
                rgba=(0, 0, 0, 1),
                euler=(0, np.pi / 2, 0),
            )

        # Add birdeye camera
        self.birdeye_cam = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam",
            mode="fixed",
            pos=(0, 0, 20),
            euler=(0, 0, 0),
            fovy=45,
        )
        self.birdeye_cam_zoom = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam_zoom",
            mode="fixed",
            pos=(11, 0, 29),
            euler=(0, 0, 0),
            fovy=45,
        )

        # Odor 

        # Reshape odor source and peak intensity arrays to simplify future claculations
        _odor_source_repeated = self.odor_source[:, np.newaxis, np.newaxis, :]
        _odor_source_repeated = np.repeat(
            _odor_source_repeated, self.odor_dimensions, axis=1
        )
        _odor_source_repeated = np.repeat(
            _odor_source_repeated, self.num_sensors, axis=2
        )
        self._odor_source_repeated = _odor_source_repeated
        _peak_intensity_repeated = self.peak_odor_intensity[:, :, np.newaxis]
        _peak_intensity_repeated = np.repeat(
            _peak_intensity_repeated, self.num_sensors, axis=2
        )
        self._peak_intensity_repeated = _peak_intensity_repeated

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

    def get_olfaction(self, antennae_pos: np.ndarray) -> np.ndarray:
        """
        Notes
        -----
        w = 4: number of sensors (2x antennae + 2x max. palps)
        3: spatial dimensionality
        k: data dimensionality
        n: number of odor sources

        Input - odor source position: [n, 3]
        Input - sensor positions: [w, 3]
        Input - peak intensity: [n, k]
        Input - difusion function: f(dist)

        Reshape sources to S = [n, k*, w*, 3] (* means repeated)
        Reshape sensor position to A = [n*, k*, w, 3] (* means repeated)
        Subtract, getting an Delta = [n, k, w, 3] array of rel difference
        Calculate Euclidean disctance: D = [n, k, w]

        Apply pre-integrated difusion function: S = f(D) -> [n, k, w]
        Reshape peak intensities to P = [n, k, w*]
        Apply scaling: I = P * S -> [n, k, w] element wise

        Output - Sum over the first axis: [k, w]
        """
        antennae_pos_repeated = antennae_pos[np.newaxis, np.newaxis, :, :]
        dist_3d = antennae_pos_repeated - self._odor_source_repeated  # (n, k, w, 3)
        dist_euc = np.linalg.norm(dist_3d, axis=3)  # (n, k, w)
        scaling = self.diffuse_func(dist_euc)  # (n, k, w)
        intensity = self._peak_intensity_repeated * scaling  # (n, k, w)
        return intensity.sum(axis=0)  # (k, w)

    def get_sucrose(self, end_effectors_pos): #end effector pos (6,3)
        """Get the sucrose level of the arena at the position of the end effectors

        Parameters
        ----------
        end_effectors : np.ndarray
            The array containing the position of the two end-effectors of the two front tarsi the fly of shape 
            (2,3) for the x, y and z coordinates of each end-effectors

        Returns
        -------
        sucrose_levels : np.ndarray
            The sucrose level at the position of the end-effectors, of shape (2,1)
        zone_center : np.ndarray
            The center of the sucrose zone at the position of the end-effectors, of shape (2,2)
        """
        sucrose_levels = np.array([0,0])
        zone_center = np.zeros((2,2))
        z_threshold = 0.3 #mm
        ratio_x = 2*self.size[0]/self.image_size[0] 
        ratio_y = 2*self.size[1]/self.image_size[1] 

        for i in range(end_effectors_pos.shape[0]):
            if (np.abs(end_effectors_pos[i][2])<z_threshold):
                x_image = int(np.abs((end_effectors_pos[i][0]+2*self.size[0]/2)/ratio_x)) 
                y_image = int(np.abs((-end_effectors_pos[i][1]+2*self.size[1]/2)/ratio_y)) 
                sucrose_levels[i] = self.img_tab[y_image][x_image][0]
                zone_center[i] = self.img_centers[y_image][x_image]
        return sucrose_levels, zone_center 

    @property
    def odor_dimensions(self) -> int:
        return self.peak_odor_intensity.shape[1]