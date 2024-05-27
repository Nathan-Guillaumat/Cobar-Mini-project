# Imports
import numpy as np
from flygym.mujoco.examples.turning_controller import HybridTurningNMF
from typing import List, Tuple, Dict, Any, Optional, Union
from scipy.spatial.transform import Rotation as R
from numpy.random import choice
from gymnasium.core import ObsType


# NMF Class
class SucroseSensorNMF(HybridTurningNMF):
    def __init__(
        self,
        sucrose_memory: List[Tuple[float, float]] = [0,0],
        egg_probability: float = 0.0,
        egg_rate: float = 0.0,
        egg_laying_speed = 1,
        egg_laying_time = 0.05,
        visited_zones = [],
        **kwargs
    ):
        # Initialize core NMF simulation
        super().__init__(**kwargs)
        self.sucrose_memory = sucrose_memory
        self.egg_probability = egg_probability
        self.egg_rate = egg_rate
        self.egg_laying_speed = egg_laying_speed
        self.egg_laying_time = egg_laying_time
        self.visited_zones = visited_zones

    def get_observation(self) -> Tuple[ObsType, Dict[str, Any]]:  #overriding method
        """Get observation without stepping the physics simulation.

        Returns
        -------
        ObsType
            The observation as defined by the environment.
        """
        # joint sensors
        joint_obs = np.zeros((3, len(self.actuated_joints)))
        joint_sensordata = self.physics.bind(self._joint_sensors).sensordata
        for i, joint in enumerate(self.actuated_joints):
            base_idx = i * 5
            # pos and vel
            joint_obs[:2, i] = joint_sensordata[base_idx : base_idx + 2]
            # torque from pos/vel/motor actuators
            joint_obs[2, i] = joint_sensordata[base_idx + 2 : base_idx + 5].sum()
        joint_obs[2, :] *= 1e-9  # convert to N

        # fly position and orientation
        cart_pos = self.physics.bind(self._body_sensors[0]).sensordata
        cart_vel = self.physics.bind(self._body_sensors[1]).sensordata

        quat = self.physics.bind(self._body_sensors[2]).sensordata
        # ang_pos = transformations.quat_to_euler(quat)
        ang_pos = R.from_quat(quat[[1, 2, 3, 0]]).as_euler(
            "ZYX"
        )  # explicitly use extrinsic ZYX
        # ang_pos[0] *= -1  # flip roll??
        ang_vel = self.physics.bind(self._body_sensors[3]).sensordata
        fly_pos = np.array([cart_pos, cart_vel, ang_pos, ang_vel])

        if self.sim_params.camera_follows_fly_orientation:
            self.fly_rot = ang_pos

        if self.sim_params.draw_gravity:
            self._last_fly_pos = cart_pos

        # contact forces from crf_ext (first three componenents are rotational)
        contact_forces = self.physics.named.data.cfrc_ext[
            self.contact_sensor_placements
        ][:, 3:].copy()
        if self.sim_params.enable_adhesion:
            # Adhesion inputs force in the contact. Lets compute this force
            # and remove it from the contact forces
            contactid_normal = {}
            self._active_adhesion = np.zeros(self.n_legs, dtype=bool)
            for contact in self.physics.data.contact:
                id = np.where(self._adhesion_actuator_geomid == contact.geom1)
                if len(id[0]) > 0 and contact.exclude == 0:
                    contact_sensor_id = self._adhesion_bodies_with_contact_sensors[id][
                        0
                    ]
                    if contact_sensor_id in contactid_normal:
                        contactid_normal[contact_sensor_id].append(contact.frame[:3])
                    else:
                        contactid_normal[contact_sensor_id] = [contact.frame[:3]]
                    self._active_adhesion[id] = True
                id = np.where(self._adhesion_actuator_geomid == contact.geom2)
                if len(id[0]) > 0 and contact.exclude == 0:
                    contact_sensor_id = self._adhesion_bodies_with_contact_sensors[id][
                        0
                    ]
                    if contact_sensor_id in contactid_normal:
                        contactid_normal[contact_sensor_id].append(contact.frame[:3])
                    else:
                        contactid_normal[contact_sensor_id] = [contact.frame[:3]]
                    self._active_adhesion[id] = True

            for contact_sensor_id, normal in contactid_normal.items():
                adh_actuator_id = (
                    self._adhesion_bodies_with_contact_sensors == contact_sensor_id
                )
                if self._last_adhesion[adh_actuator_id] > 0:
                    if len(np.shape(normal)) > 1:
                        normal = np.mean(normal, axis=0)
                    contact_forces[contact_sensor_id, :] -= (
                        self.sim_params.adhesion_force * normal
                    )

        # if draw contacts same last contact forces and positiions
        if self.sim_params.draw_contacts:
            self._last_contact_force = contact_forces
            self._last_contact_pos = (
                self.physics.named.data.xpos[self.contact_sensor_placements].copy().T
            )

        # end effector position
        ee_pos = self.physics.bind(self._end_effector_sensors).sensordata.copy()
        ee_pos = ee_pos.reshape((self.n_legs, 3))

        orientation_vec = self.physics.bind(self._body_sensors[4]).sensordata.copy()


        #self.sucrose_controller()
        egg_probability = self.egg_probability
        egg_decision = 0
        visited_zones = np.zeros((2,2))
        sucrose = np.array([0,0,0,0,0,0])

        obs = {
            "joints": joint_obs.astype(np.float32),
            "fly": fly_pos.astype(np.float32),
            "contact_forces": contact_forces.astype(np.float32),
            "end_effectors": ee_pos.astype(np.float32),
            "fly_orientation": orientation_vec.astype(np.float32),
            "sucrose": sucrose.astype(np.float32),
            "egg_probability": egg_probability,
            "egg_decision": egg_decision,
            "visited_zones": visited_zones,
        }

        # olfaction
        if self.sim_params.enable_olfaction:
            antennae_pos = self.physics.bind(self._antennae_sensors).sensordata
            odor_intensity = self.arena.get_olfaction(antennae_pos.reshape(4, 3))
            obs["odor_intensity"] = odor_intensity.astype(np.float32)

        # vision
        if self.sim_params.enable_vision:
            self._update_vision()
            obs["vision"] = self._curr_visual_input.astype(np.float32)

        # sucrose 
        sucrose_sensor_pos = np.array([obs["end_effectors"][0], obs["end_effectors"][3]]) 
        sucr, zone_center = self.arena.get_sucrose(sucrose_sensor_pos)
        self.sucrose_memory.append(sucr)
        self.visited_zones.append(zone_center)

        obs["sucrose"] = sucr.astype(np.float32)
        obs["egg_probability"] = egg_probability
        decision = self.sucrose_controller(obs["sucrose"]) # Update the probability
        obs["egg_decision"] = decision
        obs["visited_zones"] = zone_center

        return obs

    def sucrose_controller(self, sucrose):
        """Update the internal sucrose variables of the fly

        Parameters
        ----------
        sucrose : np.ndarray
            The array containing the sensed sucrose of the step
            (2,1) for the two end-effectors

        Returns
        -------
        egg_laying_decision : int
            1 if the fly wants to lay and egg, 0 otherwise
        """
        egg_laying_decision = 0
        sucrose_level = np.mean(sucrose)/256
        self.egg_rate = ((1-sucrose_level)+0.2)/60 #egg/s on the zone from 0.2 to 1.2 egg/min depending on if high or low sucrose
        self.sucrose_memory.append(sucrose_level)
        gain = (self.egg_laying_speed*self.timestep) #*time_scale
        added_prob = self.egg_rate*gain


        if ((self.egg_probability+added_prob)>=1):
            self.egg_probability=1
        elif ((self.egg_probability+added_prob)<=0):
            self.egg_probability=0
        else: self.egg_probability += added_prob

        if (self.egg_probability == 1):
            egg_laying_decision = 1
            self.egg_probability = 0

        # Second implementation using probabilitiy to lay an egg -------------------------------------------

        # if ((self.curr_time/self.timestep)%1000 < 1): #Check to lay an egg every 0.1s if 1000 
        #     print("Egg check")
        #     probability_distribution = np.array([1-self.egg_probability, self.egg_probability])
        #     egg_laying_decision = choice([0,1], 1, p=probability_distribution)
        #     if (egg_laying_decision == 1):
        #         self.egg_probability = 0

        #----------------------------------------------------------------------------------------------------

        return egg_laying_decision



    # # Abdomen bending, not used here

    # def _set_joints_stiffness_and_damping(self):
    #     # Do not forget to call the parent method
    #     super()._set_joints_stiffness_and_damping()

    #     # Set the abdomen joints stiffness and damping
    #     for body_name in ["A1A2", "A3", "A4", "A5", "A6"]:
    #         body = self.model.find("body", body_name)
    #         # add pitch degree of freedom to bed the abdomen
    #         body.add(
    #             "joint",
    #             name=f"joint_{body_name}",
    #             type="hinge",
    #             pos="0 0 0",
    #             axis="0 1 0",
    #             stiffness=5.0,
    #             springref=0.0,
    #             damping=5.0,
    #             dclass="nmf",
    #         )

    #         # adding the actuated joints to the list of actuated joints her implies there will be a sensor per leg joint
    #         # if added later that would not be the case (just be aware of the differences)
    #         self.actuated_joints.append(f"joint_{body_name}")

    # def _add_adhesion_actuators(self, gain):
    #     for body_name in ["A1A2", "A3", "A4", "A5", "A6"]:
    #         joint = self.model.find("joint", f"joint_{body_name}")
    #         actuator = self.model.actuator.add(
    #             "position",
    #             name=f"actuator_position_joint_{body_name}",
    #             joint=joint,
    #             forcelimited="true",
    #             ctrlrange="-1000000 1000000",
    #             forcerange="-10 10",
    #             kp=self.sim_params.actuator_kp,
    #             dclass="nmf",
    #         )

    #         # this is needed if you do not want to override add joint sensors
    #         vel_actuator = self.model.actuator.add(
    #             "velocity",
    #             name=f"actuator_velocity_joint_{body_name}",
    #             joint=joint,
    #             dclass="nmf",
    #         )
    #         torque_actuator = self.model.actuator.add(
    #             "motor",
    #             name=f"actuator_torque_joint_{body_name}",
    #             joint=joint,
    #             dclass="nmf",
    #         )
    #         # self.actuated_joints.append(joint)
    #         self._actuators.append(actuator)

    #     return super()._add_adhesion_actuators(gain)

