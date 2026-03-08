import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object import RigidObjectCfg

from smpl_hoi_learning.objects import *

CLOTHESSTAND_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        scale=(1.0, 1.0, 1.0),
        usd_path=CLOTHESSTAND_USD_PATH,
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            density=200.0
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
)