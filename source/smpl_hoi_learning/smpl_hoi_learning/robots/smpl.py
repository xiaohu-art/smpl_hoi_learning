import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from smpl_hoi_learning.assets import SUB10_XML_PATH

SUB10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # asset_path=SUB10_XML_PATH,
        usd_path="/home/ubuntu/Desktop/IsaacSim51/smpl_hoi_learning/source/smpl_hoi_learning/smpl_hoi_learning/assets/smpl_humanoid.usda",
        activate_contact_sensors=True,
        # Articulation properties
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # Disable self-collisions for performance
            solver_position_iteration_count=4,  # Position solver iterations
            solver_velocity_iteration_count=1,  # Velocity solver iterations
        ),
        # Rigid body properties
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
            angular_damping=0.01,               # From original config
            max_linear_velocity=50.0,           # clamp crazy velocities
            max_angular_velocity=50.0,          # 100 is usually unnecessary
        ),
        collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,  # From PhysX config
            rest_offset=0.0,
        )
    ),
    articulation_root_prim_path="/Pelvis",

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (w, x, y, z)
        joint_pos={
            ".*": 0.0,  # All joints start at zero
        },
        joint_vel={
            ".*": 0.0,  # All velocities start at zero
        },
    ),
    
    actuators = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=["L_Hip_.*", "R_Hip_.*", "L_Knee_.*", "R_Knee_.*", "L_Ankle_.*", "R_Ankle_.*", "L_Toe_.*", "R_Toe_.*"],
        stiffness=800.0, damping=80.0, effort_limit_sim=3000.0, velocity_limit_sim=50.0,
    ),
    "torso": ImplicitActuatorCfg(
        joint_names_expr=["Torso_.*", "Spine_.*", "Chest_.*"],
        stiffness=1000.0, damping=100.0, effort_limit_sim=3000.0, velocity_limit_sim=50.0,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=["L_Thorax_.*","R_Thorax_.*","L_Shoulder_.*","R_Shoulder_.*","L_Elbow_.*","R_Elbow_.*","L_Wrist_.*","R_Wrist_.*"],
        stiffness=500.0, damping=50.0, effort_limit_sim=3000.0, velocity_limit_sim=50.0,
    ),
    "fingers": ImplicitActuatorCfg(
        joint_names_expr=[".*Index.*",".*Middle.*",".*Ring.*",".*Pinky.*",".*Thumb.*"],
        stiffness=100.0, damping=10.0, effort_limit_sim=3000.0, velocity_limit_sim=50.0,
    ),
    "head": ImplicitActuatorCfg(
        joint_names_expr=["Neck_.*","Head_.*"],
        stiffness=500.0, damping=50.0, effort_limit_sim=3000.0, velocity_limit_sim=50.0,
    ),
    }
    
)