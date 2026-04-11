#!/usr/bin/env python3
"""
Dummy Publisher for Path Planner Testing
Publishes various test scenarios to /path_planner/request
"""

import rclpy
from rclpy.node import Node
from path_planner_interface.msg import PathPlannerRequest
import json
import time


class DummyPathPlannerPublisher(Node):
    def __init__(self):
        super().__init__('dummy_path_planner_publisher')
        
        self.publisher = self.create_publisher(
            PathPlannerRequest,
            '/path_planner/request',
            10
        )
        
        self.get_logger().info("="*60)
        self.get_logger().info("Dummy Path Planner Publisher Ready")
        self.get_logger().info("="*60)
        
        # Available test scenarios
        self.scenarios = {
            '1': ('Two Individual Holonomic Robots', self.scenario_two_holonomic),
            '2': ('Two Individual Differential Drive Robots', self.scenario_two_diff_drive),
            '3': ('Mixed Individual Robots (1 Holo + 1 DD)', self.scenario_mixed_individual),
            '4': ('Single Homogeneous Formation (3 DD Robots)', self.scenario_homogeneous_formation),
            '5': ('Single Heterogeneous Formation (2 DD + 1 Holo)', self.scenario_heterogeneous_formation),
            '6': ('Two Heterogeneous Formations', self.scenario_two_formations),
            '7': ('Complex Mix: 2 Individuals + 1 Formation', self.scenario_complex_mix),
            '8': ('Large Scale: 3 Formations + 2 Individuals', self.scenario_large_scale),
            '9': ('Minimal Test: 1 Individual Robot', self.scenario_minimal),
        }
        
        self.show_menu()
    
    def show_menu(self):
        """Display available test scenarios"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("AVAILABLE TEST SCENARIOS")
        self.get_logger().info("="*60)
        
        for key, (name, _) in self.scenarios.items():
            self.get_logger().info(f"  {key}. {name}")
        
        self.get_logger().info("="*60)
        self.get_logger().info("Usage: ros2 run path_planner dummy_publisher <scenario_number>")
        self.get_logger().info("Example: ros2 run path_planner dummy_publisher 1")
        self.get_logger().info("="*60 + "\n")
    
    def publish_scenario(self, scenario_num: str):
        """Publish a specific test scenario"""
        
        if scenario_num not in self.scenarios:
            self.get_logger().error(f"Invalid scenario number: {scenario_num}")
            self.get_logger().info("Valid options: " + ", ".join(self.scenarios.keys()))
            return
        
        name, scenario_func = self.scenarios[scenario_num]
        
        self.get_logger().info("="*60)
        self.get_logger().info(f"Publishing Scenario {scenario_num}: {name}")
        self.get_logger().info("="*60)
        
        plan_json = scenario_func()
        
        # Pretty print the JSON
        self.get_logger().info("\nPlan JSON:")
        self.get_logger().info("-"*60)
        self.get_logger().info(json.dumps(plan_json, indent=2))
        self.get_logger().info("-"*60 + "\n")
        
        # Create and publish message
        msg = PathPlannerRequest()
        msg.plan_json = json.dumps(plan_json)
        
        self.get_logger().info("Publishing to /path_planner/request...")
        self.publisher.publish(msg)
        
        self.get_logger().info("✓ Published successfully!\n")
    
    # ========================================================================
    # SCENARIO 1: Two Individual Holonomic Robots
    # ========================================================================
    def scenario_two_holonomic(self):
        return {
  "F1": {
    "centroid_x": 16.0,
    "centroid_y": 12.0,
    "formation_yaw": 0.0,
    "desired_radius": 0.7,
    "robots": [
      {
        "robot": "robot1",
        "x": 15.3,
        "y": 12.0,
        "yaw": 0.0,
        "radius": 0.15,
        "max_velocity": 1.0,
        "max_angular_velocity_z": 1.0,
        "colour": "blue",
        "type": "Differential Drive"
      },
      {
        "robot": "robot2",
        "x": 16.0,
        "y": 11.3,
        "yaw": 0.0,
        "radius": 0.15,
        "max_velocity": 1.0,
        "max_angular_velocity_z": 1.0,
        "colour": "blue",
        "type": "Differential Drive"
      },
      {
        "robot": "robot3",
        "x": 16.7,
        "y": 12.0,
        "yaw": 0.0,
        "radius": 0.15,
        "max_velocity": 1.0,
        "max_angular_velocity_z": 1.0,
        "colour": "blue",
        "type": "Differential Drive"
      }
    ]
  },
  "F2": {
    "centroid_x": 4.0,
    "centroid_y": 2.0,
    "formation_yaw": 0.0,
    "desired_radius": 0.7,
    "robots": [
      {
        "robot": "robot4",
        "x": 3.55,
        "y": 2.0,
        "yaw": 0.0,
        "radius": 0.15,
        "max_velocity": 1.0,
        "max_angular_velocity_z": 1.0,
        "colour": "orange",
        "type": "Holonomic"
      },
      {
        "robot": "robot5",
        "x": 4.0,
        "y": 2.55,
        "yaw": 0.0,
        "radius": 0.15,
        "max_velocity": 1.0,
        "max_angular_velocity_z": 1.0,
        "colour": "orange",
        "type": "Holonomic"
      },
      {
        "robot": "robot6",
        "x": 4.45,
        "y": 2.0,
        "yaw": 0.0,
        "radius": 0.15,
        "max_velocity": 1.0,
        "max_angular_velocity_z": 1.0,
        "colour": "orange",
        "type": "Holonomic"
      }
    ]
  }
}

    
    # ========================================================================
    # SCENARIO 2: Two Individual Differential Drive Robots
    # ========================================================================
    def scenario_two_diff_drive(self):
        return {
  "F1": {
    "centroid_x": 12.0,
    "centroid_y": 10.0,
    "formation_yaw": 0.0,
    "desired_radius": 0.5,
    "robots": [
      {
        "robot": "robot1",
        "x": 4.5,
        "y": 5.5,
        "yaw": 1.5708,
        "radius": 0.15,
        "max_velocity": 0.3,
        "max_angular_velocity_z": 0.5,
        "colour": "blue",
        "type": "Differential Drive"
      },
      {
        "robot": "robot2",
        "x": 4.0,
        "y": 4.5,
        "yaw": -2.3562,
        "radius": 0.15,
        "max_velocity": 0.3,
        "max_angular_velocity_z": 0.5,
        "colour": "blue",
        "type": "Differential Drive"
      },
      {
        "robot": "robot3",
        "x": 5.0,
        "y": 4.5,
        "yaw": -0.7854,
        "radius": 0.15,
        "max_velocity": 0.3,
        "max_angular_velocity_z": 0.5,
        "colour": "blue",
        "type": "Differential Drive"
      }
    ]
  }
}

    
    # ========================================================================
    # SCENARIO 3: Mixed Individual Robots (1 Holonomic + 1 DD)
    # ========================================================================
    def scenario_mixed_individual(self):
        return {
  "F1": {
    "centroid_x": 10.0,
    "centroid_y": 10.0,
    "formation_yaw": 0.0,
    "desired_radius": 1.0,
    # "robots": ["robot1", "robot2", "robot3"]
    "robots": [
        {
            "robot": "robot1",
            "x": 2.5,
            "y": 10.0,
            "yaw": 0.0,
            "radius": 0.35,
            "max_acceleration": 0.2,
            "max_linear_velocity_x": 0.22,
            "max_angular_velocity_z": 1.2,
            "colour": "purple",
            "type": "Differential Drive Robot"
        },
        {
            "robot": "robot2",
            "x": 3.5,
            "y": 10.0,
            "yaw": 0.0,
            "radius": 0.35,
            "max_acceleration": 0.2,
            "max_linear_velocity_x": 0.22,
            "max_angular_velocity_z": 1.2,
            "colour": "purple",
            "type": "Differential Drive Robot"
        },
        {
            "robot": "robot3",
            "x": 3.0,
            "y": 11.0,
            "yaw": 0.0,
            "radius": 0.35,
            "max_acceleration": 0.2,
            "max_linear_velocity_x": 0.22,
            "max_angular_velocity_z": 1.2,
            "colour": "purple",
            "type": "Differential Drive Robot"
        }
    ]
  }
}

    
    # ========================================================================
    # SCENARIO 4: Single Homogeneous Formation (3 DD Robots)
    # ========================================================================
    def scenario_homogeneous_formation(self):
        return {
            "F1": {
                "centroid_x": 15.0,
                "centroid_y": 10.0,
                "formation_yaw": 0.0,
                "desired_radius": 0.4,
                "robots": [
                    {
                        "robot": "burger1",
                        "x": 2.5,
                        "y": 10.0,
                        "yaw": 0.0,
                        "radius": 0.35,
                        "max_acceleration": 0.2,
                        "max_linear_velocity_x": 0.22,
                        "max_angular_velocity_z": 1.2,
                        "colour": "purple",
                        "type": "Differential Drive Robot"
                    },
                    {
                        "robot": "burger2",
                        "x": 3.5,
                        "y": 10.0,
                        "yaw": 0.0,
                        "radius": 0.35,
                        "max_acceleration": 0.2,
                        "max_linear_velocity_x": 0.22,
                        "max_angular_velocity_z": 1.2,
                        "colour": "purple",
                        "type": "Differential Drive Robot"
                    },
                    {
                        "robot": "robot3",
                        "x": 3.0,
                        "y": 11.0,
                        "yaw": 0.0,
                        "radius": 0.35,
                        "max_acceleration": 0.2,
                        "max_linear_velocity_x": 0.22,
                        "max_angular_velocity_z": 1.2,
                        "colour": "purple",
                        "type": "Differential Drive Robot"
                    }
                ]
            }
        }
    
    # ========================================================================
    # SCENARIO 5: Single Heterogeneous Formation (2 DD + 1 Holo)
    # ========================================================================
    def scenario_heterogeneous_formation(self):
        return {
#   "F1": {
#     "centroid_x": 12.0,
#     "centroid_y": 12.0,
#     "formation_yaw": 0.0,
#     "desired_radius": 1.0,
#     "robots": ["robot1", "robot2", "robot3"]
#   },
#   "F2": {
#     "centroid_x": 8.0,
#     "centroid_y": 3.2,
#     "formation_yaw": 1.57,
#     "desired_radius": 0.8,
#     "robots": ["robot4", "robot5", "robot6"]
#   }
# }
    


#   "F1": {
#     "centroid_x": 8.0,
#     "centroid_y": 2.0,
#     "formation_yaw": 0.0,
#     "desired_radius": 1.0,
#     "robots": ["robot4", "robot5", "robot6"]
#   },
#   "F2": {
#     "centroid_x": 12.0,
#     "centroid_y": 10.7,
#     "formation_yaw": 1.57,
#     "desired_radius": 1.0,
#     "robots": ["robot1", "robot2", "robot3"]
#   }
# }


#   "F1": {
#     "centroid_x": 8.0,
#     "centroid_y": 4.0,
#     "formation_yaw": 0.0,
#     "desired_radius": 1.0,
#     "robots": ["robot4", "robot5", "robot6"]
#   },
#   "F2": {
#     "centroid_x": 12.0,
#     "centroid_y": 12.0,
#     "formation_yaw": 0.0,
#     "desired_radius": 1.0,
#     "robots": ["robot1", "robot2", "robot3"]
#   }
# }
    

#   "F1": {
#     "centroid_x": 10.0,
#     "centroid_y": 15.0,
#     "formation_yaw": 0.0,
#     "desired_radius": 1.0,
#     "robots": ["robot4", "robot5", "robot6"]
#   },
#   "F2": {
#     "centroid_x": 18.0,
#     "centroid_y": 5.0,
#     "formation_yaw": 0.0,
#     "desired_radius": 1.0,
#     "robots": ["robot1", "robot2", "robot3"]
#   }
# }


#   "F1": {
#     "centroid_x": 18.0,
#     "centroid_y": 12.5,
#     "formation_yaw": 0.0,
#     "desired_radius": 1.0,
#     "robots": ["robot1", "robot2", "robot3"]
#   },
#   "F2": {
#     "centroid_x": 6.0,
#     "centroid_y": 7.5,
#     "formation_yaw": 0.0,
#     "desired_radius": 1.0,
#     "robots": ["robot4", "robot5", "robot6"]
#   }
# }
    
# "F1": {
#     "centroid_x": 12.0,
#     "centroid_y": 5.0,
#     "formation_yaw": 3.14,
#     "desired_radius": 1.0,
#     "robots": ["robot3", "robot4", "robot5"]
#   },
#   "F2": {
#     "centroid_x": 12.0,
#     "centroid_y": 16.0,
#     "formation_yaw": 0.0,
#     "desired_radius": 1.0,
#     "robots": ["robot1", "robot2", "robot6"]
#   }
# }
    
#   "F1": {
#     "centroid_x": 2.0,
#     "centroid_y": 8.0,
#     "formation_yaw": 3.14,
#     "desired_radius": 1.0,
#     "robots": ["robot3", "robot4", "robot5"]
#   },
#   "F2": {
#     "centroid_x": 12.0,
#     "centroid_y": 16.0,
#     "formation_yaw": 0.0,
#     "desired_radius": 1.0,
#     "robots": ["robot1", "robot2", "robot6"]
#   },


# "R1": [
#     {
#       "robot": "robot1",
#       "x": 16.0,
#       "y": 10.0,
#       "yaw": 1.57
#     }
#   ]
# }


    "F1": {
        "centroid_x": 3.135,
        "centroid_y": 7.583,
        "formation_yaw": 1.57,
        "desired_radius": 1.0,
        "robots": ["burger1", "burger2", "burger3"]
    },
    # "F2": {
    #     "centroid_x": 3.135,
    #     "centroid_y": 1.75,
    #     "formation_yaw": -1.57,
    #     "desired_radius": 1.0,
    #     "robots": ["tb4_1", "firebird", "go2"]
    # },
    # "R1": {
    #     "robot": "waffle", "x": 1.140, "y": 4.375, "yaw": 3.14
    # },
    # "R2": {
    #     "robot": "burger1", "x": 5.13, "y": 4.375, "yaw": 3.14
    # },
}

#     "F1": {
#         "centroid_x": 3.135,
#         "centroid_y": 7.583,
#         "formation_yaw": 1.57,
#         "desired_radius": 1.0,
#         "robots": ["robot1", "robot2", "robot3"]
#     },
#     "F2": {
#         "centroid_x": 3.135,
#         "centroid_y": 1.75,
#         "formation_yaw": -1.57,
#         "desired_radius": 1.0,
#         "robots": ["robot5", "robot6", "robot6"]
#     },
#     "R1": {
#         "robot": "robot3", "x": 1.140, "y": 4.375, "yaw": 3.14
#     },
# }

        #     "F1": {
        #         "centroid_x": 18.0,
        #         "centroid_y": 10.0,
        #         "formation_yaw": 0.0,
        #         "desired_radius": 0.35,
        #         "robots": [
        #             {
        #                 "robot": "burger1",
        #                 "x": 2.0,
        #                 "y": 10.6,
        #                 "yaw": 0.0,
        #                 "radius": 0.3,
        #                 "max_acceleration": 0.2,
        #                 "max_linear_velocity_x": 0.22,
        #                 "max_angular_velocity_z": 1.2,
        #                 "colour": "green",
        #                 "type": "Differential Drive Robot"
        #             },
        #             {
        #                 "robot": "burger2",
        #                 "x": 2.6,
        #                 "y": 10.0,
        #                 "yaw": 0.0,
        #                 "radius": 0.3,
        #                 "max_acceleration": 1.0,
        #                 "max_linear_velocity_x": 0.22,
        #                 "max_linear_velocity_y": 0.22,
        #                 "colour": "green",
        #                 "type": "Holonomic Drive Robot"
        #             },
        #             {
        #                 "robot": "robot3",
        #                 "x": 2.0,
        #                 "y": 9.4,
        #                 "yaw": 0.0,
        #                 "radius": 0.3,
        #                 "max_acceleration": 0.2,
        #                 "max_linear_velocity_x": 0.22,
        #                 "max_angular_velocity_z": 1.2,
        #                 "colour": "green",
        #                 "type": "Differential Drive Robot"
        #             }
        #         ]
        #     }
        # }
    
    # ========================================================================
    # SCENARIO 6: Two Heterogeneous Formations
    # ========================================================================
    def scenario_two_formations(self):
        return {
            "F1": {
                "centroid_x": 10.0,
                "centroid_y": 15.0,
                "formation_yaw": 1.57,
                "desired_radius": 0.35,
                "robots": [
                    {
                        "robot": "burger1",
                        "x": 2.0,
                        "y": 7.0,
                        "yaw": 1.57,
                        "radius": 0.25,
                        "max_acceleration": 0.4,
                        "max_linear_velocity_x": 0.3,
                        "max_angular_velocity_z": 1.5,
                        "colour": "blue",
                        "type": "Differential Drive Robot"
                    },
                    {
                        "robot": "burger2",
                        "x": 3.0,
                        "y": 7.0,
                        "yaw": 1.57,
                        "radius": 0.25,
                        "max_acceleration": 1.0,
                        "max_linear_velocity_x": 0.3,
                        "max_linear_velocity_y": 0.3,
                        "colour": "blue",
                        "type": "Omnidirectional Drive Robot"
                    }
                ]
            },
            "F2": {
                "centroid_x": 10.0,
                "centroid_y": 5.0,
                "formation_yaw": -1.57,
                "desired_radius": 0.35,
                "robots": [
                    {
                        "robot": "tb4",
                        "x": 17.0,
                        "y": 13.0,
                        "yaw": -1.57,
                        "radius": 0.25,
                        "max_acceleration": 0.4,
                        "max_linear_velocity_x": 0.3,
                        "max_angular_velocity_z": 1.5,
                        "colour": "green",
                        "type": "Differential Drive Robot"
                    },
                    {
                        "robot": "waffle",
                        "x": 18.0,
                        "y": 13.0,
                        "yaw": -1.57,
                        "radius": 0.25,
                        "max_acceleration": 1.0,
                        "max_linear_velocity_x": 0.3,
                        "max_linear_velocity_y": 0.3,
                        "colour": "green",
                        "type": "Holonomic Drive Robot"
                    },
                    {
                        "robot": "robot6",
                        "x": 17.5,
                        "y": 12.0,
                        "yaw": -1.57,
                        "radius": 0.25,
                        "max_acceleration": 0.4,
                        "max_linear_velocity_x": 0.3,
                        "max_angular_velocity_z": 1.5,
                        "colour": "green",
                        "type": "Differential Drive Robot"
                    }
                ]
            }
        }
    
    # ========================================================================
    # SCENARIO 7: Complex Mix - 2 Individuals + 1 Formation
    # ========================================================================
    def scenario_complex_mix(self): # Working with 8, 409 (path_planner, seed)
        return {
  "F1": {
    "centroid_x": 12.0,
    "centroid_y": 7.0,
    "formation_yaw": 0.0,
    "desired_radius": 0.7,
    "robots": ["robot1", "robot2", "robot3"]

    # "robots": [
    #   {
    #     "robot": "robot1",
    #     # "x": 11.3,
    #     # "y": 7.0,
    #     # "yaw": 0.0,
    #     # "radius": 0.25,
    #     # "max_acceleration": 0.4,
    #     # "max_linear_velocity_x": 0.2,
    #     # "max_linear_velocity_y": 0.0,
    #     # "max_angular_velocity_z": 0.4,
    #     # "colour": "green",
    #     # "type": "Differentail Drive Robot"
    #   },
    #   {
    #     "robot": "robot2",
    #     # "x": 12.0,
    #     # "y": 7.7,
    #     # "yaw": 0.0,
    #     # "radius": 0.25,
    #     # "max_acceleration": 0.4,
    #     # "max_linear_velocity_x": 0.2,
    #     # "max_linear_velocity_y": 0.0,
    #     # "max_angular_velocity_z": 0.4,
    #     # "colour": "blue",
    #     # "type": "Differentail Drive Robot"
    #   },
    #   {
    #     "robot": "robot3",
    #     # "x": 12.7,
    #     # "y": 7.0,
    #     # "yaw": 0.0,
    #     # "radius": 0.25,
    #     # "max_acceleration": 0.4,
    #     # "max_linear_velocity_x": 0.2,
    #     # "max_linear_velocity_y": 0.0,
    #     # "max_angular_velocity_z": 0.4,
    #     # "colour": "red",
    #     # "type": "Differentail Drive Robot"
    #   }
    # ]
  }
}

    
    # ========================================================================
    # SCENARIO 8: Large Scale - 3 Formations + 2 Individuals
    # ========================================================================
    def scenario_large_scale(self):
        return {
            "F1": {
                "centroid_x": 1.5,
                "centroid_y": 5.0,
                "formation_yaw": 0.0,
                "desired_radius": 0.3,
                "robots": [
                    {
                        "robot": "burger1",
                        "x": 1.0,
                        "y": 5.0,
                        "yaw": 0.0,
                        "radius": 0.25,
                        "max_acceleration": 0.4,
                        "max_linear_velocity_x": 0.3,
                        "max_angular_velocity_z": 1.5,
                        "colour": "blue",
                        "type": "Differential Drive Robot"
                    },
                    {
                        "robot": "burger2",
                        "x": 2.0,
                        "y": 5.0,
                        "yaw": 0.0,
                        "radius": 0.25,
                        "max_acceleration": 1.0,
                        "max_linear_velocity_x": 0.3,
                        "max_linear_velocity_y": 0.3,
                        "colour": "blue",
                        "type": "Holonomic Drive Robot"
                    }
                ]
            },
            "F2": {
                "centroid_x": 2.5,
                "centroid_y": 7.0,
                "formation_yaw": 3.14,
                "desired_radius": 0.3,
                "robots": [
                    {
                        "robot": "waffle",
                        "x": 2.0,
                        "y": 7.0,
                        "yaw": 3.14,
                        "radius": 0.25,
                        "max_acceleration": 0.4,
                        "max_linear_velocity_x": 0.3,
                        "max_angular_velocity_z": 1.5,
                        "colour": "green",
                        "type": "Differential Drive Robot"
                    },
                    {
                        "robot": "tb4",
                        "x": 3.0,
                        "y": 7.0,
                        "yaw": 3.14,
                        "radius": 0.25,
                        "max_acceleration": 1.0,
                        "max_linear_velocity_x": 0.3,
                        "max_linear_velocity_y": 0.3,
                        "colour": "green",
                        "type": "Omnidirectional Drive Robot"
                    }
                ]
            }
            # "R2": [
            #     {
            #         "robot": "go2",
            #         "x": 2.5,
            #         "y": 8.0,
            #         "yaw": 2.0,
            #         "radius": 0.4,
            #         "max_acceleration": 0.2,
            #         "max_linear_velocity_x": 0.22,
            #         "max_angular_velocity_z": 1.2,
            #         "colour": "yellow",
            #         "type": "Differential Drive Robot"
            #     }
            # ],
        }
    

    # ========================================================================
    # SCENARIO 9: Large Scale - 2 Formations 
    # ========================================================================
    def scenario_minimal(self):
        return {
            "F1": {
                "centroid_x": 15.33,
                "centroid_y": 9.0,
                "formation_yaw": 0.0,
                "desired_radius": 0.3,
                "robots": [
                    {
                        "robot": "robot1",
                        "x": 16.0,
                        "y": 8.0,
                        "yaw": 0.0,
                        "radius": 1.0,
                        "max_acceleration": 0.2,
                        "max_linear_velocity_x": 0.15,
                        "max_angular_velocity_z": 1.0,
                        "colour": "blue",
                        "type": "Differential Drive Robot"
                    },
                    {
                        "robot": "robot2",
                        "x": 16.0,
                        "y": 10.0,
                        "yaw": 0.0,
                        "radius": 1.0,
                        "max_acceleration": 0.2,
                        "max_linear_velocity_x": 0.15,
                        "max_angular_velocity_z": 1.0,
                        "colour": "blue",
                        "type": "Differential Drive Robot"
                    },
                    {
                        "robot": "robot3",
                        "x": 15.0,
                        "y": 9.0,
                        "yaw": 0.0,
                        "radius": 1.0,
                        "max_acceleration": 0.2,
                        "max_linear_velocity_x": 0.15,
                        "max_angular_velocity_z": 1.0,
                        "colour": "blue",
                        "type": "Differential Drive Robot"
                    }
                ]
            },
            "F2": {
                "centroid_x": 4.33,
                "centroid_y": 5.0,
                "formation_yaw": 0.0,
                "desired_radius": 0.3,
                "robots": [
                    {
                        "robot": "robot4",
                        "x": 4.0,
                        "y": 6.0,
                        "yaw": 0.0,
                        "radius": 1.0,
                        "max_acceleration": 0.3,
                        "max_linear_velocity_x": 0.15,
                        "max_linear_velocity_y": 0.15,
                        "colour": "green",
                        "type": "Holonomic Drive Robot"
                    },
                    {
                        "robot": "robot5",
                        "x": 4.0,
                        "y": 4.0,
                        "yaw": 0.0,
                        "radius": 1.0,
                        "max_acceleration": 0.3,
                        "max_linear_velocity_x": 0.15,
                        "max_linear_velocity_y": 0.15,
                        "colour": "green",
                        "type": "Holonomic Drive Robot"
                    },
                    {
                        "robot": "robot6",
                        "x": 5.0,
                        "y": 5.0,
                        "yaw": 0.0,
                        "radius": 1.0,
                        "max_acceleration": 0.3,
                        "max_linear_velocity_x": 0.15,
                        "max_linear_velocity_y": 0.15,
                        "colour": "green",
                        "type": "Omnidirectional Drive Robot"
                    }
                ]
            }
        }

def main(args=None):
    rclpy.init(args=args)
    
    node = DummyPathPlannerPublisher()
    
    scenario_num = '5'
    # Publish the scenario
    time.sleep(1.0)  # Wait for publisher to be ready
    node.publish_scenario(scenario_num)
    
    # Keep node alive for a bit
    time.sleep(2.0)
    
    node.get_logger().info("Done! Shutting down...")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()