python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config1.yaml --naive_task -pt reachavoidcf1_naive_task
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config1.yaml --naive_task -pt reachavoidcf1_naive_task_squeeze -rb 1.0
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config1.yaml -pt reachavoidcf1_lqr_task
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config1.yaml -pt reachavoidcf1_lqr_task_squeeze -rb 1.0
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config2.yaml -pt reachavoidcf2_lqr_task
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config2.yaml --naive_task -pt reachavoidcf2_naive_task
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config3.yaml -pt reachavoidcf3_lqr_task
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config3.yaml --naive_task -pt reachavoidcf3_naive_task
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config4.yaml -pt reachavoidcf4_lqr_task