from commonroad.common.file_reader import CommonRoadFileReader
import matplotlib.pyplot as plt

scenario_path = "../scenarios/USA_Lanker-1_1_T-1.xml"

# load scenairos, check with xml files in dataset (can also filter which ones aremost relevant)
scenario, planning_problem_set = CommonRoadFileReader(
    scenario_path
).open()

print("Scenario loaded successfully!")
print(f"Number of dynamic obstacles: {len(scenario.dynamic_obstacles)}")
print(f"Number of lanelets: {len(scenario.lanelet_network.lanelets)}")

for obstacle in scenario.dynamic_obstacles[:5]:
    print(
        f"Obstacle ID: {obstacle.obstacle_id}, "
        f"Type: {obstacle.obstacle_type}"
    )

fig, ax = plt.subplots(figsize=(10, 6))

scenario.draw(ax)

plt.title("CommonRoad Scenario")
plt.axis("equal")
plt.show()
