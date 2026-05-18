from planners.dataloader import load_scenario
from planners.stochastic_planner import plan_stochastic

def main():
    scenario_path = "scenarios/USA_Lanker-1_1_T-1.xml"

    print("Loading scenario...")
    scenario = load_scenario(scenario_path)

    print("Running stochastic planner...")
    result = plan_stochastic(scenario, epsilon=0.05)

    for key, value in result.items():
        if key in ["controls", "trajectory"]:
            continue
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
