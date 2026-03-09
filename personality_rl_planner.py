import heapq

class InformationRemnant:
    def __init__(self, name, preconditions, effects, complexity=1):
        self.name = name
        self.pre = set(preconditions)
        self.eff = set(effects)
        self.mass = complexity

class SupersphereEngine:
    def __init__(self, remnants):
        self.remnants = remnants
        self.c = 1 

    def get_entropic_pressure(self, goals, initial_state):
        return len(goals - initial_state)

    def evolve(self, target_information, vacuum_state):
        target_info = set(target_information)
        initial_symmetry = set(vacuum_state)

        chaos_frontier = []
        initial_pressure = self.get_entropic_pressure(target_info, initial_symmetry)
        
        heapq.heappush(chaos_frontier, (initial_pressure, 0, [], target_info))

        seen_symmetries = set()

        while chaos_frontier:
            pressure, time_elapsed, current_structure, current_goals = heapq.heappop(chaos_frontier)

            if not current_goals or current_goals.issubset(initial_symmetry):
                return current_structure[::-1]

            state_signature = tuple(sorted(list(current_goals)))
            if state_signature in seen_symmetries:
                continue
            seen_symmetries.add(state_signature)

            for action in self.remnants:
                if action.eff & current_goals:
                    next_goals = (current_goals - action.eff) | action.pre
                    next_goals = next_goals - initial_symmetry

                    new_time = time_elapsed + self.c
                    new_pressure = self.get_entropic_pressure(next_goals, initial_symmetry) + new_time / action.mass
                    
                    new_structure = current_structure + [action.name]
                    
                    heapq.heappush(chaos_frontier, (new_pressure, new_time, new_structure, next_goals))

        return None

personality_rl_build = [
    InformationRemnant("Get Sudo", [], ["have_sudo"], complexity=0.1),
    InformationRemnant("Enable Internet", [], ["have_internet"], complexity=0.1),
    InformationRemnant("Free Disk Space", [], ["disk_space"], complexity=0.5),
    
    InformationRemnant("Check OS", ["have_sudo"], ["os_supported"], complexity=1),
    InformationRemnant("Install ROCm", ["have_sudo", "os_supported"], ["rocm_installed"], complexity=8),
    InformationRemnant("Verify GPU", ["rocm_installed"], ["gpu_verified"], complexity=0.5),
    
    InformationRemnant("Install Rust", ["have_sudo", "have_internet"], ["rust_ready"], complexity=2),
    
    InformationRemnant("Download LibTorch", ["have_internet", "rocm_installed"], ["libtorch_downloaded"], complexity=3),
    InformationRemnant("Extract LibTorch", ["libtorch_downloaded", "disk_space"], ["libtorch_extracted"], complexity=1),
    InformationRemnant("Setup LibTorch Env", ["libtorch_extracted"], ["libtorch_ready"], complexity=0.5),
    
    InformationRemnant("Create Project Dir", ["rust_ready"], ["project_created"], complexity=0.2),
    InformationRemnant("Create Cargo.toml", ["project_created"], ["cargo_configured"], complexity=0.5),
    InformationRemnant("Copy Source Files", ["project_created"], ["source_ready"], complexity=0.3),
    
    InformationRemnant("Cargo Build", ["cargo_configured", "source_ready", "libtorch_ready", "disk_space"], ["binary_ready"], complexity=15),
    
    InformationRemnant("Create Checkpoints Dir", ["project_created"], ["checkpoints_ready"], complexity=0.1),
    InformationRemnant("Run Training", ["binary_ready", "gpu_verified", "checkpoints_ready", "disk_space"], ["training_running"], complexity=1),
]

initial = []
target = ["training_running"]

engine = SupersphereEngine(personality_rl_build)

stable_plan = engine.evolve(target, initial)

if stable_plan:
    print("=== PERSONALITY RL INSTALLATION PLAN ===\n")
    total_complexity = 0
    
    for i, step in enumerate(stable_plan, 1):
        action = next((r for r in personality_rl_build if r.name == step), None)
        if action:
            total_complexity += action.mass
            print(f"Step {i:2d} [{action.mass:5.1f}]: {step}")
            if action.pre:
                print(f"         Requires: {sorted(action.pre)}")
            print(f"         Produces: {sorted(action.eff)}")
    
    print(f"\nTotal Complexity: {total_complexity:.1f}")
    print(f"Number of Steps: {len(stable_plan)}")
    
    print("\n=== CRITICAL PATH ANALYSIS ===")
    heavy_ops = [(r.name, r.mass) for r in personality_rl_build if r.name in stable_plan and r.mass >= 5]
    heavy_ops.sort(key=lambda x: x[1], reverse=True)
    for name, mass in heavy_ops:
        print(f"  {name}: {mass}")
    
    print("\n=== PARALLELIZATION OPPORTUNITIES ===")
    print("Can run in parallel after 'Enable Internet' + 'Get Sudo':")
    print("  - Download LibTorch (requires internet + ROCm)")
    print("  - Install Rust (requires internet + sudo)")
    print("\nCan run in parallel after 'Create Project Dir':")
    print("  - Create Cargo.toml")
    print("  - Copy Source Files")
    print("  - Create Checkpoints Dir")
    
else:
    print("=== FAILURE: No valid build path found ===")
