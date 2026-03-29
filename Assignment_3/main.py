# run full_analysis
import os
import part1_pde_options
import part2_a_pipeline
import part2_b_calibration
import part2_c_evaluation

if __name__ == "__main__":
    print("Analysis starting...")

    # Parte 1: PDE & Exotics
    # part1_pde_options.main()

    # Parte 2: SABR Pipeline
    part2_a_pipeline.main()
    part2_b_calibration.main()
    part2_c_evaluation.main()

    print("Analysis completed. Check the results in the respective output folders.")
