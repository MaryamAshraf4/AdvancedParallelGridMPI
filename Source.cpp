#include <mpi.h>
#include <iostream>

#include "HeatDiffusion.h"
//#include "matrix_multiplication.h"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int choice = 0;

    if (rank == 0)
    {
        std::cout << "\n========== Parallel MPI Project ==========\n";
        std::cout << "1. Heat Diffusion\n";
        std::cout << "2. Matrix Multiplication\n";
        std::cout << "Choose algorithm: ";

        std::cin >> choice;
    }

    // broadcast choice to all processes
    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

    switch (choice)
    {
    case 1:
        run_heat_diffusion(argc, argv);
        break;

    case 2:
        //run_matrix_multiplication(argc, argv);
        break;

    default:
        if (rank == 0)
            std::cout << "Invalid choice\n";
    }

    MPI_Finalize();
    return 0;
}