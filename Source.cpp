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
    int comm_choice = 1; 

    if (rank == 0)
    {
        std::cout << "\n========== Parallel MPI Project ==========\n";
        std::cout << "1. Heat Diffusion\n";
        std::cout << "2. Matrix Multiplication\n";
        std::cout << "Choose algorithm: ";
        std::cin >> choice;

        if (choice == 1)
        {
            std::cout << "\nChoose communication strategy:\n";
            std::cout << "1. Blocking\n";
            std::cout << "2. Non-blocking\n";
            std::cout << "3. Deadlock Demo\n";
            std::cout << "Choice: ";
            std::cin >> comm_choice;
        }
    }

    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&comm_choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "choice=" << choice
            << " - comm_choice=" << comm_choice << "\n";
    }

    switch (choice)
    {
    case 1:
        run_heat_diffusion(argc, argv, comm_choice);
        break;

    case 2:
        // run_matrix_multiplication(argc, argv);
        break;

    default:
        if (rank == 0)
            std::cout << "Invalid choice\n";
    }

    MPI_Finalize();
    return 0;
}