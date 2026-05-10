#include <mpi.h>
#include <iostream>
#include <string>

#include "HeatDiffusion.h"
//#include "MatrixMultiplication.h"

using namespace std;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2)
    {
        if (rank == 0)
            cout << "Run with at least 2 processes.\n";

        MPI_Finalize();
        return 0;
    }

    string choice;

    if (rank == 0)
    {
        cout << "Choose Algorithm:\n";
        cout << "1. Heat Diffusion\n";
        cout << "2. Matrix Multiplication\n";
        cin >> choice;
    }

    char algo[20];

    if (rank == 0)
        strcpy_s(algo, choice.c_str());

    MPI_Bcast(algo, 20, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (string(algo) == "1")
        RunHeatDiffusion(rank, size);

    //else if (string(algo) == "2")
    //    RunMatrixMultiplication(rank, size);

    else
    {
        if (rank == 0)
            cout << "Invalid Choice.\n";
    }

    MPI_Finalize();
    return 0;
}