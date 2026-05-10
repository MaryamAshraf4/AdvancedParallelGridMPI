#include "HeatDiffusion.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

void PrintGrid(vector<vector<double>>& grid, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            cout << fixed << setprecision(1) << grid[i][j] << " ";

        cout << endl;
    }
}

void RunHeatDiffusion(int rank, int size)
{
    const int ROWS = 12;
    const int COLS = 12;
    const int ITERATIONS = 10;

    int baseRows = ROWS / size;
    int extra = ROWS % size;

    int localRows = baseRows + (rank < extra ? 1 : 0);

    vector<vector<double>> local(localRows + 2, vector<double>(COLS, 0));
    vector<vector<double>> next(localRows + 2, vector<double>(COLS, 0));

    for (int i = 1; i <= localRows; i++)
        for (int j = 0; j < COLS; j++)
            local[i][j] = rank + 1;

    if (rank == 0)
    {
        for (int j = 0; j < COLS; j++)
            local[1][j] = 100.0;
    }

    double start = MPI_Wtime();

    for (int step = 0; step < ITERATIONS; step++)
    {
        MPI_Request reqs[4];


        if (rank > 0)
        {
            MPI_Isend(local[1].data(), COLS, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(local[0].data(), COLS, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[1]);
        }

        if (rank < size - 1)
        {
            MPI_Isend(local[localRows].data(), COLS, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[2]);
            MPI_Irecv(local[localRows + 1].data(), COLS, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[3]);
        }

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        for (int i = 1; i <= localRows; i++)
        {
            for (int j = 1; j < COLS - 1; j++)
            {
                next[i][j] =
                    (local[i][j] +
                        local[i - 1][j] +
                        local[i + 1][j] +
                        local[i][j - 1] +
                        local[i][j + 1]) / 5.0;
            }
        }

        local = next;
    }

    double end = MPI_Wtime();

    if (rank == 0)
        cout << "\nHeat Diffusion Finished.\n";

    cout << "Process " << rank << " Time = " << end - start << " sec\n";
}