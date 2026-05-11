#include "Shared.h"

#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

void load_csv(const std::string& filename,
    std::vector<double>& data,
    int rows, int cols)
{
    std::ifstream file(filename);

    if (!file) {
        std::cerr << "Cannot open file: " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    data.assign(rows * cols, 0.0);

    std::string line;
    int r = 0;

    while (std::getline(file, line) && r < rows)
    {
        std::stringstream ss(line);
        std::string cell;

        int c = 0;

        while (std::getline(ss, cell, ',') && c < cols)
        {
            data[r * cols + c] = std::stod(cell);
            c++;
        }

        r++;
    }
}

void save_csv(const std::vector<double>& data,
    int rows, int cols,
    const std::string& filename)
{
    std::ofstream f(filename);

    if (!f) {
        std::cerr << "Cannot open " << filename << "\n";
        return;
    }

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            f << std::fixed << std::setprecision(4)
                << data[r * cols + c];

            if (c < cols - 1)
                f << ",";
        }

        f << '\n';
    }

    std::cout << "Saved to " << filename << "\n";
}

void ruler(int width)
{
    std::cout << std::string(width, '-') << '\n';
}