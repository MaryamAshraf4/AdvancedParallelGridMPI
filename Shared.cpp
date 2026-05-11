#include "Shared.h"

#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
using namespace std;

void load_csv(const string& filename, vector<double>& data, int rows, int cols)
{
    ifstream file(filename);

    if (!file) {
        cerr << "Cannot open file: " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    data.assign(rows * cols, 0.0);

    string line;
    int r = 0;

    while (getline(file, line) && r < rows)
    {
        stringstream ss(line);
        string cell;

        int c = 0;

        while (getline(ss, cell, ',') && c < cols)
        {
            data[r * cols + c] = stod(cell);
            c++;
        }

        r++;
    }
}

void save_csv(const vector<double>& data, int rows, int cols, const string& filename)
{
    ofstream f(filename);

    if (!f) {
        cerr << "Cannot open " << filename << "\n";
        return;
    }

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            f << fixed << setprecision(4) << data[r * cols + c];

            if (c < cols - 1)
                f << ",";
        }

        f << '\n';
    }

    cout << "Saved to " << filename << "\n";
}

void ruler(int width)
{
    cout << string(width, '-') << '\n';
}