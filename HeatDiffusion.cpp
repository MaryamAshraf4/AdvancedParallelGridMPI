#include <mpi.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "Shared.h"
using namespace std;


static const double CONVERGENCE_TOL = 1e-6;

enum CommMode { MODE_NONBLOCKING, MODE_BLOCKING, MODE_DEADLOCK_DEMO };

struct Config {
    int      rows = 512;
    int      cols = 512;
    int      steps = 500;
    CommMode mode = MODE_NONBLOCKING;
    bool     save_output = true;
    bool     demo_deadlock = false;
};

Config parse_args(int argc, char** argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i) {

        string a = argv[i];

        if (a == "--rows" && i + 1 < argc) {
            cfg.rows = stoi(argv[++i]);
            if (cfg.rows <= 0)
            {
                cerr << "Rows must be > 0\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        else if (a == "--cols" && i + 1 < argc) {
            cfg.cols = stoi(argv[++i]);
            if (cfg.cols <= 0)
            {
                cerr << "Cols must be > 0\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        else if (a == "--steps" && i + 1 < argc) {
            cfg.steps = stoi(argv[++i]);
            if (cfg.steps <= 0)
            {
                cerr << "Steps must be > 0\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        else if (a == "--no-save")
            cfg.save_output = false;

        else if (a == "--demo-deadlock")
            cfg.demo_deadlock = true;

        else if (a == "--mode" && i + 1 < argc) {
            string m = argv[++i];
            if (m == "blocking")
                cfg.mode = MODE_BLOCKING;
            else if (m == "nonblocking")
                cfg.mode = MODE_NONBLOCKING;
        }
    }
    return cfg;
}

void demo_deadlock_scenario(vector<double>& local, int local_rows, int cols, MPI_Comm comm, int rank, int size)
{
    const int TAG_DOWN = 1;
    const int TAG_UP = 2;

    double* top_ghost = local.data();
    double* bot_ghost = local.data() + (local_rows + 1) * cols;

    double* first_real = local.data() + cols;
    double* last_real = local.data() + local_rows * cols;

    if (rank == 0) {
        ruler();
        cout << "\n[DEADLOCK DEMO - HEAT DIFFUSION]\n";
        cout << "Every process sends boundary rows first.\n";
        cout << "No process is receiving yet.\n";
        cout << "This creates circular wait deadlock.\n\n";
        ruler();
    }

    MPI_Barrier(comm);


    if (rank > 0)
        MPI_Send(first_real, cols, MPI_DOUBLE,
            rank - 1, TAG_DOWN, comm);

    if (rank < size - 1)
        MPI_Send(last_real, cols, MPI_DOUBLE,
            rank + 1, TAG_UP, comm);

    if (rank > 0)
        MPI_Recv(top_ghost, cols, MPI_DOUBLE,
            rank - 1, TAG_UP, comm,
            MPI_STATUS_IGNORE);

    if (rank < size - 1)
        MPI_Recv(bot_ghost, cols, MPI_DOUBLE,
            rank + 1, TAG_DOWN, comm,
            MPI_STATUS_IGNORE);


    if (rank == 0) {
        cout << "\nBroken version skipped to avoid hanging.\n";
        ruler();
    }

    MPI_Barrier(comm);

    if (rank == 0) {
        cout << "\n[DEADLOCK FIX - NONBLOCKING]\n";
        cout << "Using MPI_Isend + MPI_Irecv for ghost rows.\n";
        cout << "This avoids circular waiting.\n\n";
    }


    MPI_Request reqs[4];
    int nreqs = 0;

    if (rank > 0)
        MPI_Irecv(top_ghost, cols, MPI_DOUBLE,
            rank - 1, TAG_UP,
            comm, &reqs[nreqs++]);

    if (rank < size - 1)
        MPI_Irecv(bot_ghost, cols, MPI_DOUBLE,
            rank + 1, TAG_DOWN,
            comm, &reqs[nreqs++]);

    if (rank > 0)
        MPI_Isend(first_real, cols, MPI_DOUBLE,
            rank - 1, TAG_DOWN,
            comm, &reqs[nreqs++]);

    if (rank < size - 1)
        MPI_Isend(last_real, cols, MPI_DOUBLE,
            rank + 1, TAG_UP,
            comm, &reqs[nreqs++]);

    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

    cout << "[rank " << rank
        << "] ghost-row exchange completed successfully.\n";

    MPI_Barrier(comm);

    if (rank == 0) {
        ruler();
        cout << "Deadlock-free Heat Diffusion communication completed.\n";
        ruler();
    }

}


void exchange_ghosts_nonblocking(vector<double>& local, int local_rows, int cols, int rank, int size, MPI_Comm comm)
{
    MPI_Request reqs[4];
    int nreqs = 0;
    const int TAG_DOWN = 1, TAG_UP = 2;

    double* top_ghost = local.data();
    double* bot_ghost = local.data() + (local_rows + 1) * cols;
    double* first_real = local.data() + cols;
    double* last_real = local.data() + local_rows * cols;

    if (rank > 0)
        MPI_Irecv(top_ghost, cols, MPI_DOUBLE, rank - 1, TAG_UP, comm, &reqs[nreqs++]);
    if (rank < size - 1)
        MPI_Irecv(bot_ghost, cols, MPI_DOUBLE, rank + 1, TAG_DOWN, comm, &reqs[nreqs++]);

    if (rank > 0)
        MPI_Isend(first_real, cols, MPI_DOUBLE, rank - 1, TAG_DOWN, comm, &reqs[nreqs++]);
    if (rank < size - 1)
        MPI_Isend(last_real, cols, MPI_DOUBLE, rank + 1, TAG_UP, comm, &reqs[nreqs++]);

    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
}

void exchange_ghosts_blocking(vector<double>& local, int local_rows, int cols, int rank, int size, MPI_Comm comm)
{
    const int TAG_DOWN = 1, TAG_UP = 2;

    double* top_ghost = local.data();
    double* bot_ghost = local.data() + (local_rows + 1) * cols;
    double* first_real = local.data() + cols;
    double* last_real = local.data() + local_rows * cols;

    if (rank % 2 == 0) {
        if (rank < size - 1) {
            MPI_Send(last_real, cols, MPI_DOUBLE, rank + 1, TAG_UP, comm);
            MPI_Recv(bot_ghost, cols, MPI_DOUBLE, rank + 1, TAG_DOWN, comm, MPI_STATUS_IGNORE);
        }
        if (rank > 0) {
            MPI_Send(first_real, cols, MPI_DOUBLE, rank - 1, TAG_DOWN, comm);
            MPI_Recv(top_ghost, cols, MPI_DOUBLE, rank - 1, TAG_UP, comm, MPI_STATUS_IGNORE);
        }
    }
    else {
        if (rank > 0) {
            MPI_Recv(top_ghost, cols, MPI_DOUBLE, rank - 1, TAG_UP, comm, MPI_STATUS_IGNORE);
            MPI_Send(first_real, cols, MPI_DOUBLE, rank - 1, TAG_DOWN, comm);
        }
        if (rank < size - 1) {
            MPI_Recv(bot_ghost, cols, MPI_DOUBLE, rank + 1, TAG_DOWN, comm, MPI_STATUS_IGNORE);
            MPI_Send(last_real, cols, MPI_DOUBLE, rank + 1, TAG_UP, comm);
        }
    }
}


double compute_stencil(const vector<double>& cur, vector<double>& nxt, int local_rows, int cols)
{
    double max_delta = 0.0;

    for (int i = 1; i <= local_rows; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            double val =
                (
                    cur[(i - 1) * cols + j] +
                    cur[(i + 1) * cols + j] +
                    cur[i * cols + (j - 1)] +
                    cur[i * cols + (j + 1)] +
                    cur[i * cols + j]
                    ) / 5.0;
            nxt[i * cols + j] = val;
            double delta = fabs(val - cur[i * cols + j]);
            if (delta > max_delta)
                max_delta = delta;
        }
    }
    return max_delta;
}

void run_solver(const Config& cfg, MPI_Comm compute_comm)
{
    const int ROWS = cfg.rows;
    const int COLS = cfg.cols;

    int compute_size, compute_rank;
    MPI_Comm_size(compute_comm, &compute_size);
    MPI_Comm_rank(compute_comm, &compute_rank);


    vector<int> row_counts(compute_size), row_offsets(compute_size);
    int base = ROWS / compute_size;
    int rem = ROWS % compute_size;
    for (int p = 0; p < compute_size; ++p) {
        row_counts[p] = base + (p < rem ? 1 : 0);
        row_offsets[p] = (p == 0) ? 0 : row_offsets[p - 1] + row_counts[p - 1];
    }
    int local_rows = row_counts[compute_rank];

    if (compute_rank == 0) {
        ruler();
        cout << "Grid: " << ROWS << " x " << COLS
            << "  |  Processes: " << compute_size
            << "  |  Steps: " << cfg.steps << "\n";
        cout << "Mode: ";

        switch (cfg.mode)
        {
        case MODE_BLOCKING:
            cout << "blocking";
            break;
        case MODE_NONBLOCKING:
            cout << "non-blocking";
            break;
        case MODE_DEADLOCK_DEMO:
            cout << "deadlock-demo";
            break;
        }
        cout << "\n";
        cout << "Row distribution: ";
        for (int p = 0; p < compute_size; ++p)
            cout << "P" << p << "=" << row_counts[p] << " ";
        cout << "\n";
        ruler();
    }

    int buf_rows = local_rows + 2;
    vector<double> cur(buf_rows * COLS, 0.0);
    vector<double> nxt(buf_rows * COLS, 0.0);
    if (cfg.demo_deadlock) {
        demo_deadlock_scenario(
            cur,
            local_rows,
            COLS,
            compute_comm,
            compute_rank,
            compute_size
        );

        MPI_Barrier(compute_comm);
    }

    vector<double> global_grid;
    vector<int> send_counts(compute_size), send_offsets(compute_size);
    for (int p = 0; p < compute_size; ++p) {
        send_counts[p] = row_counts[p] * COLS;
        send_offsets[p] = row_offsets[p] * COLS;
    }

    if (compute_rank == 0) {
        global_grid.resize(ROWS * COLS);
        load_csv("heat_input.csv", global_grid, ROWS, COLS);
    }

    MPI_Scatterv(
        compute_rank == 0 ? global_grid.data() : nullptr,
        send_counts.data(), send_offsets.data(), MPI_DOUBLE,
        cur.data() + COLS,
        local_rows * COLS, MPI_DOUBLE,
        0, compute_comm
    );
    copy(cur.begin(), cur.end(), nxt.begin());

    double t_start = MPI_Wtime();
    bool converged = false;
    int  actual_steps = 0;

    for (int step = 0; step < cfg.steps && !converged; ++step, ++actual_steps) {

        if (cfg.mode == MODE_NONBLOCKING)
            exchange_ghosts_nonblocking(cur, local_rows, COLS,
                compute_rank, compute_size, compute_comm);
        else
            exchange_ghosts_blocking(cur, local_rows, COLS,
                compute_rank, compute_size, compute_comm);

        double local_delta = compute_stencil(cur, nxt, local_rows, COLS);

        double global_delta = 0.0;
        MPI_Allreduce(&local_delta, &global_delta, 1,
            MPI_DOUBLE, MPI_MAX, compute_comm);

        if (global_delta < CONVERGENCE_TOL) converged = true;

        swap(cur, nxt);

        if (compute_rank == 0 && step % 100 == 0)
            cout << "  step " << setw(5) << step
            << "  max_delta = " << scientific << global_delta
            << (converged ? "  [CONVERGED]" : "") << "\n";
    }

    double t_elapsed = MPI_Wtime() - t_start;

    if (compute_rank == 0)
        global_grid.assign(ROWS * COLS, 0.0);

    MPI_Gatherv(
        cur.data() + COLS,
        local_rows * COLS, MPI_DOUBLE,
        compute_rank == 0 ? global_grid.data() : nullptr,
        send_counts.data(), send_offsets.data(), MPI_DOUBLE,
        0, compute_comm
    );

    if (compute_rank == 0) {
        ruler();
        cout << "\nPerformance Summary\n";
        cout << "  Elapsed time  : " << fixed << setprecision(4)
            << t_elapsed << " s\n";
        cout << "  Steps/second  : " << fixed << setprecision(1)
            << actual_steps / t_elapsed << "\n";
        double cells = (double)ROWS * COLS;
        double mcells_per_sec = (cells * actual_steps) / (t_elapsed * 1e6);
        cout << "  MCells/second : " << fixed << setprecision(2)
            << mcells_per_sec << "\n";
        cout << "  Convergence   : " << (converged ? "YES" : "NO (max steps reached)") << "\n";
        ruler();

        if (cfg.save_output)
            save_csv(global_grid, ROWS, COLS, "heat_output.csv");
    }

    double min_t, max_t, sum_t;
    MPI_Reduce(&t_elapsed, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, compute_comm);
    MPI_Reduce(&t_elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, compute_comm);
    MPI_Reduce(&t_elapsed, &sum_t, 1, MPI_DOUBLE, MPI_SUM, 0, compute_comm);

    if (compute_rank == 0) {
        cout << "Per-rank timing (min/avg/max): "
            << fixed << setprecision(4)
            << min_t << " / " << sum_t / compute_size << " / " << max_t
            << " s\n";
        ruler();
    }
}

void print_performance_header(int rank)
{
    if (rank != 0) return;
    ruler(70);
    ruler(70);
}


void run_heat_diffusion(int argc, char** argv, int comm_choice)
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        if (world_rank == 0)
            cerr << "ERROR: This program requires at least 2 MPI processes.\n";
        return;
    }

    Config cfg = parse_args(argc, argv);
    cfg.mode = MODE_NONBLOCKING;
    if (comm_choice == 1)
        cfg.mode = MODE_BLOCKING;
    else if (comm_choice == 2)
        cfg.mode = MODE_NONBLOCKING;
    else if (comm_choice == 3)
    {
        cfg.mode = MODE_DEADLOCK_DEMO;
        cfg.demo_deadlock = true;
    }

    int compute_color = 0;
    MPI_Comm compute_comm;
    MPI_Comm_split(MPI_COMM_WORLD, compute_color, world_rank, &compute_comm);
    int compute_rank, compute_size;
    MPI_Comm_rank(compute_comm, &compute_rank);
    MPI_Comm_size(compute_comm, &compute_size);
    if (compute_rank == 0)
    {
        ruler(); cout
            << " Heat Diffusion MPI Solver\n"; ruler();
    }
    print_performance_header(compute_rank);
    run_solver(cfg, compute_comm);
    MPI_Comm_free(&compute_comm);

}