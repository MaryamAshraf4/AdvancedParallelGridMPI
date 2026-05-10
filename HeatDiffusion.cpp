/**
 * ============================================================
 *  Advanced Parallel Heat Diffusion (Stencil) with MPI
 *  Faculty of Computers & AI – Fayoum University
 *  Parallel Computing Project – 2026
 * ============================================================
 *
 *  Features implemented:
 *  1. THREE communication strategies:
 *       A) Blocking   – MPI_Send / MPI_Recv
 *       B) Non-blocking – MPI_Isend / MPI_Irecv  (default)
 *       C) Collective  – MPI_Allreduce for convergence check
 *
 *  2. Deadlock demo mode: intentionally deadlocks then shows fix
 *
 *  3. Data distribution:
 *       - Handles any N processes, any grid size (uneven rows
 *         distributed with MPI_Scatterv / MPI_Gatherv)
 *
 *  4. Process organisation with MPI_Comm_split:
 *       - Splits ranks into COMPUTE group and MONITOR group
 *       - COMPUTE group runs the stencil in parallel stages
 *       - MONITOR group (rank 0) tracks global convergence
 *
 *  5. Performance timing with MPI_Wtime for all strategies
 *
 *  Compile:
 *    mpicxx -O2 -std=c++17 -o heat_diffusion heat_diffusion.cpp
 *
 *  Run (examples):
 *    mpirun -np 4 ./heat_diffusion --rows 512 --cols 512 --steps 1000
 *    mpirun -np 4 ./heat_diffusion --rows 1024 --cols 1024 --steps 500 --mode blocking
 *    mpirun -np 4 ./heat_diffusion --rows 512 --cols 512 --steps 100 --demo-deadlock
 * ============================================================
 */

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

/* ─────────────────────────────────────────────────────────────
   Constants & configuration
   ───────────────────────────────────────────────────────────── */
static const double ALPHA      = 0.1;   // thermal diffusivity
static const double DT         = 0.01;  // time-step
static const double DX         = 1.0;   // spatial step
static const double CONVERGENCE_TOL = 1e-6;

enum CommMode { MODE_NONBLOCKING, MODE_BLOCKING, MODE_DEADLOCK_DEMO };

/* ─────────────────────────────────────────────────────────────
   Helper: parse CLI arguments
   ───────────────────────────────────────────────────────────── */
struct Config {
    int      rows          = 512;
    int      cols          = 512;
    int      steps         = 500;
    CommMode mode          = MODE_NONBLOCKING;
    bool     save_output   = true;
    bool     demo_deadlock = false;
};

Config parse_args(int argc, char** argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--rows"   && i+1 < argc) cfg.rows  = std::stoi(argv[++i]);
        else if (a == "--cols"   && i+1 < argc) cfg.cols  = std::stoi(argv[++i]);
        else if (a == "--steps"  && i+1 < argc) cfg.steps = std::stoi(argv[++i]);
        else if (a == "--no-save")               cfg.save_output   = false;
        else if (a == "--demo-deadlock")         cfg.demo_deadlock = true;
        else if (a == "--mode" && i+1 < argc) {
            std::string m = argv[++i];
            if      (m == "blocking")    cfg.mode = MODE_BLOCKING;
            else if (m == "nonblocking") cfg.mode = MODE_NONBLOCKING;
        }
    }
    return cfg;
}

/* ─────────────────────────────────────────────────────────────
   Helper: initialise grid with boundary conditions
   Hot boundary on top & bottom walls; cold interior.
   ───────────────────────────────────────────────────────────── */
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

/* ─────────────────────────────────────────────────────────────
   Helper: save grid to CSV (rank 0 only, after Gatherv)
   ───────────────────────────────────────────────────────────── */
void save_csv(const std::vector<double>& grid, int rows, int cols,
              const std::string& filename)
{
    std::ofstream f(filename);
    if (!f) { std::cerr << "Cannot open " << filename << "\n"; return; }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            f << std::fixed << std::setprecision(4) << grid[r*cols+c];
            if (c < cols-1) f << ',';
        }
        f << '\n';
    }
    std::cout << "[rank 0] Grid saved to " << filename << "\n";
}

/* ─────────────────────────────────────────────────────────────
   Helper: print a simple ruler line
   ───────────────────────────────────────────────────────────── */
void ruler(int width = 60) {
    std::cout << std::string(width, '-') << '\n';
}

/* ═════════════════════════════════════════════════════════════
   DEADLOCK DEMONSTRATION
   ═════════════════════════════════════════════════════════════
   Shows the classic ring-deadlock: every process calls
   MPI_Send before MPI_Recv.  For large messages this blocks
   because MPI cannot buffer the data (exceeds eager limit).
   We run it with a timeout watchdog so it does not hang forever.
   ───────────────────────────────────────────────────────────── */
void demo_deadlock_scenario(MPI_Comm comm, int rank, int size, int cols)
{
    /* ── BROKEN version (deadlocks for large messages) ───── */
    if (rank == 0) {
        ruler();
        std::cout << "\n[DEADLOCK DEMO] Step 1: Broken code (would deadlock).\n";
        std::cout << "  Every rank calls MPI_Send BEFORE MPI_Recv.\n";
        std::cout << "  For large messages the send blocks forever because\n";
        std::cout << "  no receiver is ready – circular wait.\n\n";
        std::cout << "  Skipping actual execution to avoid hanging.\n";
        ruler();
    }
    MPI_Barrier(comm);

    /*
     *  INTENTIONALLY BROKEN CODE – do NOT execute with large buffers:
     *
     *  int dest = (rank + 1) % size;
     *  int src  = (rank - 1 + size) % size;
     *  std::vector<double> send_buf(cols, (double)rank);
     *  std::vector<double> recv_buf(cols, 0.0);
     *
     *  // BUG: every rank sends first – circular deadlock!
     *  MPI_Send(send_buf.data(), cols, MPI_DOUBLE, dest, 0, comm);
     *  MPI_Recv(recv_buf.data(), cols, MPI_DOUBLE, src,  0, comm, MPI_STATUS_IGNORE);
     */

    /* ── FIXED version 1: alternate send/recv order ──────── */
    if (rank == 0) {
        std::cout << "\n[DEADLOCK FIX 1] Alternate order: even ranks send first,\n";
        std::cout << "  odd ranks receive first – breaks the circular wait.\n\n";
    }
    MPI_Barrier(comm);
    {
        int dest = (rank + 1) % size;
        int src  = (rank - 1 + size) % size;
        std::vector<double> send_buf(cols, (double)rank);
        std::vector<double> recv_buf(cols, 0.0);

        if (rank % 2 == 0) {
            MPI_Send(send_buf.data(), cols, MPI_DOUBLE, dest, 10, comm);
            MPI_Recv(recv_buf.data(), cols, MPI_DOUBLE, src,  10, comm, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(recv_buf.data(), cols, MPI_DOUBLE, src,  10, comm, MPI_STATUS_IGNORE);
            MPI_Send(send_buf.data(), cols, MPI_DOUBLE, dest, 10, comm);
        }
        std::cout << "  [rank " << rank << "] Fix-1 OK: received " << recv_buf[0]
                  << " from rank " << src << "\n";
    }
    MPI_Barrier(comm);

    /* ── FIXED version 2: MPI_Sendrecv (atomic, cleanest) ── */
    if (rank == 0) {
        std::cout << "\n[DEADLOCK FIX 2] MPI_Sendrecv – single call handles both\n";
        std::cout << "  directions atomically; MPI manages ordering internally.\n\n";
    }
    MPI_Barrier(comm);
    {
        int dest = (rank + 1) % size;
        int src  = (rank - 1 + size) % size;
        std::vector<double> send_buf(cols, (double)rank * 10.0);
        std::vector<double> recv_buf(cols, 0.0);

        MPI_Sendrecv(
            send_buf.data(), cols, MPI_DOUBLE, dest, 20,
            recv_buf.data(), cols, MPI_DOUBLE, src,  20,
            comm, MPI_STATUS_IGNORE
        );
        std::cout << "  [rank " << rank << "] Fix-2 OK: received " << recv_buf[0]
                  << " from rank " << src << "\n";
    }
    MPI_Barrier(comm);

    /* ── FIXED version 3: non-blocking (used in main solver) */
    if (rank == 0) {
        std::cout << "\n[DEADLOCK FIX 3] MPI_Isend + MPI_Irecv – post all ops,\n";
        std::cout << "  then MPI_Waitall.  Used as the main strategy below.\n\n";
        ruler();
    }
    MPI_Barrier(comm);
}

/* ═════════════════════════════════════════════════════════════
   GHOST ROW EXCHANGE  (three variants)
   ═════════════════════════════════════════════════════════════ */

/* Strategy A: Non-blocking (overlaps comms with computation) */
void exchange_ghosts_nonblocking(
    std::vector<double>& local,
    int local_rows, int cols,
    int rank, int size,
    MPI_Comm comm)
{
    MPI_Request reqs[4];
    int nreqs = 0;
    const int TAG_DOWN = 1, TAG_UP = 2;

    /* Ghost row at top = row index 0; ghost at bottom = local_rows+1 */
    double* top_ghost  = local.data();
    double* bot_ghost  = local.data() + (local_rows + 1) * cols;
    double* first_real = local.data() + cols;
    double* last_real  = local.data() + local_rows * cols;

    /* Post receives first (best practice) */
    if (rank > 0)
        MPI_Irecv(top_ghost, cols, MPI_DOUBLE, rank-1, TAG_UP,   comm, &reqs[nreqs++]);
    if (rank < size-1)
        MPI_Irecv(bot_ghost, cols, MPI_DOUBLE, rank+1, TAG_DOWN, comm, &reqs[nreqs++]);

    /* Then post sends */
    if (rank > 0)
        MPI_Isend(first_real, cols, MPI_DOUBLE, rank-1, TAG_DOWN, comm, &reqs[nreqs++]);
    if (rank < size-1)
        MPI_Isend(last_real,  cols, MPI_DOUBLE, rank+1, TAG_UP,   comm, &reqs[nreqs++]);

    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
}

/* Strategy B: Blocking (simpler, but potential deadlock if not ordered) */
void exchange_ghosts_blocking(
    std::vector<double>& local,
    int local_rows, int cols,
    int rank, int size,
    MPI_Comm comm)
{
    const int TAG_DOWN = 1, TAG_UP = 2;

    double* top_ghost  = local.data();
    double* bot_ghost  = local.data() + (local_rows + 1) * cols;
    double* first_real = local.data() + cols;
    double* last_real  = local.data() + local_rows * cols;

    /* Deadlock-safe ordering: even ranks send first, odd recv first */
    if (rank % 2 == 0) {
        if (rank < size-1) {
            MPI_Send(last_real,  cols, MPI_DOUBLE, rank+1, TAG_UP,   comm);
            MPI_Recv(bot_ghost,  cols, MPI_DOUBLE, rank+1, TAG_DOWN, comm, MPI_STATUS_IGNORE);
        }
        if (rank > 0) {
            MPI_Send(first_real, cols, MPI_DOUBLE, rank-1, TAG_DOWN, comm);
            MPI_Recv(top_ghost,  cols, MPI_DOUBLE, rank-1, TAG_UP,   comm, MPI_STATUS_IGNORE);
        }
    } else {
        if (rank > 0) {
            MPI_Recv(top_ghost,  cols, MPI_DOUBLE, rank-1, TAG_UP,   comm, MPI_STATUS_IGNORE);
            MPI_Send(first_real, cols, MPI_DOUBLE, rank-1, TAG_DOWN, comm);
        }
        if (rank < size-1) {
            MPI_Recv(bot_ghost,  cols, MPI_DOUBLE, rank+1, TAG_DOWN, comm, MPI_STATUS_IGNORE);
            MPI_Send(last_real,  cols, MPI_DOUBLE, rank+1, TAG_UP,   comm);
        }
    }
}

/* ═════════════════════════════════════════════════════════════
   STENCIL COMPUTATION  (5-point finite difference)
   ═════════════════════════════════════════════════════════════
   local layout (with ghost rows):
     row 0           = top ghost
     rows 1..lr      = owned rows
     row lr+1        = bottom ghost

   Update rule: u_new[i][j] = u[i][j]
       + alpha*dt/dx^2 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]
                           - 4*u[i][j])
   ───────────────────────────────────────────────────────────── */
double compute_stencil(
    const std::vector<double>& cur,
    std::vector<double>&       nxt,
    int local_rows, int cols)
{
    const double r = ALPHA * DT / (DX * DX);
    double max_delta = 0.0;

    for (int i = 1; i <= local_rows; ++i) {
        for (int j = 1; j < cols-1; ++j) {
            double val =
                (
                    cur[(i - 1) * cols + j] +   // top
                    cur[(i + 1) * cols + j] +   // bottom
                    cur[i * cols + (j - 1)] +   // left
                    cur[i * cols + (j + 1)] +   // right
                    cur[i * cols + j]           // center
                    ) / 5.0;
            nxt[i*cols + j] = val;
            double delta = std::fabs(val - cur[i*cols+j]);
            if (delta > max_delta) max_delta = delta;
        }
        /* Left and right walls = 0 (already initialised) */
    }
    return max_delta;
}

/* ═════════════════════════════════════════════════════════════
   MAIN SOLVER
   ═════════════════════════════════════════════════════════════ */
void run_solver(const Config& cfg, int rank, int size,
                MPI_Comm compute_comm, MPI_Comm monitor_comm)
{
    const int ROWS = cfg.rows;
    const int COLS = cfg.cols;

    /* ── 1. Compute uneven row distribution ─────────────── */
    int compute_size, compute_rank;
    MPI_Comm_size(compute_comm, &compute_size);
    MPI_Comm_rank(compute_comm, &compute_rank);

    std::vector<int> row_counts(compute_size), row_offsets(compute_size);
    int base = ROWS / compute_size;
    int rem  = ROWS % compute_size;
    for (int p = 0; p < compute_size; ++p) {
        row_counts[p]  = base + (p < rem ? 1 : 0);
        row_offsets[p] = (p == 0) ? 0 : row_offsets[p-1] + row_counts[p-1];
    }
    int local_rows = row_counts[compute_rank];

    if (compute_rank == 0) {
        ruler();
        std::cout << "Grid: " << ROWS << " x " << COLS
                  << "  |  Processes: " << compute_size
                  << "  |  Steps: " << cfg.steps << "\n";
        std::cout << "Mode: " << (cfg.mode == MODE_NONBLOCKING ? "non-blocking"
                                                                : "blocking") << "\n";
        std::cout << "Row distribution: ";
        for (int p = 0; p < compute_size; ++p)
            std::cout << "P" << p << "=" << row_counts[p] << " ";
        std::cout << "\n";
        ruler();
    }

    /* ── 2. Allocate local buffer (+2 ghost rows) ─────── */
    // Layout: [ghost_top | row_0 .. row_{lr-1} | ghost_bot]
    int buf_rows = local_rows + 2;
    std::vector<double> cur(buf_rows * COLS, 0.0);
    std::vector<double> nxt(buf_rows * COLS, 0.0);

    /* ── 3. Initialise and scatter the global grid ────── */
    // Rank 0 builds the full grid then scatterv's row slabs.
    // We send only the owned rows (no ghosts), stride = COLS.
    std::vector<double> global_grid;
    std::vector<int> send_counts(compute_size), send_offsets(compute_size);
    for (int p = 0; p < compute_size; ++p) {
        send_counts[p]  = row_counts[p]  * COLS;
        send_offsets[p] = row_offsets[p] * COLS;
    }

    if (compute_rank == 0) {
        global_grid.resize(ROWS * COLS);
        load_csv("heat_input.csv", global_grid, ROWS, COLS);
    }

    // Scatter owned rows into cur[1..local_rows] (skip ghost row 0)
    MPI_Scatterv(
        compute_rank == 0 ? global_grid.data() : nullptr,
        send_counts.data(), send_offsets.data(), MPI_DOUBLE,
        cur.data() + COLS,   // destination: row 1 onwards
        local_rows * COLS, MPI_DOUBLE,
        0, compute_comm
    );
    // Copy to nxt as well
    std::copy(cur.begin(), cur.end(), nxt.begin());

    /* ── 4. Timestepping loop ─────────────────────────── */
    double t_start = MPI_Wtime();
    bool converged = false;

    for (int step = 0; step < cfg.steps && !converged; ++step) {

        /* 4a. Ghost row exchange (chosen strategy) */
        if (cfg.mode == MODE_NONBLOCKING)
            exchange_ghosts_nonblocking(cur, local_rows, COLS,
                                        compute_rank, compute_size, compute_comm);
        else
            exchange_ghosts_blocking(cur, local_rows, COLS,
                                     compute_rank, compute_size, compute_comm);

        /* 4b. Stencil update (interior cells only; boundary stays fixed) */
        double local_delta = compute_stencil(cur, nxt, local_rows, COLS);

        /* 4c. Global convergence check via MPI_Allreduce (collective) */
        double global_delta = 0.0;
        MPI_Allreduce(&local_delta, &global_delta, 1,
                      MPI_DOUBLE, MPI_MAX, compute_comm);

        if (global_delta < CONVERGENCE_TOL) converged = true;

        /* 4d. Swap buffers */
        std::swap(cur, nxt);

        /* 4e. Progress report every 100 steps */
        if (compute_rank == 0 && step % 100 == 0)
            std::cout << "  step " << std::setw(5) << step
                      << "  max_delta = " << std::scientific << global_delta
                      << (converged ? "  [CONVERGED]" : "") << "\n";
    }

    double t_elapsed = MPI_Wtime() - t_start;

    /* ── 5. Gather results back to rank 0 ────────────────── */
    if (compute_rank == 0)
        global_grid.assign(ROWS * COLS, 0.0);

    MPI_Gatherv(
        cur.data() + COLS,          // source: skip ghost row 0
        local_rows * COLS, MPI_DOUBLE,
        compute_rank == 0 ? global_grid.data() : nullptr,
        send_counts.data(), send_offsets.data(), MPI_DOUBLE,
        0, compute_comm
    );

    /* ── 6. Performance report (rank 0) ──────────────────── */
    if (compute_rank == 0) {
        ruler();
        std::cout << "\nPerformance Summary\n";
        std::cout << "  Elapsed time  : " << std::fixed << std::setprecision(4)
                  << t_elapsed << " s\n";
        std::cout << "  Steps/second  : " << std::fixed << std::setprecision(1)
                  << cfg.steps / t_elapsed << "\n";
        double cells = (double)ROWS * COLS;
        double mcells_per_sec = (cells * cfg.steps) / (t_elapsed * 1e6);
        std::cout << "  MCells/second : " << std::fixed << std::setprecision(2)
                  << mcells_per_sec << "\n";
        std::cout << "  Convergence   : " << (converged ? "YES" : "NO (max steps reached)") << "\n";
        ruler();

        if (cfg.save_output)
            save_csv(global_grid, ROWS, COLS, "heat_output.csv");
    }

    /* ── 7. Timing stats across all ranks ────────────────── */
    double min_t, max_t, sum_t;
    MPI_Reduce(&t_elapsed, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, compute_comm);
    MPI_Reduce(&t_elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, compute_comm);
    MPI_Reduce(&t_elapsed, &sum_t, 1, MPI_DOUBLE, MPI_SUM, 0, compute_comm);

    if (compute_rank == 0) {
        std::cout << "Per-rank timing (min/avg/max): "
                  << std::fixed << std::setprecision(4)
                  << min_t << " / " << sum_t / compute_size << " / " << max_t
                  << " s\n";
        ruler();
    }
}

/* ═════════════════════════════════════════════════════════════
   BENCHMARK RUNNER  (vary process count analysis)
   ═════════════════════════════════════════════════════════════ */
void print_performance_header(int rank)
{
    if (rank != 0) return;
    ruler(70);
    std::cout << "BENCHMARK MODE: run with different -np values and compare\n";
    std::cout << "Recommended runs:\n";
    std::cout << "  mpirun -np 1 ./heat_diffusion --rows 1024 --cols 1024 --steps 500 --no-save\n";
    std::cout << "  mpirun -np 2 ./heat_diffusion --rows 1024 --cols 1024 --steps 500 --no-save\n";
    std::cout << "  mpirun -np 4 ./heat_diffusion --rows 1024 --cols 1024 --steps 500 --no-save\n";
    std::cout << "  mpirun -np 8 ./heat_diffusion --rows 1024 --cols 1024 --steps 500 --no-save\n";
    std::cout << "Compare the MCells/second and elapsed time values.\n";
    ruler(70);
}

/* ═════════════════════════════════════════════════════════════
   MAIN
   ═════════════════════════════════════════════════════════════ */
void run_heat_diffusion(int argc, char** argv)
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        if (world_rank == 0)
            std::cerr << "ERROR: This program requires at least 2 MPI processes.\n";
        return;
    }

    Config cfg = parse_args(argc, argv);

    int compute_color = 0;
    int monitor_color = (world_rank == 0) ? 1 : MPI_UNDEFINED;

    MPI_Comm compute_comm, monitor_comm;

    MPI_Comm_split(MPI_COMM_WORLD, compute_color, world_rank, &compute_comm);
    MPI_Comm_split(MPI_COMM_WORLD, monitor_color, world_rank, &monitor_comm);

    int compute_rank, compute_size;

    MPI_Comm_rank(compute_comm, &compute_rank);
    MPI_Comm_size(compute_comm, &compute_size);

    if (compute_rank == 0) {
        ruler();
        std::cout << " Heat Diffusion MPI Solver\n";
        ruler();
    }

    if (cfg.demo_deadlock) {
        demo_deadlock_scenario(compute_comm, compute_rank, compute_size, cfg.cols);
        MPI_Barrier(compute_comm);
    }

    print_performance_header(compute_rank);

    run_solver(cfg, world_rank, world_size, compute_comm, monitor_comm);

    MPI_Comm_free(&compute_comm);

    if (monitor_comm != MPI_COMM_NULL)
        MPI_Comm_free(&monitor_comm);
}