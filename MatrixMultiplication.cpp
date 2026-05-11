#include "MatrixMultiplication.h"
#include "Shared.h"

#include <mpi.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

/* ═════════════════════════════════════════════════════════════
   CONFIGURATION
   ═════════════════════════════════════════════════════════════ */

enum MatCommMode { MAT_BLOCKING, MAT_NONBLOCKING, MAT_DEADLOCK_DEMO };

struct MatConfig {
    int         M = 512;   // rows of A  (and C)
    int         K = 512;   // cols of A  / rows of B
    int         N = 512;   // cols of B  (and C)
    MatCommMode mode = MAT_NONBLOCKING;
    bool        save_output = true;
    bool        verify = false; // cheap correctness check (small matrices only)
};

static MatConfig parse_args(int argc, char** argv)
{
    MatConfig cfg;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];

        if (a == "--M" && i + 1 < argc) {
            cfg.M = stoi(argv[++i]);
            if (cfg.M <= 0) { cerr << "M must be > 0\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
        }
        else if (a == "--K" && i + 1 < argc) {
            cfg.K = stoi(argv[++i]);
            if (cfg.K <= 0) { cerr << "K must be > 0\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
        }
        else if (a == "--N" && i + 1 < argc) {
            cfg.N = stoi(argv[++i]);
            if (cfg.N <= 0) { cerr << "N must be > 0\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
        }
        else if (a == "--no-save")  cfg.save_output = false;
        else if (a == "--verify")   cfg.verify = true;
    }
    return cfg;
}

/* ═════════════════════════════════════════════════════════════
   DEADLOCK DEMO
   Mirrors the heat-diffusion deadlock demo:
   shows a broken send-before-receive pattern on rank pairs,
   then fixes it with non-blocking calls.
   ═════════════════════════════════════════════════════════════ */
static void demo_deadlock_scenario(int rank, int size, MPI_Comm comm,
    int N, const vector<double>& local_row_block)
{
    // Each rank will try to exchange its local block with rank+1 (or rank-1).
    // The "broken" version has every rank MPI_Send before MPI_Recv → deadlock.

    const int TAG = 10;
    int partner = (rank % 2 == 0) ? rank + 1 : rank - 1;
    bool has_partner = (partner >= 0 && partner < size);

    int block_size = (int)local_row_block.size();
    vector<double> recv_buf(block_size, 0.0);

    if (rank == 0) {
        ruler();
        cout << "\n[DEADLOCK DEMO - MATRIX MULTIPLICATION]\n";
        cout << "Every rank tries to MPI_Send its row block first,\n";
        cout << "then MPI_Recv from its partner.\n";
        cout << "With large enough buffers this deadlocks.\n\n";
        ruler();
    }
    MPI_Barrier(comm);

    // ── BROKEN (skipped to avoid hanging) ──────────────────────
    if (rank == 0) {
        cout << "\nBroken version skipped to avoid hanging.\n";
        ruler();
    }
    MPI_Barrier(comm);

    // ── FIXED: non-blocking ─────────────────────────────────────
    if (rank == 0) {
        cout << "\n[DEADLOCK FIX - NONBLOCKING]\n";
        cout << "Using MPI_Isend + MPI_Irecv for row-block exchange.\n";
        cout << "This avoids circular waiting.\n\n";
    }
    MPI_Barrier(comm);

    MPI_Request reqs[2];
    int nreqs = 0;

    if (has_partner) {
        MPI_Irecv(recv_buf.data(), block_size, MPI_DOUBLE,
            partner, TAG, comm, &reqs[nreqs++]);
        MPI_Isend(local_row_block.data(), block_size, MPI_DOUBLE,
            partner, TAG, comm, &reqs[nreqs++]);
        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
    }

    cout << "[rank " << rank << "] row-block exchange completed successfully.\n";
    MPI_Barrier(comm);

    if (rank == 0) {
        ruler();
        cout << "Deadlock-free Matrix Multiplication communication completed.\n";
        ruler();
    }
}

/* ═════════════════════════════════════════════════════════════
   ROW DISTRIBUTION HELPERS
   ═════════════════════════════════════════════════════════════ */
static void build_distribution(int total_rows, int size,
    vector<int>& counts, vector<int>& offsets)
{
    counts.resize(size);
    offsets.resize(size);
    int base = total_rows / size;
    int rem = total_rows % size;
    for (int p = 0; p < size; ++p) {
        counts[p] = base + (p < rem ? 1 : 0);
        offsets[p] = (p == 0) ? 0 : offsets[p - 1] + counts[p - 1];
    }
}

/* ═════════════════════════════════════════════════════════════
   LOCAL MULTIPLY  –  C_local = A_local * B
   A_local : local_rows × K
   B        : K × N  (every rank holds a full copy)
   C_local  : local_rows × N
   ═════════════════════════════════════════════════════════════ */
static void local_multiply(const vector<double>& A_local,
    const vector<double>& B,
    vector<double>& C_local,
    int local_rows, int K, int N)
{
    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k)
                sum += A_local[i * K + k] * B[k * N + j];
            C_local[i * N + j] = sum;
        }
}

/* ═════════════════════════════════════════════════════════════
   SCATTER / GATHER WRAPPERS
   Blocking and non-blocking flavours keep the same interface so
   the solver can pick between them via the comm_choice flag.
   ═════════════════════════════════════════════════════════════ */

   // Scatter rows of A from rank 0 to all ranks (blocking Scatterv).
static void scatter_A_blocking(const vector<double>& global_A,
    vector<double>& local_A,
    int local_rows, int K,
    const vector<int>& send_counts,
    const vector<int>& send_offsets,
    int rank, MPI_Comm comm)
{
    (void)rank; // used implicitly via MPI
    MPI_Scatterv(
        global_A.empty() ? nullptr : global_A.data(),
        send_counts.data(), send_offsets.data(), MPI_DOUBLE,
        local_A.data(), local_rows * K, MPI_DOUBLE,
        0, comm
    );
}

// Non-blocking scatter: rank 0 Isend's each slice; every rank Irecv's its slice.
static void scatter_A_nonblocking(const vector<double>& global_A,
    vector<double>& local_A,
    int local_rows, int K,
    const vector<int>& send_counts,
    const vector<int>& send_offsets,
    int rank, int size, MPI_Comm comm)
{
    const int TAG_SCATTER = 20;
    vector<MPI_Request> reqs;

    if (rank == 0) {
        reqs.resize(size);
        for (int p = 0; p < size; ++p)
            MPI_Isend(global_A.data() + send_offsets[p],
                send_counts[p], MPI_DOUBLE,
                p, TAG_SCATTER, comm, &reqs[p]);
    }

    MPI_Request recv_req;
    MPI_Irecv(local_A.data(), local_rows * K, MPI_DOUBLE,
        0, TAG_SCATTER, comm, &recv_req);
    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

    if (rank == 0)
        MPI_Waitall(size, reqs.data(), MPI_STATUSES_IGNORE);
}

// Gather C rows back to rank 0 (blocking Gatherv).
static void gather_C_blocking(const vector<double>& local_C,
    vector<double>& global_C,
    int local_rows, int N,
    const vector<int>& recv_counts,
    const vector<int>& recv_offsets,
    int rank, MPI_Comm comm)
{
    (void)rank;
    MPI_Gatherv(
        local_C.data(), local_rows * N, MPI_DOUBLE,
        global_C.empty() ? nullptr : global_C.data(),
        recv_counts.data(), recv_offsets.data(), MPI_DOUBLE,
        0, comm
    );
}

// Non-blocking gather: every rank Isend's its result; rank 0 Irecv's all.
static void gather_C_nonblocking(const vector<double>& local_C,
    vector<double>& global_C,
    int local_rows, int N,
    const vector<int>& recv_counts,
    const vector<int>& recv_offsets,
    int rank, int size, MPI_Comm comm)
{
    const int TAG_GATHER = 21;
    MPI_Request send_req;
    MPI_Isend(local_C.data(), local_rows * N, MPI_DOUBLE,
        0, TAG_GATHER, comm, &send_req);

    if (rank == 0) {
        vector<MPI_Request> reqs(size);
        for (int p = 0; p < size; ++p)
            MPI_Irecv(global_C.data() + recv_offsets[p],
                recv_counts[p], MPI_DOUBLE,
                p, TAG_GATHER, comm, &reqs[p]);
        MPI_Waitall(size, reqs.data(), MPI_STATUSES_IGNORE);
    }

    MPI_Wait(&send_req, MPI_STATUS_IGNORE);
}

/* ═════════════════════════════════════════════════════════════
   MAIN SOLVER
   ═════════════════════════════════════════════════════════════ */
static void run_solver(const MatConfig& cfg, MPI_Comm compute_comm)
{
    int rank, size;
    MPI_Comm_rank(compute_comm, &rank);
    MPI_Comm_size(compute_comm, &size);

    const int M = cfg.M, K = cfg.K, N = cfg.N;

    // ── 1. Row distribution for A and C ──────────────────────
    vector<int> row_counts, row_offsets;
    build_distribution(M, size, row_counts, row_offsets);
    int local_rows = row_counts[rank];

    // Element-level counts/offsets for Scatterv / Gatherv
    vector<int> send_counts(size), send_offsets(size);
    vector<int> recv_counts(size), recv_offsets(size);
    for (int p = 0; p < size; ++p) {
        send_counts[p] = row_counts[p] * K;
        send_offsets[p] = row_offsets[p] * K;
        recv_counts[p] = row_counts[p] * N;
        recv_offsets[p] = row_offsets[p] * N;
    }

    if (rank == 0) {
        ruler();
        cout << "Matrix dimensions  : A(" << M << "x" << K
            << ") * B(" << K << "x" << N << ") = C(" << M << "x" << N << ")\n";
        cout << "Processes          : " << size << "\n";
        cout << "Mode               : ";
        switch (cfg.mode) {
        case MAT_BLOCKING:      cout << "blocking\n";      break;
        case MAT_NONBLOCKING:   cout << "non-blocking\n";  break;
        case MAT_DEADLOCK_DEMO: cout << "deadlock-demo\n"; break;
        }
        cout << "Row distribution   : ";
        for (int p = 0; p < size; ++p)
            cout << "P" << p << "=" << row_counts[p] << " ";
        cout << "\n";
        ruler();
    }

    // ── 2. Allocate buffers ───────────────────────────────────
    vector<double> global_A, global_C;
    vector<double> B(K * N, 0.0);

    if (rank == 0) {
        global_A.resize(M * K);
        global_C.resize(M * N, 0.0);
        load_csv("mat_A.csv", global_A, M, K);
        load_csv("mat_B.csv", B, K, N);
    }

    vector<double> local_A(local_rows * K, 0.0);
    vector<double> local_C(local_rows * N, 0.0);

    // ── 3. Broadcast B to all ranks ───────────────────────────
    MPI_Bcast(B.data(), K * N, MPI_DOUBLE, 0, compute_comm);

    // ── 4. Deadlock demo (if requested) ──────────────────────
    if (cfg.mode == MAT_DEADLOCK_DEMO) {
        // Populate local_A with dummy data for the demo exchange
        local_A.assign(local_rows * K, (double)rank);
        demo_deadlock_scenario(rank, size, compute_comm, N, local_A);
        MPI_Barrier(compute_comm);
        return; // skip actual computation in demo mode
    }

    // ── 5. Scatter A rows ─────────────────────────────────────
    double t_start = MPI_Wtime();

    if (cfg.mode == MAT_NONBLOCKING)
        scatter_A_nonblocking(global_A, local_A, local_rows, K,
            send_counts, send_offsets, rank, size, compute_comm);
    else
        scatter_A_blocking(global_A, local_A, local_rows, K,
            send_counts, send_offsets, rank, compute_comm);

    // ── 6. Local multiply ────────────────────────────────────
    local_multiply(local_A, B, local_C, local_rows, K, N);

    // ── 7. Gather C rows ─────────────────────────────────────
    if (cfg.mode == MAT_NONBLOCKING)
        gather_C_nonblocking(local_C, global_C, local_rows, N,
            recv_counts, recv_offsets, rank, size, compute_comm);
    else
        gather_C_blocking(local_C, global_C, local_rows, N,
            recv_counts, recv_offsets, rank, compute_comm);

    double t_elapsed = MPI_Wtime() - t_start;

    // ── 8. Performance report ─────────────────────────────────
    double min_t, max_t, sum_t;
    MPI_Reduce(&t_elapsed, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, compute_comm);
    MPI_Reduce(&t_elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, compute_comm);
    MPI_Reduce(&t_elapsed, &sum_t, 1, MPI_DOUBLE, MPI_SUM, 0, compute_comm);

    if (rank == 0) {
        double flops = 2.0 * M * K * N;   // 2 flops per multiply-add
        double gflops = flops / (t_elapsed * 1e9);

        ruler();
        cout << "\nPerformance Summary\n";
        cout << "  Elapsed time       : " << fixed << setprecision(4)
            << t_elapsed << " s\n";
        cout << "  GFlops/s           : " << fixed << setprecision(3)
            << gflops << "\n";
        cout << "  Per-rank (min/avg/max): "
            << fixed << setprecision(4)
            << min_t << " / " << sum_t / size << " / " << max_t << " s\n";
        ruler();

        // ── 9. Optional verification (small matrices) ─────────
        if (cfg.verify && M <= 64 && K <= 64 && N <= 64) {
            // Re-compute serial result and compare
            vector<double> C_ref(M * N, 0.0);
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    for (int k = 0; k < K; ++k)
                        C_ref[i * N + j] += global_A[i * K + k] * B[k * N + j];

            double max_err = 0.0;
            for (int idx = 0; idx < M * N; ++idx)
                max_err = max(max_err, fabs(global_C[idx] - C_ref[idx]));

            cout << "  Verification max_err : " << scientific << max_err
                << (max_err < 1e-9 ? "  [PASS]" : "  [FAIL]") << "\n";
            ruler();
        }

        // ── 10. Save result ───────────────────────────────────
        if (cfg.save_output)
            save_csv(global_C, M, N, "mat_C.csv");
    }
}

/* ═════════════════════════════════════════════════════════════
   PUBLIC ENTRY POINT  (called from Source.cpp)
   ═════════════════════════════════════════════════════════════ */
void run_matrix_multiplication(int argc, char** argv, int comm_choice)
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        if (world_rank == 0)
            cerr << "ERROR: This program requires at least 2 MPI processes.\n";
        return;
    }

    MatConfig cfg = parse_args(argc, argv);

    if (comm_choice == 1) cfg.mode = MAT_BLOCKING;
    else if (comm_choice == 2) cfg.mode = MAT_NONBLOCKING;
    else if (comm_choice == 3) cfg.mode = MAT_DEADLOCK_DEMO;

    // Split into a dedicated communicator (mirrors heat-diffusion pattern)
    MPI_Comm compute_comm;
    MPI_Comm_split(MPI_COMM_WORLD, /*color=*/0, world_rank, &compute_comm);

    int compute_rank;
    MPI_Comm_rank(compute_comm, &compute_rank);

    if (compute_rank == 0) {
        ruler();
        cout << " Matrix Multiplication MPI Solver\n";
        ruler();
    }

    run_solver(cfg, compute_comm);

    MPI_Comm_free(&compute_comm);
}