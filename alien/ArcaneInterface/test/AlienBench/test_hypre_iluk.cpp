#include <stdio.h>
#include <unistd.h>   // pour getpid()
#include <stdlib.h>
#include <mpi.h>
#ifdef ALIEN_USE_HIP
#include <hip/hip_runtime.h>
#endif

#include "HYPRE.h"
#include "HYPRE_utilities.h"
#include "HYPRE_seq_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_IJ_mv.h"

#include <iostream>
// Umpire
//#include "umpire/ResourceManager.hpp"
//#include "umpire/Allocator.hpp"
//#include "umpire/strategy/QuickPool.hpp"

// Helpers génériques pour lire les variables d'environnement

// Lecture d'un entier (GiB) avec fallback + message
std::size_t get_env_size_gb_verbose(const char* name,
                                    std::size_t default_gb,
                                    int rank)
{
  const char* val = std::getenv(name);
  if (!val) {
    if (rank == 0) {
      std::cout << "[ENV] " << name << " non défini, défaut = "
                << default_gb << " GiB\n";
    }
    return default_gb;
  }

  try {
    std::size_t gb = static_cast<std::size_t>(std::stoul(val));
    if (rank == 0) {
      std::cout << "[ENV] " << name << " = " << gb << " GiB\n";
    }
    return gb;
  } catch (...) {
    if (rank == 0) {
      std::cerr << "[ENV] " << name << " invalide ('" << val
                << "'), fallback = " << default_gb << " GiB\n";
    }
    return default_gb;
  }
}

// Lecture d'un entier (MiB) avec fallback + message
std::size_t get_env_size_mb_verbose(const char* name,
                                    std::size_t default_mb,
                                    int rank)
{
  const char* val = std::getenv(name);
  if (!val) {
    if (rank == 0) {
      std::cout << "[ENV] " << name << " non défini, défaut = "
                << default_mb << " MiB\n";
    }
    return default_mb;
  }

  try {
    std::size_t mb = static_cast<std::size_t>(std::stoul(val));
    if (rank == 0) {
      std::cout << "[ENV] " << name << " = " << mb << " MiB\n";
    }
    return mb;
  } catch (...) {
    if (rank == 0) {
      std::cerr << "[ENV] " << name << " invalide ('" << val
                << "'), fallback = " << default_mb << " MiB\n";
    }
    return default_mb;
  }
}

// Lecture d'un double (fraction 0.0–1.0) avec fallback + message
double get_env_double_verbose(const char* name,
                              double default_val,
                              int rank)
{
  const char* val = std::getenv(name);
  if (!val) {
    if (rank == 0) {
      std::cout << "[ENV] " << name << " non défini, défaut = "
                << default_val << "\n";
    }
    return default_val;
  }

  try {
    double x = std::stod(val);
    if (rank == 0) {
      std::cout << "[ENV] " << name << " = " << x << "\n";
    }
    return x;
  } catch (...) {
    if (rank == 0) {
      std::cerr << "[ENV] " << name << " invalide ('" << val
                << "'), fallback = " << default_val << "\n";
    }
    return default_val;
  }
}

#define CHECK_HYPRE(call) \
do { \
    int err = (call); \
    if (err) { \
        fprintf(stderr, "[%d] Erreur HYPRE à %s:%d : code %d\n", rank, __FILE__, __LINE__, err); \
        MPI_Abort(MPI_COMM_WORLD, err); \
    } \
} while(0)

#ifdef ALIEN_USE_HIP
#define CHECK_HIP(call) \
do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "[%d] Erreur HIP à %s:%d : %s\n", rank, __FILE__, __LINE__, hipGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Kernel HIP pour remplir un buffer device
__global__ void fill_buffer(double *buf, double v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = v;
}

// Variantes taggées par rang MPI (même code, nom différent)
__global__ void fill_buffer_rank0(double *buf, double v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = v;
}

__global__ void fill_buffer_rank1(double *buf, double v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = v;
}

__global__ void fill_buffer_rank2(double *buf, double v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = v;
}

__global__ void fill_buffer_rank3(double *buf, double v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = v;
}
#endif

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 0. Mapping PID ↔ rank pour post-traitement rocprof
    {
        int pid = getpid();
        if (rank == 0) {
            std::remove("rank_pid_map.txt");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        FILE* f = fopen("rank_pid_map.txt", "a");
        if (f) {
            fprintf(f, "%d %d\n", rank, pid);
            fclose(f);
        } else if (rank == 0) {
            fprintf(stderr, "Impossible d'ouvrir rank_pid_map.txt\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // 1. Initialisation de HYPRE
    HYPRE_Init();
    HYPRE_DeviceInitialize();
    HYPRE_SetSpGemmUseVendor(0);  // hypre GPU (pas rocSPARSE)
    HYPRE_SetUseGpuRand(1);       // RNG GPU

    // 2. Initialisation Umpire + création des pools contrôlés par env

    // Pool DEVICE ample pour Hypre (pour ce cas mono-GPU)
    std::size_t dev_pool_gb = get_env_size_gb_verbose(
        "UMPIRE_DEVICE_POOL_SIZE_GB", 24, rank);   // défaut 24 GiB

    std::size_t host_pool_mb = get_env_size_mb_verbose(
        "UMPIRE_HOST_POOL_SIZE_MB", 8192, rank);   // défaut 8192 MiB (~8 GiB)

    double hypre_gpu_frac = 1.0; // Hypre utilise tout ce pool DEVICE

    //auto& rm = umpire::ResourceManager::getInstance();

    // --- Pool DEVICE ---
    //auto dev_alloc = rm.getAllocator("DEVICE");

    std::size_t dev_pool_bytes_total =
        dev_pool_gb * 1024ul * 1024ul * 1024ul;

    std::size_t hypre_dev_bytes = dev_pool_bytes_total;

    //auto hypre_dev_pool = rm.makeAllocator<umpire::strategy::QuickPool>(
    //    "HYPRE_DEVICE_POOL", dev_alloc, hypre_dev_bytes);

    if (rank == 0) {
        std::cout << "[UMPIRE] HYPRE_DEVICE_POOL : "
                  << hypre_dev_bytes / (1024.0 * 1024.0 * 1024.0)
                  << " GiB (pool total DEVICE = " << dev_pool_gb
                  << " GiB, fraction = " << hypre_gpu_frac << ")\n";
    }

    // --- Pool UM (Unified Memory) optionnel ---
    //bool has_um_base = rm.isAllocator("UM");
    bool has_um_base = false ;
    /*
    if (has_um_base) {
        auto um_alloc = rm.getAllocator("UM");

        std::size_t um_pool_gb = get_env_size_gb_verbose(
            "UMPIRE_UM_POOL_SIZE_GB", 8, rank);   // défaut 8 GiB
        std::size_t um_pool_bytes =
            um_pool_gb * 1024ul * 1024ul * 1024ul;

        auto hypre_um_pool = rm.makeAllocator<umpire::strategy::QuickPool>(
            "HYPRE_UVM_POOL", um_alloc, um_pool_bytes);

        if (rank == 0) {
            std::cout << "[UMPIRE] UM base présent, création HYPRE_UVM_POOL ("
                      << um_pool_bytes / (1024.0 * 1024.0 * 1024.0)
                      << " GiB)\n";
        }
    } else if (rank == 0) {
        std::cout << "[UMPIRE] Aucun allocateur 'UM' trouvé, pas de HYPRE_UVM_POOL.\n";
    }*/

    // --- Pool HOST (DDR) ---
    /*
    auto host_alloc = rm.getAllocator("HOST");
    std::size_t host_pool_bytes =
        host_pool_mb * 1024ul * 1024ul;

    auto hypre_host_pool = rm.makeAllocator<umpire::strategy::QuickPool>(
        "HYPRE_HOST_POOL", host_alloc, host_pool_bytes);

    if (rank == 0) {
        std::cout << "[UMPIRE] HYPRE_HOST_POOL : "
                  << host_pool_bytes / (1024.0 * 1024.0)
                  << " MiB\n";
    }

    // 3. Lier hypre aux pools Umpire AVANT toute création d'objet HYPRE
    if (has_um_base)
    {
        CHECK_HYPRE(HYPRE_SetUmpireDevicePoolName("HYPRE_DEVICE_POOL"));
        CHECK_HYPRE(HYPRE_SetUmpireUMPoolName("HYPRE_UVM_POOL"));
    }
    // Vérification des allocateurs Umpire visibles
    for (auto name : rm.getAllocatorNames()) {
        if (rank == 0) std::cout << "UMPIRE allocator: " << name << std::endl;
    }

    bool has_device_pool = rm.isAllocator("HYPRE_DEVICE_POOL");
    bool has_uvm_pool    = rm.isAllocator("HYPRE_UVM_POOL");

    if (rank == 0) {
        std::cout << "HYPRE_DEVICE_POOL présent ? " << has_device_pool << std::endl;
        std::cout << "HYPRE_UVM_POOL présent ? "   << has_uvm_pool    << std::endl;
    }
    */
    // ===========================
    // DEBUT DU CODE HYPRE CLASSIQUE
    // ===========================
    const HYPRE_Int nx = 200, ny = 200, nz = 100;
    const HYPRE_Int global_rows = nx * ny * nz;
    const HYPRE_Int local_rows = global_rows / size;
    const HYPRE_Int ilower = rank * local_rows;
    const HYPRE_Int iupper = (rank + 1) * local_rows - 1;
    int max_nnz_per_row = 7;
    int nnz_alloc = local_rows * max_nnz_per_row;

    double t_asm_start = MPI_Wtime(); // assemblage host

    HYPRE_Int *h_row_ptr = (HYPRE_Int*) malloc((local_rows + 1) * sizeof(HYPRE_Int));
    HYPRE_Int *h_col_ind = (HYPRE_Int*) malloc(nnz_alloc * sizeof(HYPRE_Int));
    double    *h_values  = (double*)    malloc(nnz_alloc * sizeof(double));
    double    *h_b       = (double*)    malloc(local_rows * sizeof(double));

    if (!h_row_ptr || !h_col_ind || !h_values || !h_b) {
        fprintf(stderr, "[%d] Erreur malloc CPU\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int pos = 0;
    for (int local_i = 0; local_i < local_rows; ++local_i) {
        HYPRE_Int global_row = ilower + local_i;
        HYPRE_Int k = global_row / (nx * ny);
        HYPRE_Int rem = global_row % (nx * ny);
        HYPRE_Int j = rem / nx;
        HYPRE_Int i = rem % nx;

        h_row_ptr[local_i] = pos;
        int diag_pos = pos;

        h_col_ind[pos] = global_row;
        h_values[pos]  = 0.0;
        double diag    = 0.0;
        pos++;

        if (i > 0) {
            h_col_ind[pos] = global_row - 1;
            h_values[pos]  = -1.0;
            diag += 1.0;
            pos++;
        }
        if (i < nx-1) {
            h_col_ind[pos] = global_row + 1;
            h_values[pos]  = -1.0;
            diag += 1.0;
            pos++;
        }
        if (j > 0) {
            h_col_ind[pos] = global_row - nx;
            h_values[pos]  = -1.0;
            diag += 1.0;
            pos++;
        }
        if (j < ny-1) {
            h_col_ind[pos] = global_row + nx;
            h_values[pos]  = -1.0;
            diag += 1.0;
            pos++;
        }
        if (k > 0) {
            h_col_ind[pos] = global_row - nx*ny;
            h_values[pos]  = -1.0;
            diag += 1.0;
            pos++;
        }
        if (k < nz-1) {
            h_col_ind[pos] = global_row + nx*ny;
            h_values[pos]  = -1.0;
            diag += 1.0;
            pos++;
        }

        h_values[diag_pos] = diag + 1.0;
        h_b[local_i]       = 1.0;
    }
    h_row_ptr[local_rows] = pos;

    // Assemblage HYPRE en mode host
    CHECK_HYPRE(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST));
    CHECK_HYPRE(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST));
    //CHECK_HYPRE(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
    //CHECK_HYPRE(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));


    HYPRE_IJMatrix A_ij;
    CHECK_HYPRE(HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A_ij));
    CHECK_HYPRE(HYPRE_IJMatrixSetObjectType(A_ij, HYPRE_PARCSR));
    CHECK_HYPRE(HYPRE_IJMatrixInitialize(A_ij));

    HYPRE_Int *rows  = (HYPRE_Int*) malloc(local_rows * sizeof(HYPRE_Int));
    HYPRE_Int *ncols = (HYPRE_Int*) malloc(local_rows * sizeof(HYPRE_Int));
    for (int i = 0; i < local_rows; ++i) {
        rows[i]  = ilower + i;
        ncols[i] = h_row_ptr[i+1] - h_row_ptr[i];
    }
    for (int i = 0; i < local_rows; ++i) {
        CHECK_HYPRE(HYPRE_IJMatrixSetValues(
            A_ij, 1, &ncols[i], &rows[i],
            &h_col_ind[h_row_ptr[i]],
            &h_values[h_row_ptr[i]]
        ));
    }
    CHECK_HYPRE(HYPRE_IJMatrixAssemble(A_ij));
    HYPRE_ParCSRMatrix par_A;
    CHECK_HYPRE(HYPRE_IJMatrixGetObject(A_ij, (void**)&par_A));

    HYPRE_IJVector B_ij, X_ij;
    CHECK_HYPRE(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &B_ij));
    CHECK_HYPRE(HYPRE_IJVectorSetObjectType(B_ij, HYPRE_PARCSR));
    CHECK_HYPRE(HYPRE_IJVectorInitialize(B_ij));
    CHECK_HYPRE(HYPRE_IJVectorSetValues(B_ij, local_rows, rows, h_b));
    CHECK_HYPRE(HYPRE_IJVectorAssemble(B_ij));

    CHECK_HYPRE(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &X_ij));
    CHECK_HYPRE(HYPRE_IJVectorSetObjectType(X_ij, HYPRE_PARCSR));
    CHECK_HYPRE(HYPRE_IJVectorInitialize(X_ij));
    HYPRE_Complex *x_values = (HYPRE_Complex*) malloc(local_rows * sizeof(HYPRE_Complex));
    for (int i = 0; i < local_rows; ++i)
        x_values[i] = 1e-6 * ((double)rand() / RAND_MAX);
    CHECK_HYPRE(HYPRE_IJVectorSetValues(X_ij, local_rows, rows, x_values));
    CHECK_HYPRE(HYPRE_IJVectorAssemble(X_ij));

    HYPRE_ParVector par_b, par_x;
    CHECK_HYPRE(HYPRE_IJVectorGetObject(B_ij, (void**)&par_b));
    CHECK_HYPRE(HYPRE_IJVectorGetObject(X_ij, (void**)&par_x));

    double t_asm_end = MPI_Wtime();
    double t_asm = t_asm_end - t_asm_start;

    free(h_row_ptr); free(h_col_ind); free(h_values); free(h_b);
    free(rows); free(ncols); free(x_values);

    // Passage en mode DEVICE pour la résolution
    //CHECK_HYPRE(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
    CHECK_HYPRE(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));
    //CHECK_HYPRE(HYPRE_SetGPUMemoryPoolSize(0));  // désactive le pool interne

    // ===========================
    // Kernel de marquage par rang juste avant BiCGSTAB+FSAI
    // ===========================
    printf("[DEBUG] rank %d avant kernel de marquage\n", rank); fflush(stdout);

#ifdef ALIEN_USE_HIP
    {
        const int n = 1;
        double *d_buf = nullptr;
        CHECK_HIP( hipMalloc(&d_buf, n * sizeof(double)) );
        dim3 grid(1), block(1);
        if (rank == 0) {
            printf("[DEBUG] rank 0 -> fill_buffer_rank0\n"); fflush(stdout);
            fill_buffer_rank0<<<grid, block>>>(d_buf, 0.0, n);
        } else if (rank == 1) {
            printf("[DEBUG] rank 1 -> fill_buffer_rank1\n"); fflush(stdout);
            fill_buffer_rank1<<<grid, block>>>(d_buf, 0.0, n);
        } else if (rank == 2) {
            printf("[DEBUG] rank 2 -> fill_buffer_rank2\n"); fflush(stdout);
            fill_buffer_rank2<<<grid, block>>>(d_buf, 0.0, n);
        } else {
            printf("[DEBUG] rank 3 -> fill_buffer_rank3\n"); fflush(stdout);
            fill_buffer_rank3<<<grid, block>>>(d_buf, 0.0, n);
        }
        CHECK_HIP( hipDeviceSynchronize() );
        CHECK_HIP( hipFree(d_buf) );
    }
#endif

    printf("[DEBUG] rank %d après kernel de marquage\n", rank); fflush(stdout);

    // ===========================
    // BiCGSTAB + ILU
    // ===========================
    HYPRE_Solver solver_main;
    CHECK_HYPRE(HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver_main));
    CHECK_HYPRE(HYPRE_ParCSRBiCGSTABSetMaxIter(solver_main, 10000));
    CHECK_HYPRE(HYPRE_ParCSRBiCGSTABSetTol(solver_main, 1e-9));
    CHECK_HYPRE(HYPRE_ParCSRBiCGSTABSetPrintLevel(solver_main, 0));
    printf("[DEBUG] rank %d aprèsbicgstab build\n", rank); fflush(stdout);

    HYPRE_Solver ilu;
    std::string precond_name = "iluk";
    CHECK_HYPRE(HYPRE_ILUCreate(&ilu));
    int ilu_type = 0 ;
    int max_iter = 1 ;
    double tol = 0. ;
    int reordering = 0 ;
    int fill = 3 ;
    int print_level = 3 ;
    CHECK_HYPRE(HYPRE_ILUSetType(ilu, ilu_type)); /* 0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50 */
    CHECK_HYPRE(HYPRE_ILUSetMaxIter(ilu, max_iter));
    CHECK_HYPRE(HYPRE_ILUSetTol(ilu, tol));
    CHECK_HYPRE(HYPRE_ILUSetLocalReordering(ilu, reordering)); /* 0: none, 1: RCM */
    CHECK_HYPRE(HYPRE_ILUSetLevelOfFill(ilu, fill));
    //precond_destroy_function = HYPRE_ILUDestroy;
    CHECK_HYPRE(HYPRE_ParCSRBiCGSTABSetPrecond(
        solver_main,
        HYPRE_ILUSolve,      // Solve ILU
        HYPRE_ILUSetup,      // Setup ILU
        ilu
    ));
    printf("[DEBUG] rank %d aprè iluk build\n", rank); fflush(stdout);
    
      
    int    its_main = 0;
    double time_main_setup_start = MPI_Wtime();

    // SETUP : construction ILU + structures GPU
    CHECK_HYPRE(HYPRE_ParCSRBiCGSTABSetup(solver_main, par_A, par_b, par_x));
    printf("[DEBUG] rank %d aprè bics setup\n", rank); fflush(stdout);

    double time_main_setup_end = MPI_Wtime();

    // Coalescence du pool DEVICE après setup FSAI pour limiter les realloc/memset en solve
    bool use_umpire = false ;
    /*
    if(use_umpire)
    {
        auto& rm2 = umpire::ResourceManager::getInstance();
        auto alloc = rm2.getAllocator("HYPRE_DEVICE_POOL");
        auto* strat = alloc.getAllocationStrategy();
        auto* qp = dynamic_cast<umpire::strategy::QuickPool*>(strat);

        if (qp) {
            if (rank == 0)
                std::cout << "[UMPIRE] Coalescing HYPRE_DEVICE_POOL après FSAISetup\n";
            qp->coalesce();
        } else if (rank == 0) {
            std::cout << "[UMPIRE] HYPRE_DEVICE_POOL n'est pas un QuickPool, pas de coalesce.\n";
        }
    }*/

    printf("[DEBUG] rank %d avant bics solve\n", rank); fflush(stdout);
    double time_main_solve_start = MPI_Wtime();

    // SOLVE : la mémoire est stabilisée, on limite les fillBufferAligned massifs
    CHECK_HYPRE(HYPRE_ParCSRBiCGSTABSolve(solver_main, par_A, par_b, par_x));
    printf("[DEBUG] rank %d apres bics solve\n", rank); fflush(stdout);

    double time_main_solve_end = MPI_Wtime();

    double time_main_setup = time_main_setup_end - time_main_setup_start;
    double time_main_solve = time_main_solve_end - time_main_solve_start;
    double time_main_total = time_main_setup + time_main_solve;

    CHECK_HYPRE(HYPRE_ParCSRBiCGSTABGetNumIterations(solver_main, &its_main));
    double final_res_main = 0.0;
    CHECK_HYPRE(HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver_main, &final_res_main));

    if (rank == 0) {
        printf("=== BiCGSTAB+ILUK ===\n");
        printf("Assemblage host (s)         : %g\n", t_asm);
        printf("Setup ILUK (s)              : %g\n", time_main_setup);
        printf("Solve ILUK (s)              : %g\n", time_main_solve);
        printf("Temps total ILUK (s)        : %g\n", time_main_total);
        printf("ItérationsILUK               %d\n", its_main);
        printf("Résidu relatif finalILUK   : %e\n", final_res_main);
    }

    // Nettoyage
    CHECK_HYPRE(HYPRE_ILUDestroy(ilu));
    CHECK_HYPRE(HYPRE_ParCSRBiCGSTABDestroy(solver_main));
    CHECK_HYPRE(HYPRE_IJMatrixDestroy(A_ij));
    CHECK_HYPRE(HYPRE_IJVectorDestroy(B_ij));
    CHECK_HYPRE(HYPRE_IJVectorDestroy(X_ij));
    HYPRE_Finalize();
    MPI_Finalize();
    return 0;
}
