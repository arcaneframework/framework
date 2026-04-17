// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example 5

   Interface:    Linear-Algebraic (IJ)

   Compile with: make ex5

   Sample run:   mpirun -np 4 ex5

   Description:  This example solves the 2-D Laplacian problem with zero boundary
                 conditions on an n x n grid.  The number of unknowns is N=n^2.
                 The standard 5-point stencil is used, and we solve for the
                 interior nodes only.

                 This example solves the same problem as Example 3.  Available
                 solvers are AMG, PCG, and PCG with AMG or Parasails
                 preconditioners.  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Header file for examples
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_EXAMPLES_INCLUDES
#define HYPRE_EXAMPLES_INCLUDES

#include <HYPRE_config.h>

#if defined(HYPRE_EXAMPLE_USING_CUDA)

#include <cuda_runtime.h>

#ifndef HYPRE_USING_UNIFIED_MEMORY
#error *** Running the examples on GPUs requires Unified Memory. Please reconfigure and rebuild with --enable-unified-memory ***
#endif

static inline void*
gpu_malloc(size_t size)
{
   void *ptr = NULL;
   cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
   return ptr;
}

static inline void*
gpu_calloc(size_t num, size_t size)
{
   void *ptr = NULL;
   cudaMallocManaged(&ptr, num * size, cudaMemAttachGlobal);
   cudaMemset(ptr, 0, num * size);
   return ptr;
}

#define malloc(size) gpu_malloc(size)
#define calloc(num, size) gpu_calloc(num, size)
#define free(ptr) ( cudaFree(ptr), ptr = NULL )
#endif /* #if defined(HYPRE_EXAMPLE_USING_CUDA) */
#endif /* #ifndef HYPRE_EXAMPLES_INCLUDES */

#ifdef HYPRE_EXVIS
#include "vis.c"
#endif

#include <memory>
#include <vector>
#include <iostream>

#include "arcane/utils/Convert.h"
#include "arcane/utils/FatalErrorException.h"
#include "arccore/alina/Profiler.h"

using namespace Arcane;

int hypre_FlexGMRESModifyPCAMGExample(void *precond_data, int iterations,
                                      double rel_residual_norm);

#define my_min(a,b)  (((a)<(b)) ? (a) : (b))

extern "C++" void
_doHypreSolver(int nb_row,
               std::vector<ptrdiff_t> const& _ptr,
               std::vector<ptrdiff_t> const& _col,
               std::vector<double> const& _val,
               std::vector<double> const& _rhs,
               std::vector<double>& _x,
               int argc, char* argv[])
{
  auto& prof = Alina::Profiler::globalProfiler();
  auto t = prof.scoped_tic("Hypre");

  std::cout << "DO_HYPRE nb_row=" << nb_row << "\n";
  int i;
  int myid, num_procs;
  const int N = nb_row;

  int ilower, iupper;
  int local_size, extra;

  int solver_id;
  int vis, print_system;

  //double h, h2;

  HYPRE_IJMatrix A;
  HYPRE_ParCSRMatrix parcsr_A;
  HYPRE_IJVector b;
  HYPRE_ParVector par_b;
  HYPRE_IJVector x;
  HYPRE_ParVector par_x;

  HYPRE_Solver solver, precond;

  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  /* Initialize HYPRE */
  HYPRE_Initialize();

  /* Print GPU info */
  /* HYPRE_PrintDeviceInfo(); */
#if defined(HYPRE_USING_GPU)
  /* use vendor implementation for SpGEMM */
  HYPRE_SetSpGemmUseVendor(0);
#endif

  /* Default problem parameters */
  const int n = nb_row;
  solver_id = 0;
  vis = 0;
  print_system = 0;

  /* Parse command line */
  {
    int arg_index = 0;
    int print_usage = 0;

    while (arg_index < argc) {
      if (strcmp(argv[arg_index], "-n") == 0) {
        arg_index++;
        //n = atoi(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-solver") == 0) {
        arg_index++;
        solver_id = atoi(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-vis") == 0) {
        arg_index++;
        vis = 1;
      }
      else if (strcmp(argv[arg_index], "-print_system") == 0) {
        arg_index++;
        print_system = 1;
      }
      else if (strcmp(argv[arg_index], "-help") == 0) {
        print_usage = 1;
        break;
      }
      else {
        arg_index++;
      }
    }

    if ((print_usage) && (myid == 0)) {
      printf("\n");
      printf("Usage: %s [<options>]\n", argv[0]);
      printf("\n");
      printf("  -n <n>              : problem size in each direction (default: 33)\n");
      printf("  -solver <ID>        : solver ID\n");
      printf("                        0  - AMG (default) \n");
      printf("                        1  - AMG-PCG\n");
      printf("                        8  - ParaSails-PCG\n");
      printf("                        50 - PCG\n");
      printf("                        61 - AMG-FlexGMRES\n");
      printf("  -vis                : save the solution for GLVis visualization\n");
      printf("  -print_system       : print the matrix and rhs\n");
      printf("\n");
    }

    if (print_usage) {
      MPI_Finalize();
      return;
    }
  }

  // Fill nb value per row.
  std::vector<int> nb_value_per_row(n);
  for (int i = 0; i < n; ++i)
    nb_value_per_row[i] = static_cast<HYPRE_BigInt>(_ptr[i + 1] - _ptr[i]);

  // The column index is the same that '_col' from CSR Matrix
  // but we do a copy if index size is différent between Hypre and Alina.
  std::vector<HYPRE_BigInt> hypre_column_index(_col.begin(), _col.end());

  // Id of each row (in sequential, this is the same that the index)
  std::vector<HYPRE_Int> hypre_row_index(n);
  for (int i = 0; i < n; ++i) {
    hypre_row_index[i] = i;
  }

  /* Each processor knows only of its own rows - the range is denoted by ilower
     and upper.  Here we partition the rows. We account for the fact that
     N may not divide evenly by the number of processors. */
  local_size = N / num_procs;
  extra = N - local_size * num_procs;

  ilower = local_size * myid;
  ilower += my_min(myid, extra);

  iupper = local_size * (myid + 1);
  iupper += my_min(myid + 1, extra);
  iupper = iupper - 1;
  std::cout << "LOWER=" << ilower << " UPPER=" << iupper << "\n";
  /* How many rows do I have? */
  local_size = iupper - ilower + 1;

  {
    auto t = prof.scoped_tic("IJMatrix Create");
    /* Create the matrix.
       Note that this is a square matrix, so we indicate the row partition
       size twice (since number of rows = number of cols) */
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);

    /* Choose a parallel csr format storage (see the User's Manual) */
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);

    /* Initialize before setting coefficients */
    HYPRE_IJMatrixInitialize(A);
  }

  // Fill the matrix.
  {
    auto t = prof.scoped_tic("IJMatrix SetValues");

    HYPRE_IJMatrixSetValues(A, n,
                            nb_value_per_row.data(),
                            hypre_row_index.data(),
                            hypre_column_index.data(),
                            _val.data());
  }

  {
    auto t = prof.scoped_tic("IJMatrix Assemble");
    /* Assemble after setting the coefficients */
    HYPRE_IJMatrixAssemble(A);
  }

  /* Note: for the testing of small problems, one may wish to read
      in a matrix in IJ format (for the format, see the output files
      from the -print_system option).
      In this case, one would use the following routine:
      HYPRE_IJMatrixRead( <filename>, MPI_COMM_WORLD,
                          HYPRE_PARCSR, &A );
      <filename>  = IJ.A.out to read in what has been printed out
      by -print_system (processor numbers are omitted).
      A call to HYPRE_IJMatrixRead is an *alternative* to the
      following sequence of HYPRE_IJMatrix calls:
      Create, SetObjectType, Initialize, SetValues, and Assemble
   */

  /* Get the parcsr matrix object to use */
  HYPRE_IJMatrixGetObject(A, (void**)&parcsr_A);

  /* Create the rhs and solution */
  HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
  HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(b);

  HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
  HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(x);

  /* Set the rhs values to h^2 and the solution to zero */
  {
    //double *rhs_values, *x_values;
    int* rows;

    //rhs_values = (double*)calloc(local_size, sizeof(double));
    //x_values = (double*)calloc(local_size, sizeof(double));
    rows = (int*)calloc(local_size, sizeof(int));

    for (i = 0; i < local_size; i++) {
      //rhs_values[i] = h2;
      //x_values[i] = 0.0;
      rows[i] = ilower + i;
    }

    HYPRE_IJVectorSetValues(b, local_size, rows, _rhs.data());
    HYPRE_IJVectorSetValues(x, local_size, rows, _x.data());

    //free(x_values);
    //free(rhs_values);
    free(rows);
  }

  HYPRE_IJVectorAssemble(b);
  /*  As with the matrix, for testing purposes, one may wish to read in a rhs:
       HYPRE_IJVectorRead( <filename>, MPI_COMM_WORLD,
                                 HYPRE_PARCSR, &b );
       as an alternative to the
       following sequence of HYPRE_IJVectors calls:
       Create, SetObjectType, Initialize, SetValues, and Assemble
   */
  HYPRE_IJVectorGetObject(b, (void**)&par_b);

  HYPRE_IJVectorAssemble(x);
  HYPRE_IJVectorGetObject(x, (void**)&par_x);

  /*  Print out the system  - files names will be IJ.out.A.XXXXX
        and IJ.out.b.XXXXX, where XXXXX = processor id */
  if (print_system) {
    HYPRE_IJMatrixPrint(A, "IJ.out.A");
    HYPRE_IJVectorPrint(b, "IJ.out.b");
  }
  solver_id = 0;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ALINA_HYPRE_SOLVER", true))
    solver_id = v.value();

  double solver_tolerance = 1.0e-8;

  /* Choose a solver and solve the system */
  std::cout << "FINISH ASSEMBLING solver_id=" << solver_id << "\n";
  /* AMG */
  if (solver_id == 0) {
    auto t = prof.scoped_tic("HypreSolver AMG");
    int num_iterations;
    double final_res_norm;

    /* Create solver */
    HYPRE_BoomerAMGCreate(&solver);

    /* Set some parameters (See Reference Manual for more parameters) */
    HYPRE_BoomerAMGSetPrintLevel(solver, 3); /* print solve info + parameters */
    HYPRE_BoomerAMGSetOldDefault(solver); /* Falgout coarsening with modified classical interpolaiton */
    HYPRE_BoomerAMGSetRelaxType(solver, 3); /* G-S/Jacobi hybrid relaxation */
    HYPRE_BoomerAMGSetRelaxOrder(solver, 1); /* uses C/F relaxation */
    HYPRE_BoomerAMGSetNumSweeps(solver, 1); /* Sweeeps on each level */
    HYPRE_BoomerAMGSetMaxLevels(solver, 20); /* maximum number of levels */
    HYPRE_BoomerAMGSetTol(solver, solver_tolerance); /* conv. tolerance */

    /* Now setup and solve! */
    {
      auto t = prof.scoped_tic("AMG Setup");
      HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
    }
    {
      auto t = prof.scoped_tic("AMG Solve");
      HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
    }

    /* Run info - needed logging turned on */
    HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
    HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
    if (myid == 0) {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %e\n", final_res_norm);
      printf("\n");
    }

    /* Destroy solver */
    HYPRE_BoomerAMGDestroy(solver);
  }
  /* PCG */
  else if (solver_id == 50) {
    auto t = prof.scoped_tic("HypreSolver PCG");
    int num_iterations;
    double final_res_norm;

    /* Create solver */
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

    /* Set some parameters (See Reference Manual for more parameters) */
    HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
    HYPRE_PCGSetTol(solver, solver_tolerance); /* conv. tolerance */
    HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
    HYPRE_PCGSetPrintLevel(solver, 2); /* prints out the iteration info */
    HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

    /* Now setup and solve! */
    HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
    HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);

    /* Run info - needed logging turned on */
    HYPRE_PCGGetNumIterations(solver, &num_iterations);
    HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
    if (myid == 0) {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %e\n", final_res_norm);
      printf("\n");
    }

    /* Destroy solver */
    HYPRE_ParCSRPCGDestroy(solver);
  }
  /* PCG with AMG preconditioner */
  else if (solver_id == 1) {
    auto t = prof.scoped_tic("HypreSolver PCG-AMG");
    int num_iterations;
    double final_res_norm;

    /* Create solver */
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

    /* Set some parameters (See Reference Manual for more parameters) */
    HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
    HYPRE_PCGSetTol(solver, solver_tolerance); /* conv. tolerance */
    HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
    HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
    HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

    /* Now set up the AMG preconditioner and specify any parameters */
    HYPRE_BoomerAMGCreate(&precond);
    HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info */
    HYPRE_BoomerAMGSetCoarsenType(precond, 6);
    HYPRE_BoomerAMGSetOldDefault(precond);
    HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
    HYPRE_BoomerAMGSetNumSweeps(precond, 1);
    HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
    HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

    /* Set the PCG preconditioner */
    HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, precond);

    /* Now setup and solve! */
    prof.tic("Setup");
    HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
    prof.toc("Setup");
    prof.tic("Solve");
    HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);
    prof.toc("Solve");

    /* Run info - needed logging turned on */
    HYPRE_PCGGetNumIterations(solver, &num_iterations);
    HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
    if (myid == 0) {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %e\n", final_res_norm);
      printf("\n");
    }

    /* Destroy solver and preconditioner */
    HYPRE_ParCSRPCGDestroy(solver);
    HYPRE_BoomerAMGDestroy(precond);
  }
  /* PCG with Parasails Preconditioner */
  else if (solver_id == 8) {
    auto t = prof.scoped_tic("HypreSolver PCG - Parasails");
    int num_iterations;
    double final_res_norm;

    int sai_max_levels = 1;
    double sai_threshold = 0.1;
    double sai_filter = 0.05;
    int sai_sym = 1;

    /* Create solver */
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

    /* Set some parameters (See Reference Manual for more parameters) */
    HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
    HYPRE_PCGSetTol(solver, solver_tolerance); /* conv. tolerance */
    HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
    HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
    HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

    /* Now set up the ParaSails preconditioner and specify any parameters */
    HYPRE_ParaSailsCreate(MPI_COMM_WORLD, &precond);

    /* Set some parameters (See Reference Manual for more parameters) */
    HYPRE_ParaSailsSetParams(precond, sai_threshold, sai_max_levels);
    HYPRE_ParaSailsSetFilter(precond, sai_filter);
    HYPRE_ParaSailsSetSym(precond, sai_sym);
    HYPRE_ParaSailsSetLogging(precond, 3);

    /* Set the PCG preconditioner */
    HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn)HYPRE_ParaSailsSolve,
                        (HYPRE_PtrToSolverFcn)HYPRE_ParaSailsSetup, precond);

    /* Now setup and solve! */
    HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
    HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);

    /* Run info - needed logging turned on */
    HYPRE_PCGGetNumIterations(solver, &num_iterations);
    HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
    if (myid == 0) {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %e\n", final_res_norm);
      printf("\n");
    }

    /* Destory solver and preconditioner */
    HYPRE_ParCSRPCGDestroy(solver);
    HYPRE_ParaSailsDestroy(precond);
  }
  /* Flexible GMRES with  AMG Preconditioner */
  else if (solver_id == 61) {
    auto t = prof.scoped_tic("HypreSolver Flexible GMRES - AMG");
    int num_iterations;
    double final_res_norm;
    int restart = 30;
    int modify = 1;

    /* Create solver */
    HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &solver);

    /* Set some parameters (See Reference Manual for more parameters) */
    HYPRE_FlexGMRESSetKDim(solver, restart);
    HYPRE_FlexGMRESSetMaxIter(solver, 1000); /* max iterations */
    HYPRE_FlexGMRESSetTol(solver, solver_tolerance); /* conv. tolerance */
    HYPRE_FlexGMRESSetPrintLevel(solver, 2); /* print solve info */
    HYPRE_FlexGMRESSetLogging(solver, 1); /* needed to get run info later */

    /* Now set up the AMG preconditioner and specify any parameters */
    HYPRE_BoomerAMGCreate(&precond);
    HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info */
    HYPRE_BoomerAMGSetCoarsenType(precond, 6);
    HYPRE_BoomerAMGSetOldDefault(precond);
    HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
    HYPRE_BoomerAMGSetNumSweeps(precond, 1);
    HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
    HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

    /* Set the FlexGMRES preconditioner */
    HYPRE_FlexGMRESSetPrecond(solver, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                              (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, precond);

    if (modify) {
      /* this is an optional call  - if you don't call it, hypre_FlexGMRESModifyPCDefault
            is used - which does nothing.  Otherwise, you can define your own, similar to
            the one used here */
      HYPRE_FlexGMRESSetModifyPC(solver, (HYPRE_PtrToModifyPCFcn)hypre_FlexGMRESModifyPCAMGExample);
    }

    /* Now setup and solve! */
    {
      auto t = prof.scoped_tic("FlexGMRES Setup");
      HYPRE_ParCSRFlexGMRESSetup(solver, parcsr_A, par_b, par_x);
    }
    {
      auto t = prof.scoped_tic("FlexGMRES Solve");
      HYPRE_ParCSRFlexGMRESSolve(solver, parcsr_A, par_b, par_x);
    }

    /* Run info - needed logging turned on */
    HYPRE_FlexGMRESGetNumIterations(solver, &num_iterations);
    HYPRE_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
    if (myid == 0) {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %e\n", final_res_norm);
      printf("\n");
    }

    /* Destory solver and preconditioner */
    HYPRE_ParCSRFlexGMRESDestroy(solver);
    HYPRE_BoomerAMGDestroy(precond);
  }
  else {
    if (myid == 0) {
      ARCANE_FATAL("Invalid solver id '{0}' specified.", solver_id);
    }
  }

  if (print_system)
    HYPRE_IJVectorPrint(x, "IJ.out.x");

  /* Save the solution for GLVis visualization, see vis/glvis-ex5.sh */
  if (vis) {
#ifdef HYPRE_EXVIS
    FILE* file;
    char filename[255];

    int nvalues = local_size;
    int* rows = (int*)calloc(nvalues, sizeof(int));
    double* values = (double*)calloc(nvalues, sizeof(double));

    for (i = 0; i < nvalues; i++) {
      rows[i] = ilower + i;
    }

    /* get the local solution */
    HYPRE_IJVectorGetValues(x, nvalues, rows, values);

    sprintf(filename, "%s.%06d", "vis/ex5.sol", myid);
    if ((file = fopen(filename, "w")) == NULL) {
      printf("Error: can't open output file %s\n", filename);
      MPI_Finalize();
      exit(1);
    }

    /* save solution */
    for (i = 0; i < nvalues; i++) {
      fprintf(file, "%.14e\n", values[i]);
    }

    fflush(file);
    fclose(file);

    free(rows);
    free(values);

    /* save global finite element mesh */
    if (myid == 0) {
      GLVis_PrintGlobalSquareMesh("vis/ex5.mesh", n - 1);
    }
#endif
  }

  /* Clean up */
  HYPRE_IJMatrixDestroy(A);
  HYPRE_IJVectorDestroy(b);
  HYPRE_IJVectorDestroy(x);

  /* Finalize HYPRE */
  HYPRE_Finalize();

  /* Finalize MPI*/
  MPI_Finalize();

  return;
}

/*--------------------------------------------------------------------------
  hypre_FlexGMRESModifyPCAMGExample -

  This is an example (not recommended)
  of how we can modify things about AMG that
  affect the solve phase based on how FlexGMRES is doing...For
  another preconditioner it may make sense to modify the tolerance..
 *--------------------------------------------------------------------------*/

int hypre_FlexGMRESModifyPCAMGExample(void* precond_data, [[maybe_unused]] int iterations,
                                      double rel_residual_norm)
{

  if (rel_residual_norm > .1) {
    HYPRE_BoomerAMGSetNumSweeps((HYPRE_Solver)precond_data, 10);
  }
  else {
    HYPRE_BoomerAMGSetNumSweeps((HYPRE_Solver)precond_data, 1);
  }

  return 0;
}
