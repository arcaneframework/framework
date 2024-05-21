//@HEADER
// ************************************************************************
//
//                 Belos: Block Linear Solvers Package
//                  Copyright 2004 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
//@HEADER
//
// This test generates diagonal matrices for block GMRES to solve.
//
// NOTE: No preconditioner is used in this case.
//

// Belos
#include <BelosConfigDefs.hpp>
#include <BelosTpetraAdapter.hpp>

// Tpetra
#include <Tpetra_Core.hpp>
#include <Tpetra_Map_fwd.hpp>
#include <Tpetra_Vector_fwd.hpp>
#include <Tpetra_CrsMatrix_fwd.hpp>
#include <Tpetra_MultiVector_fwd.hpp>

// Teuchos
#include <Teuchos_Time.hpp>
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_StandardCatchMacros.hpp"


using std::vector;
using Teuchos::RCP;
using Teuchos::rcp;

using namespace Belos;

template<typename ScalarT, typename MVectorT, typename OpT>
RCP<Belos::SolverManager<ScalarT, MVectorT, OpT> >
private_belos_solver_create(std::string const& solverName, Teuchos::RCP<Teuchos::ParameterList> const& solver_parameters)
{
    std::cout<<"CREATE BELOS SOLVER : "<<solverName<<std::endl ;

    using Teuchos::Comm;
    using Teuchos::RCP;
    using Teuchos::rcp;

    Belos::SolverFactory<double, MVectorT, OpT> factory;
    RCP<Belos::SolverManager<double, MVectorT, OpT> > solver;
    try {
      solver = factory.create (solverName, solver_parameters);
    } catch (std::exception& e) {
      std::cout << "*** FAILED: Belos::SolverFactory::create threw an exception: "
          << e.what () << std::endl;
    }
    if (solver.is_null ()) {
      std::cout << "*** FAILED to create solver \"" << solverName
          << "\" from Belos package" << std::endl;
    }
    std::cout<<"FIN CREATE BELOS SOLVER : "<<solver.is_null ()<<std::endl ;
    return solver ;
}

template<typename ScalarT, typename MVectorT, typename OpT>
RCP<Belos::SolverManager<ScalarT, MVectorT, OpT> >
belos_solver_create(std::string const& solverName,Teuchos::RCP<Teuchos::ParameterList> const& solver_parameters) ;

#include <Kokkos_Macros.hpp>

#ifdef KOKKOS_ENABLE_SERIAL
typedef Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> SerialNT;
typedef Tpetra::MultiVector<double,int,int,SerialNT> SerialMV;
typedef Tpetra::Operator<double,int,int,SerialNT> SerialOP;

template<>
RCP<Belos::SolverManager<double, SerialMV, SerialOP> >
belos_solver_create(std::string const& solverName, Teuchos::RCP<Teuchos::ParameterList> const& solver_parameters)
{
  return private_belos_solver_create<double,SerialMV,SerialOP>(solverName,solver_parameters) ;
}
#endif

#ifdef KOKKOS_ENABLE_OPENMP
typedef Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::OpenMP, Kokkos::HostSpace> OMPNT;
typedef Tpetra::MultiVector<double,int,int,OMPNT> OMPMV;
typedef Tpetra::Operator<double,int,int,OMPNT> OMPOP;

template<>
RCP<Belos::SolverManager<double, OMPMV, OMPOP> >
belos_solver_create(std::string const& solverName, Teuchos::RCP<Teuchos::ParameterList> const& solver_parameters)
{
  return private_belos_solver_create<double,OMPMV,OMPOP>(solverName,solver_parameters) ;
}
#endif

#ifdef KOKKOS_ENABLE_THREADS
typedef Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::Threads, Kokkos::HostSpace> THNT;
typedef Tpetra::MultiVector<double,int,int,THNT> THMV;
typedef Tpetra::Operator<double,int,int,THNT> THOP;

template<>
RCP<Belos::SolverManager<double, THMV, THOP> >
belos_solver_create(std::string const& solverName, Teuchos::RCP<Teuchos::ParameterList> const& solver_parameters)
{
  return private_belos_solver_create<double,THMV,THOP>(solverName,solver_parameters) ;
}
#endif
/*---------------------------------------------------------------------------*/
#ifdef KOKKOS_ENABLE_CUDA
typedef Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::Cuda, Kokkos::HostSpace> CudaNT;
typedef Tpetra::MultiVector<double,int,int,CudaNT> CudaMV;
typedef Tpetra::Operator<double,int,int,CudaNT> CudaOP;

template<>
RCP<Belos::SolverManager<double, CudaMV, CudaOP> >
belos_solver_create(std::string const& solverName, Teuchos::RCP<Teuchos::ParameterList> const& solver_parameters)
{
  return private_belos_solver_create<double,CudaMV,CudaOP>(solverName,solver_parameters) ;
}
#endif
