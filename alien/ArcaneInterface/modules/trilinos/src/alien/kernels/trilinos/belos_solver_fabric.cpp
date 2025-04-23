// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <Trilinos_version.h>
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

#if (TRILINOS_MAJOR_VERSION < 15)
#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#else
#include <Tpetra_KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#endif

#ifdef KOKKOS_ENABLE_SERIAL
#if (TRILINOS_MAJOR_VERSION < 15)
typedef Kokkos::Compat::KokkosSerialWrapperNode SerialNT ;
#else
typedef Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> SerialNT;
#endif
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
#if (TRILINOS_MAJOR_VERSION < 15)
typedef Kokkos::Compat::KokkosOpenMPWrapperNode OMPNT ;
#else
typedef Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::OpenMP, Kokkos::HostSpace> OMPNT;
#endif
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
#if (TRILINOS_MAJOR_VERSION < 15)
typedef Kokkos::Compat::KokkosThreadsWrapperNode THNT ;
#else
typedef Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::Threads, Kokkos::HostSpace> THNT;
#endif
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
#if (TRILINOS_MAJOR_VERSION < 15)
typedef Kokkos::Compat::KokkosCudaWrapperNode CudaNT ;
#else
typedef Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::Cuda, Kokkos::HostSpace> CudaNT;
#endif
typedef Tpetra::MultiVector<double,int,int,CudaNT> CudaMV;
typedef Tpetra::Operator<double,int,int,CudaNT> CudaOP;

template<>
RCP<Belos::SolverManager<double, CudaMV, CudaOP> >
belos_solver_create(std::string const& solverName, Teuchos::RCP<Teuchos::ParameterList> const& solver_parameters)
{
  return private_belos_solver_create<double,CudaMV,CudaOP>(solverName,solver_parameters) ;
}
#endif
