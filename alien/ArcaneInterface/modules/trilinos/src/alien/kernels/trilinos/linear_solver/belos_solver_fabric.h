// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

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
belos_solver_create(std::string const& solverName, Teuchos::RCP<Teuchos::ParameterList> const& solver_parameters) ;
