// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

/*! \brief Basic interface for builder
 *  Due to visitor techniques, all implementations are described below
 *  with a default implementation ("not implemented" error).
 *  When available, inherited class implements this interface (without error if not)
 */

// D�finit s'il faut controler le profil lors du remplissage (acc�s couteux)
#define CHECKPROFILE_ON_DEBUG_FILLING

#include "alien/local_direct_solvers/ILinearSystemVisitor.h"

namespace Alien {

class ILinearSystem ;
class LocalLinearSystem ;


#if not defined(NDEBUG) && defined(CHECKPROFILE_ON_DEBUG_FILLING)
#define CHECKPROFILE_ON_FILLING
#endif /* NDEBUG */


class ILinearSystemBuilder : public ILinearSystemVisitor
{
public:
  /** Constructeur de la classe */
  ILinearSystemBuilder()
    {
      ;
    }

  /** Destructeur de la classe */
  virtual ~ILinearSystemBuilder() { }

  virtual void start() = 0 ;
  virtual void freeData() = 0 ;
  virtual void end() = 0 ;

  virtual bool connect       (ILinearSystem * system) { throw Arccore:: FatalErrorException(A_FUNCINFO,"not implemented"); }
  virtual bool commitSolution(ILinearSystem * system) { throw Arccore::FatalErrorException(A_FUNCINFO,"not implemented"); }

  virtual bool connect       (LocalLinearSystem* system) { throw Arccore::FatalErrorException(A_FUNCINFO,"not implemented"); }
  virtual bool commitSolution(LocalLinearSystem* system) { throw Arccore::FatalErrorException(A_FUNCINFO,"not implemented"); }

};
}

