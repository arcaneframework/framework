// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#ifndef PETSCCONFIG_H_
#define PETSCCONFIG_H_

#include <alien/utils/ObjectWithTrace.h>
#include <alien/kernels/petsc/linear_solver/IPETScPC.h>

namespace Alien {

class PETScConfig : protected ObjectWithTrace
{
 public:
  PETScConfig(bool is_parallel)
  : m_is_parallel(is_parallel)
  {
  }

  bool isParallel() const { return m_is_parallel; }

  //! Explains error
  void checkError(const Arccore::String& msg, int ierr) const
  {
    if (ierr != 0) {
      const char* text;
#if PETSC_VERSION_GE(3, 25, 0)
      const char* specific;
#else
      char* specific;
#endif
      PetscErrorMessage(ierr, &text, &specific);
      alien_fatal([&] {
        cout() << msg << " failed : " << text << " / " << specific << "[code=" << ierr
               << "]";
      });
    }
  }

  void init(){};

  bool m_is_parallel;
};

} // namespace Alien

#endif /* PETSCCONFIG_H_ */
