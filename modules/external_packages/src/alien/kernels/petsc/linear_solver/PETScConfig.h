/*
 * PETScConfig.h
 *
 *  Created on: 13 janv. 2015
 *      Author: gratienj
 */

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
      char* specific;
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
