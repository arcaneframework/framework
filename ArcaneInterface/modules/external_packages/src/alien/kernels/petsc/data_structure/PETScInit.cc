/*
 * PETScInit.cc
 *
 *  Created on: 27 mai 2015
 *      Author: chevalic
 */

#include "PETScInit.h"

#include <petscsys.h>

namespace {
bool
checkInit()
{
#if ((PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR >= 3) || (PETSC_VERSION_MAJOR > 3))
  PetscBool b;
#else
  PetscTruth b;
#endif
  PetscInitialized(&b);
  return (b);
}
};

namespace Alien::PETScInternal {

bool
initPETSc(int* argc, char*** argv)
{
  if (checkInit())
    return true;

  PetscInitialize(argc, argv, nullptr, "PETSc Initialisation");
  // Reduce memory due to log for graphical viewer
  PetscLogActions(PETSC_FALSE);
  PetscLogObjects(PETSC_FALSE);

  return true;
}

bool
initPETSc()
{
  if (checkInit())
    return true;

  PetscInitializeNoArguments();
  // Reduce memory due to log for graphical viewer
  PetscLogActions(PETSC_FALSE);
  PetscLogObjects(PETSC_FALSE);

  return true;
}

} // namespace Alien::PETScInternal
