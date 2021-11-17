/*
 * PETScInit.h
 *
 *  Created on: 27 mai 2015
 *      Author: chevalic
 */

#ifndef ALIEN_KERNELS_PETSC_DATASTRUCTURE_PETSCINIT_H_
#define ALIEN_KERNELS_PETSC_DATASTRUCTURE_PETSCINIT_H_

#include <alien/kernels/petsc/PETScPrecomp.h>

namespace Alien::PETScInternal {

bool initPETSc(int* argc, char*** argv);
bool initPETSc();

} // namespace Alien::PETScInternal

#endif /* ALIEN_KERNELS_PETSC_DATASTRUCTURE_PETSCINIT_H_ */
