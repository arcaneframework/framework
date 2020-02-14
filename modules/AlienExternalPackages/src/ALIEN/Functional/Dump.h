// -*- C++ -*-
#ifndef ALIEN_IFPEN_FUNCTIONAL_DUMP_H
#define ALIEN_IFPEN_FUNCTIONAL_DUMP_H

#include <ALIEN/Utils/Precomp.h>
#include <ALIEN/Alien-ExternalPackagesPrecomp.h>
#include <ALIEN/Kernels/PETSc/IO/AsciiDumper.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMatrix;
class IVector;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Dump on screen with default matlab mode

// WARNING : should induce matrix conversion to PETSc format

ALIEN_EXTERNALPACKAGES_EXPORT extern void dump(const IMatrix& a, const AsciiDumper::Style style = AsciiDumper::Style::eDefaultStyle);
ALIEN_EXTERNALPACKAGES_EXPORT extern void dump(const IVector& a, const AsciiDumper::Style style = AsciiDumper::Style::eDefaultStyle);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_IFPEN_FUNCTIONAL_DUMP_H */
