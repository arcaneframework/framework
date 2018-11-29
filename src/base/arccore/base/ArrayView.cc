/*---------------------------------------------------------------------------*/
/* ArrayView.cc                                                (C) 2000-2018 */
/*                                                                           */
/* Déclarations générales de Arccore.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/ArgumentException.h"
#include "arccore/base/TraceInfo.h"

// On n'utilise pas directement ces fichiers mais on les inclus pour tester
// la compilation. Lorsque les tests seront en place on pourra supprmer
// ces inclusions
#include "arccore/base/Array2View.h"
#include "arccore/base/Array3View.h"
#include "arccore/base/Array4View.h"
#include "arccore/base/Span.h"
#include "arccore/base/Span2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(unsigned long long size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    ARCCORE_THROW(ArgumentException,"value '{0}' too big for Array size",size);
  return (Integer)size;
}

extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(long long size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    ARCCORE_THROW(ArgumentException,"value '{0}' too big for Array size",size);
  if (size<0)
    ARCCORE_THROW(ArgumentException,"invalid negative value '{0}' for Array size",size);
  return (Integer)size;
}

extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(unsigned long size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    ARCCORE_THROW(ArgumentException,"value '{0}' too big for Array size",size);
  return (Integer)size;
}

extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(long size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    ARCCORE_THROW(ArgumentException,"value '{0}' too big for Array size",size);
  if (size<0)
    ARCCORE_THROW(ArgumentException,"invalid negative value '{0}' for Array size",size);
  return (Integer)size;
}

extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(unsigned int size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    ARCCORE_THROW(ArgumentException,"value '{0}' too big for Array size",size);
  return (Integer)size;
}

extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(int size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    ARCCORE_THROW(ArgumentException,"value '{0}' too big for Array size",size);
  if (size<0)
    ARCCORE_THROW(ArgumentException,"invalid negative value '{0}' for Array size",size);
  return (Integer)size;
}

extern "C++" ARCCORE_BASE_EXPORT Int64
arccoreCheckLargeArraySize(size_t size)
{
  if (size>=ARCCORE_INT64_MAX)
    ARCCORE_THROW(ArgumentException,"value '{0}' too big to fit in Int64",size);
  return (Int64)size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
