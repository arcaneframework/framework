/*---------------------------------------------------------------------------*/
/* ArrayView.cc                                                (C) 2000-2018 */
/*                                                                           */
/* Déclarations générales de Arccore.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

#if ARCCORE_NEED_IMPLEMENTATION
#else
#undef ARCCORE_THROW
#define ARCCORE_THROW(exception_class,...) throw std::exception()
#endif

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
