/*---------------------------------------------------------------------------*/
/* FatalErrorException.cc                                      (C) 2000-2018 */
/*                                                                           */
/* Exception lorsqu'une erreur fatale est survenue.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/base/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FatalErrorException::
FatalErrorException(const String& awhere)
: Exception("FatalError",awhere)
{
  arccoreDebugPause("FatalError");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FatalErrorException::
FatalErrorException(const String& awhere,const String& amessage)
: Exception("FatalError",awhere,amessage)
{
  arccoreDebugPause("FatalError");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FatalErrorException::
FatalErrorException(const TraceInfo& awhere)
: Exception("FatalError",awhere)
{
  arccoreDebugPause("FatalError");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FatalErrorException::
FatalErrorException(const TraceInfo& awhere,const String& amessage)
: Exception("FatalError",awhere,amessage)
{
  arccoreDebugPause("FatalError");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FatalErrorException::
FatalErrorException(const FatalErrorException& rhs) ARCCORE_NOEXCEPT
: Exception(rhs)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FatalErrorException::
explain(std::ostream& m) const
{
  m << "Fatal error occured.\n"
    << "Can not further proceed.\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

