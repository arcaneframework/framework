/*---------------------------------------------------------------------------*/
/* ArgumentException.h                                         (C) 2000-2018 */
/*                                                                           */
/* Exception lorsqu'un argument est invalide.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARGUMENTEXCEPTION_H
#define ARCCORE_BASE_ARGUMENTEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Core
 * \brief Exception lorsqu'un argument est invalide.
 */
class ARCCORE_BASE_EXPORT ArgumentException
: public Exception
{
 public:
	
  explicit ArgumentException(const String& where);
  ArgumentException(const String& where,const String& message);
  explicit ArgumentException(const TraceInfo& where);
  ArgumentException(const TraceInfo& where,const String& message);
  ArgumentException(const ArgumentException& rhs) ARCCORE_NOEXCEPT;
  ~ArgumentException() ARCCORE_NOEXCEPT override;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arrcore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

