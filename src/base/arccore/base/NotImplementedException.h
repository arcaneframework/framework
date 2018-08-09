/*---------------------------------------------------------------------------*/
/* NotImplementedException.h                                   (C) 2000-2018 */
/*                                                                           */
/* Exception lorsqu'une fonction n'est pas implémentée.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_NOTIMPLEMENTEDEXCEPTION_H
#define ARCCORE_BASE_NOTIMPLEMENTEDEXCEPTION_H
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
 * \internal
 * \brief Exception lorsqu'une fonction n'est pas implémentée.
 */
class ARCCORE_BASE_EXPORT NotImplementedException
: public Exception
{
 public:
	
  explicit NotImplementedException(const String& where);
  NotImplementedException(const String& where,const String& message);
  explicit NotImplementedException(const TraceInfo& where);
  NotImplementedException(const TraceInfo& where,const String& message);
  NotImplementedException(const NotImplementedException& rhs) ARCCORE_NOEXCEPT;
  ~NotImplementedException() ARCCORE_NOEXCEPT {}

 public:
	
  virtual void explain(std::ostream& m) const;

 private:

  String m_message;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

