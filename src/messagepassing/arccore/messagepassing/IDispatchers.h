/*---------------------------------------------------------------------------*/
/* IDispatchers.h                                              (C) 2000-2018 */
/*                                                                           */
/* Interface du conteneur des dispatchers.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_IDISPATCHERS_H
#define ARCCORE_MESSAGEPASSING_IDISPATCHERS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/messagepassing/MessagePassingGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
namespace MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du conteneur des dispatchers.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IDispatchers
{
 public:

  virtual ~IDispatchers(){}

 public:

  virtual ITypeDispatcher<char>* dispatcher(char*) =0;
  virtual ITypeDispatcher<signed char>* dispatcher(signed char*) =0;
  virtual ITypeDispatcher<unsigned char>* dispatcher(unsigned char*) =0;
  virtual ITypeDispatcher<short>* dispatcher(short*) =0;
  virtual ITypeDispatcher<unsigned short>* dispatcher(unsigned short*) =0;
  virtual ITypeDispatcher<int>* dispatcher(int*) =0;
  virtual ITypeDispatcher<unsigned int>* dispatcher(unsigned int*) =0;
  virtual ITypeDispatcher<long>* dispatcher(long*) =0;
  virtual ITypeDispatcher<unsigned long>* dispatcher(unsigned long*) =0;
  virtual ITypeDispatcher<long long>* dispatcher(long long*) =0;
  virtual ITypeDispatcher<unsigned long long>* dispatcher(unsigned long long*) =0;
  virtual ITypeDispatcher<float>* dispatcher(float*) =0;
  virtual ITypeDispatcher<double>* dispatcher(double*) =0;
  virtual ITypeDispatcher<long double>* dispatcher(long double*) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
