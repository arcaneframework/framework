/*---------------------------------------------------------------------------*/
/* Dispatchers.h                                               (C) 2000-2018 */
/*                                                                           */
/* Conteneur des dispatchers.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_DISPATCHERS_H
#define ARCCORE_MESSAGEPASSING_DISPATCHERS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/IDispatchers.h"

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
class ARCCORE_MESSAGEPASSING_EXPORT Dispatchers
: public IDispatchers
{
 public:

  Dispatchers();
  ~Dispatchers() override;

 public:

  ITypeDispatcher<char>* dispatcher(char*) override { return m_char; }
  ITypeDispatcher<signed char>* dispatcher(signed char*) override { return  m_signed_char; }
  ITypeDispatcher<unsigned char>* dispatcher(unsigned char*) override { return m_unsigned_char; }
  ITypeDispatcher<short>* dispatcher(short*) override{ return  m_short; }
  ITypeDispatcher<unsigned short>* dispatcher(unsigned short*) override { return m_unsigned_short; }
  ITypeDispatcher<int>* dispatcher(int*) override { return m_int; }
  ITypeDispatcher<unsigned int>* dispatcher(unsigned int*) override { return m_unsigned_int; }
  ITypeDispatcher<long>* dispatcher(long*) override { return m_long; }
  ITypeDispatcher<unsigned long>* dispatcher(unsigned long*) override { return m_unsigned_long; }
  ITypeDispatcher<long long>* dispatcher(long long*) override { return m_long_long; }
  ITypeDispatcher<unsigned long long>* dispatcher(unsigned long long*) override { return m_unsigned_long_long; }
  ITypeDispatcher<float>* dispatcher(float*) override { return m_float; }
  ITypeDispatcher<double>* dispatcher(double*) override { return m_double; }
  ITypeDispatcher<long double>* dispatcher(long double*) override { return m_long_double; }

 public:

  void setDispatcher(ITypeDispatcher<char>* x) { m_char = x; }
  void setDispatcher(ITypeDispatcher<signed char>* x) { m_signed_char = x; }
  void setDispatcher(ITypeDispatcher<unsigned char>* x) { m_unsigned_char = x; }
  void setDispatcher(ITypeDispatcher<short>* x) { m_short = x; }
  void setDispatcher(ITypeDispatcher<unsigned short>* x) { m_unsigned_short = x; }
  void setDispatcher(ITypeDispatcher<int>* x) { m_int = x; }
  void setDispatcher(ITypeDispatcher<unsigned int>* x) { m_unsigned_int = x; }
  void setDispatcher(ITypeDispatcher<long>* x) { m_long = x; }
  void setDispatcher(ITypeDispatcher<unsigned long>* x) { m_unsigned_long = x; }
  void setDispatcher(ITypeDispatcher<long long>* x) { m_long_long = x; }
  void setDispatcher(ITypeDispatcher<unsigned long long>* x) { m_unsigned_long_long = x; }
  void setDispatcher(ITypeDispatcher<float>* x) { m_float = x; }
  void setDispatcher(ITypeDispatcher<double>* x) { m_double = x; }
  void setDispatcher(ITypeDispatcher<long double>* x) { m_long_double = x; }

  //! Indique si lors de la destruction on appelle l'opérateur delete sur les instances (faux par défaut)
  void setDeleteDispatchers(bool v) { m_is_delete_dispatchers = v; }

 private:

  ITypeDispatcher<char>* m_char;
  ITypeDispatcher<unsigned char>* m_unsigned_char;
  ITypeDispatcher<signed char>* m_signed_char;
  ITypeDispatcher<short>* m_short;
  ITypeDispatcher<unsigned short>* m_unsigned_short;
  ITypeDispatcher<int>* m_int;
  ITypeDispatcher<unsigned int>* m_unsigned_int;
  ITypeDispatcher<long>* m_long;
  ITypeDispatcher<unsigned long>* m_unsigned_long;
  ITypeDispatcher<long long>* m_long_long;
  ITypeDispatcher<unsigned long long>* m_unsigned_long_long;
  ITypeDispatcher<float>* m_float;
  ITypeDispatcher<double>* m_double;
  ITypeDispatcher<long double>* m_long_double;

  bool m_is_delete_dispatchers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
