/*---------------------------------------------------------------------------*/
/* GlibAdapter.h                                               (C) 2000-2018 */
/*                                                                           */
/* Classes utilitaires pour s'adapter aux différentes versions de la 'glib'. */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_GLIBADAPTER_H
#define ARCCORE_CONCURRENCY_GLIBADAPTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Encapsule un GMutex de la glib.
 */
class ARCCORE_CONCURRENCY_EXPORT GlibMutex
{
 public:
  class Impl;
 public:
  class Lock
  {
   public:
    Lock(GlibMutex& x);
    ~Lock();
    Lock() = delete;
    Lock(const Lock&) = delete;
    void operator=(const Lock&) = delete;
   private:
    Impl* m_mutex;
  };
 public:
  GlibMutex() ARCCORE_NOEXCEPT;
  ~GlibMutex();
 public:
  void lock();
  void unlock();
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Encapsule un GPrivate de la glib.
 */
class ARCCORE_CONCURRENCY_EXPORT GlibPrivate
{
 public:
  class Impl;
 public:
  GlibPrivate();
  ~GlibPrivate();
  void create();
  void setValue(void* value);
  void* getValue();
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
